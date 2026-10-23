from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from runtime_benchmark_common import RuntimeProtocol, module_parameter_counts, run_runtime_protocol


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Porto runtime benchmark for ControlTraj.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--road-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default=ROOT / "ControlTraj/data/porto", type=Path)
    parser.add_argument("--canonical-real", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--throughput-samples", type=int, default=9984)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--throughput-warmup-batches", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.road_checkpoint, args.canonical_real):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the unified benchmark")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    project = ROOT / "ControlTraj"
    sys.path.insert(0, str(project))
    import inference_controltraj as module

    if module.DEVICE.type != device.type:
        raise RuntimeError(f"ControlTraj configured device {module.DEVICE}, requested {device}")
    protocol = RuntimeProtocol(
        latency_samples=args.latency_samples,
        throughput_samples=args.throughput_samples,
        throughput_batch_size=args.throughput_batch_size,
        repeats=args.repeats,
        latency_warmup=args.latency_warmup,
        throughput_warmup_batches=args.throughput_warmup_batches,
    )
    with np.load(args.canonical_real, allow_pickle=False) as payload:
        selected = np.asarray(
            payload["sample_id"][: max(protocol.latency_samples, protocol.throughput_samples)],
            dtype=np.int64,
        )
    trajectories_np = np.load(args.data_root / "porto_trajs.npy", mmap_mode="r")
    attributes_np = np.load(args.data_root / "porto_heads.npy", mmap_mode="r")
    roads_np = np.load(args.data_root / "porto_roads.npy", mmap_mode="r")
    trajectories = torch.from_numpy(np.asarray(trajectories_np[selected])).float()
    if trajectories.ndim == 3 and trajectories.shape[1] != 2:
        trajectories = trajectories.permute(0, 2, 1)
    attributes = torch.from_numpy(np.asarray(attributes_np[selected])).float()
    roads = torch.from_numpy(np.asarray(roads_np[selected])).float()
    road_encoder, model = module.init_models()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    road_state = torch.load(args.road_checkpoint, map_location=device, weights_only=False)
    road_encoder.load_state_dict(road_state.get("model_state_dict", road_state))
    model.eval()
    road_encoder.eval()
    processor = module.BaselineRoadProcessor(device)
    beta = torch.linspace(
        float(module.config.diffusion.beta_start),
        float(module.config.diffusion.beta_end),
        int(module.config.diffusion.num_diffusion_timesteps),
        device=device,
    )
    sampling = module.make_sampling_func(
        beta,
        smoothing_sigma=float(args.smoothing_sigma),
        sampling_steps=int(args.sampling_steps),
    )

    def infer(start: int, count: int):
        x0 = trajectories[start : start + count].to(device, non_blocking=True)
        attr = attributes[start : start + count].to(device, non_blocking=True)
        road = roads[start : start + count].to(device, non_blocking=True)
        road_input = processor.prepare_road_mae_input(road)
        road_embedding = processor.extract_road_features(road_encoder, road_input, no_mask=True)
        noise = torch.randn_like(x0)
        return sampling(model, noise, road_embedding, attr, road)

    total, trainable = module_parameter_counts([model, road_encoder])
    run_runtime_protocol(
        method="ControlTraj",
        infer=infer,
        device=device,
        output=args.output,
        total_parameters=total,
        trainable_parameters=trainable,
        protocol=protocol,
        metadata={
            "checkpoint": str(args.checkpoint.resolve()),
            "road_checkpoint": str(args.road_checkpoint.resolve()),
            "dataset": "porto",
            "split": "test",
            "sample_ids_source": str(args.canonical_real.resolve()),
            "steps_or_k": f"S={args.sampling_steps}",
            "sampling_steps": args.sampling_steps,
            "smoothing_sigma": args.smoothing_sigma,
            "inference_boundary": "OD_and_road_condition_to_coordinates_including_road_encoding",
        },
    )


if __name__ == "__main__":
    main()
