from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from runtime_benchmark_common import RuntimeProtocol, module_parameter_counts, run_runtime_protocol


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Porto runtime benchmark for neural baselines.")
    parser.add_argument("--method", required=True, choices=("vae", "trajgan", "difftraj", "traj-transformer"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default=ROOT / "DiffTraj/data/porto_od_only", type=Path)
    parser.add_argument("--canonical-real", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--sampling-stride", type=int, default=5)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--throughput-samples", type=int, default=9984)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--throughput-warmup-batches", type=int, default=5)
    return parser.parse_args()


def add_project_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_canonical_ids(test_indices: np.ndarray, canonical_path: Path) -> None:
    with np.load(canonical_path, allow_pickle=False) as payload:
        canonical_ids = np.asarray(payload["sample_id"], dtype=np.int64)
    if not np.array_equal(np.asarray(test_indices, dtype=np.int64), canonical_ids):
        raise ValueError("ordered test indices do not match the canonical Porto sample IDs")


def load_vae_or_gan(args: argparse.Namespace, device: torch.device):
    project = ROOT / ("VAE" if args.method == "vae" else "TrajGAN")
    add_project_path(project)
    data_module = __import__("data")
    model_module = __import__("model")
    info = data_module.resolve_dataset(args.data_root.resolve(), "porto")
    attributes = np.load(info["paths"]["attributes"], mmap_mode="r", allow_pickle=False)
    test_indices = np.asarray(info["splits"]["test"], dtype=np.int64)
    assert_canonical_ids(test_indices, args.canonical_real)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.method == "vae":
        model = model_module.ConditionalTrajectoryVAE(
            info["shape"][2], latent_dim=int(checkpoint["config"]["latent_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        method_name = "VAE"
    else:
        model = model_module.TrajectoryGenerator(
            info["shape"][2], noise_dim=int(checkpoint["config"]["noise_dim"])
        ).to(device)
        model.load_state_dict(checkpoint["generator_state_dict"])
        method_name = "TrajGAN"
    model.eval()

    def infer(start: int, count: int):
        rows = np.asarray(attributes[test_indices[start : start + count]], dtype=np.float32)
        condition = torch.from_numpy(rows).to(device, non_blocking=True)
        return model.sample(condition)

    total, trainable = module_parameter_counts([model])
    return method_name, infer, total, trainable, {"steps_or_k": None, "inference_boundary": "condition_to_coordinates"}


def load_difftraj(args: argparse.Namespace, device: torch.device):
    project = ROOT / "DiffTraj"
    add_project_path(project)
    import evaluate_table2 as module

    trajectory_path = args.data_root / "porto_trajs.npy"
    attribute_path = args.data_root / "porto_heads.npy"
    trajectories = module.normalize_trajectory_shape(np.load(trajectory_path, mmap_mode="r", allow_pickle=False))
    splits = module.load_splits(args.data_root / "split_indices.npz", len(trajectories))
    config = module.build_config("porto", int(trajectories.shape[2]))
    attributes = module.normalize_attribute_shape(
        np.load(attribute_path, mmap_mode="r", allow_pickle=False), config.model.attr_dim
    )
    model = module.Guide_UNet(config).to(device)
    model.config = config
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    assert_canonical_ids(test_indices, args.canonical_real)

    def infer(start: int, count: int):
        rows = np.asarray(attributes[test_indices[start : start + count]], dtype=np.float32)
        condition = torch.from_numpy(rows).to(device, non_blocking=True)
        return module.sample_batch(
            model, condition, int(trajectories.shape[2]), args.sampling_steps, 0.0
        )

    total, trainable = module_parameter_counts([model])
    metadata = {
        "steps_or_k": f"S={args.sampling_steps}",
        "sampling_steps": args.sampling_steps,
        "smoothing_sigma": 0.0,
        "inference_boundary": "condition_to_unsmoothed_coordinates",
    }
    return "DiffTraj", infer, total, trainable, metadata


def load_traj_transformer(args: argparse.Namespace, device: torch.device):
    project = ROOT / "Traj-Transformer-master"
    add_project_path(project)
    import evaluate_table2 as module
    from table2_data import load_table2_data
    from table2_model import build_model, strip_state_prefixes

    data = load_table2_data(args.data_root, "porto", full_scan=False)
    test_indices = np.asarray(data.splits["test"], dtype=np.int64)
    assert_canonical_ids(test_indices, args.canonical_real)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["run_config"]
    model = build_model(data.trajectory_length, config["model_size"]).to(device)
    model.load_state_dict(strip_state_prefixes(checkpoint["model_state_dict"]))
    model.eval()
    heads = data.heads

    def infer(start: int, count: int):
        rows = np.asarray(heads[test_indices[start : start + count]], dtype=np.float32)
        condition = torch.from_numpy(rows).to(device, non_blocking=True)
        return module.sample_batch(
            model,
            condition,
            data.trajectory_length,
            int(config["diffusion_steps"]),
            args.sampling_stride,
            device,
            args.amp,
            args.seed + start,
        )

    total, trainable = module_parameter_counts([model])
    sampling_steps = int(config["diffusion_steps"]) // args.sampling_stride
    metadata = {
        "steps_or_k": f"S={sampling_steps}",
        "sampling_steps": sampling_steps,
        "sampling_stride": args.sampling_stride,
        "amp": args.amp,
        "inference_boundary": "OD_condition_to_coordinates",
    }
    return "Traj-Transformer", infer, total, trainable, metadata


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.canonical_real):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the unified benchmark")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if args.method in {"vae", "trajgan"}:
        loaded = load_vae_or_gan(args, device)
    elif args.method == "difftraj":
        loaded = load_difftraj(args, device)
    else:
        loaded = load_traj_transformer(args, device)
    method, infer, total, trainable, metadata = loaded
    metadata.update({"checkpoint": str(args.checkpoint.resolve()), "dataset": "porto", "split": "test"})
    protocol = RuntimeProtocol(
        latency_samples=args.latency_samples,
        throughput_samples=args.throughput_samples,
        throughput_batch_size=args.throughput_batch_size,
        repeats=args.repeats,
        latency_warmup=args.latency_warmup,
        throughput_warmup_batches=args.throughput_warmup_batches,
    )
    run_runtime_protocol(
        method=method,
        infer=infer,
        device=device,
        output=args.output,
        total_parameters=total,
        trainable_parameters=trainable,
        protocol=protocol,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
