#!/usr/bin/env python
"""Eval-only ControlTraj sampling on LangDirectTraj instruction splits.

This wrapper avoids the original ControlTraj training entry point. It loads a
saved UNet checkpoint, samples trajectories for the sample_id order in an
instruction jsonl split, saves generated/real arrays in one NPZ, and computes
the same Table I distribution metrics used by LangDirectTraj.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import evaluate_distribution  # noqa: E402


def dict_to_sns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: dict_to_sns(v) for k, v in value.items()})
    return value


def add_controltraj_imports(files_dir: Path):
    files_dir = files_dir.resolve()
    if str(files_dir) not in sys.path:
        sys.path.insert(0, str(files_dir))
    from utils.config import args as controltraj_args  # noqa: PLC0415
    from utils.geounet import UNetModel  # noqa: PLC0415
    from utils.road_encoder import MAE_ViT  # noqa: PLC0415

    return dict_to_sns(controltraj_args), UNetModel, MAE_ViT


def gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)


def p_xt(xt: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, next_t: torch.Tensor, beta: torch.Tensor, eta: float = 0):
    at = compute_alpha(beta, t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    c2 = ((1 - at_next) - c1**2).sqrt()
    eps = torch.randn(xt.shape, device=xt.device)
    return at_next.sqrt() * x0_t + c1 * eps + c2 * noise


def resample_trajectory(x: np.ndarray, length: int = 200) -> np.ndarray:
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    # Match the original ControlTraj BaselineRoadProcessor: RoadMAE consumes
    # two coordinate channels even when road segments are stored as [x1,y1,x2,y2].
    resampled = np.zeros((2, length), dtype=np.float32)
    for i in range(2):
        resampled[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled.T


def denormalize_trajs(x: torch.Tensor, stats: dict) -> np.ndarray:
    out = x.detach().clone()
    out[:, 0, :] = out[:, 0, :] * stats["lon_std"] + stats["lon_mean"]
    out[:, 1, :] = out[:, 1, :] * stats["lat_std"] + stats["lat_mean"]
    out[:, 0, :] = torch.clamp(out[:, 0, :], -180.0, 180.0)
    out[:, 1, :] = torch.clamp(out[:, 1, :], -90.0, 90.0)
    return out.cpu().numpy().transpose(0, 2, 1).astype(np.float32)


def load_instruction_split(input_path: Path, compact_base: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    sample_ids: list[int] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_samples > 0 and len(rows) >= max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            refs = record.get("compact_refs", {})
            traj_ref = refs.get("trajectory", {})
            row = traj_ref.get("row", record.get("base_index"))
            if row is None:
                raise ValueError(f"Missing compact trajectory row at line {line_idx}: {input_path}")
            rows.append(int(row))
            sample_ids.append(int(record.get("sample_id", line_idx)))

    if not rows:
        raise ValueError(f"No records loaded from {input_path}")
    base = np.load(compact_base, mmap_mode="r")
    real = np.asarray(base[np.asarray(rows, dtype=np.int64)], dtype=np.float32)
    return np.asarray(sample_ids, dtype=np.int64), np.asarray(rows, dtype=np.int64), real


def load_state_dict(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict", "unet"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return payload


@torch.no_grad()
def sample_controltraj(
    *,
    unet,
    road_encoder,
    loader: DataLoader,
    stats: dict,
    beta: torch.Tensor,
    device: torch.device,
    target_timesteps: int,
) -> np.ndarray:
    unet.eval()
    road_encoder.eval()
    n_steps = int(beta.numel())
    skip = max(1, n_steps // int(target_timesteps))
    seq = range(0, n_steps, skip)
    seq_next = [-1] + list(seq[:-1])
    generated = []

    for x0, attr, road in tqdm(loader, desc="controltraj sampling"):
        x0 = x0.to(device)
        attr = attr.to(device)
        road = road.to(device)

        roads_200 = np.stack([resample_trajectory(r.cpu().numpy(), length=200) for r in road], axis=0).transpose(0, 2, 1)
        road_in = torch.from_numpy(roads_200).float().to(device)
        guide, _ = road_encoder.encoder(road_in)
        guide = rearrange(guide[1:, :, :], "t b c -> b t c")

        x_t = torch.randn_like(x0)
        for t_idx, t_next_idx in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((x_t.size(0),), int(t_idx), device=device)
            next_t = torch.full((x_t.size(0),), int(t_next_idx), device=device)
            pred_noise = unet(x_t.float(), t, guide, attr)
            x_t = p_xt(x_t, pred_noise, t, next_t, beta)
        x_t = torch.clamp(x_t, -5.0, 5.0)
        generated.append(denormalize_trajs(x_t, stats))

    return np.concatenate(generated, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controltraj-files-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compact-base", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config, UNetModel, MAE_ViT = add_controltraj_imports(args.controltraj_files_dir)
    dataset = args.dataset.lower()
    stats_path = args.data_root / f"{dataset}_global_stats.json"
    traj_path = args.data_root / f"{dataset}_trajs.npy"
    road_path = args.data_root / f"{dataset}_roads.npy"
    head_path = args.data_root / f"{dataset}_heads.npy"
    compact_base = args.compact_base or (args.input.parent / f"{dataset}_base_lonlat.npy")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    sample_ids, compact_rows, real = load_instruction_split(args.input, compact_base, args.max_samples)
    trajs = np.load(traj_path, mmap_mode="r")
    roads = np.load(road_path, mmap_mode="r")
    heads = np.load(head_path, mmap_mode="r")
    if int(sample_ids.max()) >= len(trajs):
        raise ValueError(f"sample_id max {int(sample_ids.max())} exceeds ControlTraj data length {len(trajs)}")

    x = np.asarray(trajs[sample_ids], dtype=np.float32)
    attr = np.asarray(heads[sample_ids], dtype=np.float32)
    road = np.asarray(roads[sample_ids], dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(attr).float(), torch.from_numpy(road).float()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    unet = UNetModel(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        n_res_blocks=config.model.num_res_blocks,
        attention_levels=config.model.attention_levels,
        channel_multipliers=config.model.channel_multipliers,
        n_heads=config.model.n_heads,
        tf_layers=config.model.tf_layers,
        d_cond=128,
    ).to(device)
    unet.load_state_dict(load_state_dict(args.checkpoint, device), strict=True)

    road_encoder = MAE_ViT(
        image_size=200,
        patch_size=5,
        emb_dim=128,
        encoder_layer=8,
        encoder_head=4,
        decoder_layer=4,
        decoder_head=4,
        mask_ratio=0.00,
    ).to(device)
    road_encoder.load_state_dict(load_state_dict(args.road_checkpoint, device), strict=True)

    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.num_diffusion_timesteps).to(device)
    generated = sample_controltraj(
        unet=unet,
        road_encoder=road_encoder,
        loader=loader,
        stats=stats,
        beta=beta,
        device=device,
        target_timesteps=args.target_timesteps,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        generated=generated.astype(np.float32),
        trajectories=generated.astype(np.float32),
        real=real.astype(np.float32),
        sample_ids=sample_ids.astype(np.int64),
        compact_rows=compact_rows.astype(np.int64),
    )
    metrics = evaluate_distribution(real, generated)
    metrics.update(
        {
            "selector": "controltraj_unet150_resampled_on_langdirecttraj_split",
            "checkpoint": str(args.checkpoint),
            "road_checkpoint": str(args.road_checkpoint),
            "input": str(args.input),
            "output_npz": str(args.output_npz),
            "seed": int(args.seed),
            "target_timesteps": int(args.target_timesteps),
            "num_samples": int(len(real)),
        }
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
