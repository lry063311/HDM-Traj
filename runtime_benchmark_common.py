from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeProtocol:
    latency_samples: int = 1000
    throughput_samples: int = 9984
    throughput_batch_size: int = 128
    repeats: int = 5
    latency_warmup: int = 20
    throughput_warmup_batches: int = 5


def module_parameter_counts(modules: Iterable[torch.nn.Module]) -> tuple[int, int]:
    seen: set[int] = set()
    total = 0
    trainable = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            count = int(parameter.numel())
            total += count
            if parameter.requires_grad:
                trainable += count
    return total, trainable


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def run_runtime_protocol(
    *,
    method: str,
    infer: Callable[[int, int], object],
    device: torch.device,
    output: Path,
    total_parameters: int,
    trainable_parameters: int,
    aux_storage_bytes: int = 0,
    protocol: RuntimeProtocol = RuntimeProtocol(),
    metadata: dict | None = None,
) -> dict:
    required = max(protocol.latency_samples, protocol.throughput_samples)
    if required < 1:
        raise ValueError("benchmark sample counts must be positive")

    with torch.inference_mode():
        print(
            f"[{method}] warm-up: B=1 x {protocol.latency_warmup}, "
            f"B={protocol.throughput_batch_size} x {protocol.throughput_warmup_batches}",
            flush=True,
        )
        for index in range(protocol.latency_warmup):
            infer(index % required, 1)
        for batch_index in range(protocol.throughput_warmup_batches):
            start = (batch_index * protocol.throughput_batch_size) % required
            infer(start, protocol.throughput_batch_size)
        _sync(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        latency_runs = []
        for run_index in range(protocol.repeats):
            print(
                f"[{method}] latency run {run_index + 1}/{protocol.repeats}: "
                f"{protocol.latency_samples} requests",
                flush=True,
            )
            latencies_ms: list[float] = []
            for index in range(protocol.latency_samples):
                _sync(device)
                started = time.perf_counter()
                infer(index, 1)
                _sync(device)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
            latency_runs.append(
                {
                    "median_ms_per_trajectory": float(statistics.median(latencies_ms)),
                    "p95_ms_per_trajectory": _percentile(latencies_ms, 95.0),
                    "mean_ms_per_trajectory": float(statistics.fmean(latencies_ms)),
                }
            )
            print(
                f"[{method}] latency run {run_index + 1} complete: "
                f"median={latency_runs[-1]['median_ms_per_trajectory']:.3f} ms, "
                f"p95={latency_runs[-1]['p95_ms_per_trajectory']:.3f} ms",
                flush=True,
            )

        throughput_runs = []
        batch_size = protocol.throughput_batch_size
        if protocol.throughput_samples % batch_size != 0:
            raise ValueError("throughput_samples must be divisible by throughput_batch_size")
        for run_index in range(protocol.repeats):
            print(
                f"[{method}] throughput run {run_index + 1}/{protocol.repeats}: "
                f"{protocol.throughput_samples} samples at B={batch_size}",
                flush=True,
            )
            _sync(device)
            started = time.perf_counter()
            for start in range(0, protocol.throughput_samples, batch_size):
                infer(start, batch_size)
            _sync(device)
            elapsed = time.perf_counter() - started
            throughput_runs.append(
                {
                    "elapsed_seconds": float(elapsed),
                    "trajectories_per_second": float(protocol.throughput_samples / elapsed),
                }
            )
            print(
                f"[{method}] throughput run {run_index + 1} complete: "
                f"{throughput_runs[-1]['trajectories_per_second']:.2f} traj/s",
                flush=True,
            )

    peak_allocated = 0
    peak_reserved = 0
    gpu_name = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        gpu_name = torch.cuda.get_device_name(device)

    result = {
        "method": method,
        "protocol": {
            "latency_batch_size": 1,
            "latency_samples_per_run": protocol.latency_samples,
            "throughput_batch_size": protocol.throughput_batch_size,
            "throughput_samples_per_run": protocol.throughput_samples,
            "repeats": protocol.repeats,
            "latency_warmup_requests": protocol.latency_warmup,
            "throughput_warmup_batches": protocol.throughput_warmup_batches,
            "cuda_sync_latency": "before_and_after_each_request",
            "cuda_sync_throughput": "once_before_and_once_after_timed_loop",
            "sample_selection": "first_N_ordered_porto_test_samples",
        },
        "latency": {
            "median_ms_per_trajectory": float(
                statistics.median(row["median_ms_per_trajectory"] for row in latency_runs)
            ),
            "p95_ms_per_trajectory": float(
                statistics.median(row["p95_ms_per_trajectory"] for row in latency_runs)
            ),
            "runs": latency_runs,
        },
        "throughput": {
            "median_trajectories_per_second": float(
                statistics.median(row["trajectories_per_second"] for row in throughput_runs)
            ),
            "runs": throughput_runs,
        },
        "parameters": {
            "total": int(total_parameters),
            "trainable": int(trainable_parameters),
            "total_millions": float(total_parameters / 1e6),
            "trainable_millions": float(trainable_parameters / 1e6),
        },
        "memory": {
            "aux_storage_bytes": int(aux_storage_bytes),
            "aux_storage_mb": float(aux_storage_bytes / (1024**2)),
            "peak_gpu_allocated_bytes": int(peak_allocated),
            "peak_gpu_allocated_gb": float(peak_allocated / (1024**3)),
            "peak_gpu_reserved_bytes_audit": int(peak_reserved),
        },
        "environment": {
            "device": str(device),
            "gpu": gpu_name,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "metadata": metadata or {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    gc.collect()
    return result
