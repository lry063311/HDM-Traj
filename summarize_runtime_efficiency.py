from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ORDER = [
    "VAE",
    "TrajGAN",
    "DiffTraj",
    "ControlTraj",
    "Traj-Transformer",
    "HDM-Traj Mode A",
    "HDM-Traj Mode B (ordinary)",
    "HDM-Traj Mode B (counterfactual)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge unified runtime benchmark JSON files.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Include only the methods present in the input dir; do not error on "
        "missing expected methods. Use for standalone Mode-A or Mode-B runs only; "
        "the final submission table should run without this flag so every expected "
        "result must be present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = []
    for path in sorted(args.input_dir.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("method") in ORDER:
            item["source_json"] = str(path.resolve())
            payloads.append(item)
    by_method = {item["method"]: item for item in payloads}
    expected = [method for method in ORDER if method in by_method]
    missing = [method for method in ORDER if method not in by_method]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"Missing runtime results: {missing}. Re-run the missing benchmarks, or pass "
            f"--allow-partial for a standalone partial table (not for the final submission)."
        )
    payloads = [by_method[method] for method in expected]
    protocol = payloads[0]["protocol"]
    for item in payloads[1:]:
        if item["protocol"] != protocol:
            raise SystemExit(f"Protocol mismatch for {item['method']}: expected {protocol}, got {item['protocol']}")

    rows = []
    for item in payloads:
        meta = item.get("metadata", {})
        throughput = item["throughput"]["median_trajectories_per_second"]
        if meta.get("throughput_batched") is False:
            # Inherently sequential inference (e.g. Mode B counterfactual pairs):
            # the measured number is single-stream, not a genuine B=128 batch,
            # so leave it blank in the table rather than compare it side-by-side.
            throughput = None
        rows.append(
            {
                "Method": item["method"],
                "Steps / K": meta.get("steps_or_k"),
                "Total Params (M)": item["parameters"]["total_millions"],
                "Trainable Params (M)": item["parameters"]["trainable_millions"],
                "Aux Storage (MB)": item["memory"]["aux_storage_mb"],
                "B=1 Median (ms/traj)": item["latency"]["median_ms_per_trajectory"],
                "B=1 P95 (ms/traj)": item["latency"]["p95_ms_per_trajectory"],
                "B=128 Throughput (traj/s)": throughput,
                "Peak GPU Allocated (GB)": item["memory"]["peak_gpu_allocated_gb"],
            }
        )

    summary = {
        "protocol": protocol,
        "environment": payloads[0]["environment"],
        "rows": rows,
        "raw_results": payloads,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Method | Steps / K | Total / Trainable Params (M) | Aux. Storage (MB) | B=1 Median / P95 (ms/traj) | B=128 Throughput (traj/s) | Peak GPU (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        throughput_value = row["B=128 Throughput (traj/s)"]
        throughput_cell = (
            f"{throughput_value:.2f}" if isinstance(throughput_value, (int, float)) else "N/A (single-stream)"
        )
        lines.append(
            "| {Method} | {Steps / K} | {Total Params (M):.3f} / {Trainable Params (M):.3f} | "
            "{Aux Storage (MB):.2f} | {B=1 Median (ms/traj):.3f} / {B=1 P95 (ms/traj):.3f} | "
            f"{throughput_cell} | "
            "{Peak GPU Allocated (GB):.3f} |".format(**row)
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
