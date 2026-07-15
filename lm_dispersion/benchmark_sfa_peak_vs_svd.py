import argparse
import csv
import time
from typing import Dict, List

import torch


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_matrix(rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(rows, cols, device=device, dtype=dtype)
    x = x - x.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(x, p=2, dim=1)


def top_energy_svd(x: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(x)
    return singular_values[0].square()


def top_energy_sfa_peak_power(
    x: torch.Tensor,
    iterations: int,
    generator: torch.Generator = None,
) -> torch.Tensor:
    # Estimate lambda_max(X^T X) with k power iterations without computing full SVD.
    v = torch.randn(x.shape[1], device=x.device, dtype=x.dtype, generator=generator)
    v = torch.nn.functional.normalize(v, p=2, dim=0)

    for _ in range(iterations):
        u = x @ v
        v = x.transpose(0, 1) @ u
        v = torch.nn.functional.normalize(v, p=2, dim=0)

    xv = x @ v
    return xv.square().sum()


def time_method(fn, warmup: int, repeats: int, device: torch.device) -> Dict[str, float]:
    for _ in range(warmup):
        y = fn()
        if isinstance(y, torch.Tensor):
            y.detach()
    sync_if_needed(device)

    values: List[float] = []
    last_value = None
    for _ in range(repeats):
        sync_if_needed(device)
        start = time.perf_counter()
        y = fn()
        sync_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        values.append(elapsed_ms)
        last_value = float(y.detach().cpu())

    times = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std(unbiased=False)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "value": last_value,
    }


def run_case(args, rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    x = make_matrix(rows, cols, device, dtype)

    svd_stats = time_method(
        lambda: top_energy_svd(x),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    sfa_stats = time_method(
        lambda: top_energy_sfa_peak_power(x, iterations=args.power_iters),
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )

    rel_error = abs(sfa_stats["value"] - svd_stats["value"]) / max(abs(svd_stats["value"]), 1e-12)
    speedup = svd_stats["mean_ms"] / max(sfa_stats["mean_ms"], 1e-12)
    return {
        "rows": rows,
        "cols": cols,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "power_iters": args.power_iters,
        "svd_mean_ms": svd_stats["mean_ms"],
        "svd_std_ms": svd_stats["std_ms"],
        "sfa_peak_mean_ms": sfa_stats["mean_ms"],
        "sfa_peak_std_ms": sfa_stats["std_ms"],
        "speedup_svd_over_sfa": speedup,
        "svd_top_energy": svd_stats["value"],
        "sfa_peak_top_energy_est": sfa_stats["value"],
        "relative_error": rel_error,
    }


def parse_sizes(raw_sizes: str) -> List[tuple]:
    sizes = []
    for item in raw_sizes.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid size '{item}', expected ROWSxCOLS.")
        rows, cols = item.split("x", 1)
        sizes.append((int(rows), int(cols)))
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact SVD vs SFA-Peak power-iteration top-energy estimate."
    )
    parser.add_argument(
        "--sizes",
        default="4096x1024,4096x2048,4096x4096",
        help="Comma-separated matrix sizes, e.g. 4096x1024,4096x2048.",
    )
    parser.add_argument("--power_iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--csv", default=None, help="Optional output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    rows = []
    for n_rows, n_cols in parse_sizes(args.sizes):
        result = run_case(args, n_rows, n_cols, device, dtype)
        rows.append(result)
        print(
            f"{n_rows}x{n_cols} | "
            f"SVD {result['svd_mean_ms']:.3f} ms | "
            f"SFA-Peak k={args.power_iters} {result['sfa_peak_mean_ms']:.3f} ms | "
            f"speedup {result['speedup_svd_over_sfa']:.2f}x | "
            f"rel.err {result['relative_error']:.4e}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
