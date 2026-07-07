import argparse
import csv
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def add_project_import_path() -> None:
    search_roots = []
    here = os.path.abspath(os.path.dirname(__file__))
    search_roots.append(here)
    search_roots.append(os.getcwd())

    for root in list(search_roots):
        cur = root
        while True:
            search_roots.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    for root in search_roots:
        key_obs = os.path.join(root, "key_observations")
        if os.path.isdir(key_obs):
            sys.path.insert(0, key_obs)
            return

        nested_key_obs = os.path.join(root, "LM-Dispersion-main", "key_observations")
        if os.path.isdir(nested_key_obs):
            sys.path.insert(0, nested_key_obs)
            return

    raise RuntimeError(
        "Could not find key_observations. Run this script from the project root "
        "or from inside LM-Dispersion-main."
    )


add_project_import_path()
from utils.text_data import get_random_long_text  # noqa: E402


def find_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    ckpt_dirs = []
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run directory does not exist: {run_dir}")

    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("eval_ckpt_") or "step" not in name:
            continue
        try:
            step = int(name.split("step")[-1])
        except ValueError:
            continue
        ckpt_dirs.append((step, path))

    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs


def select_checkpoints(run_dir: str, which: str, stride: int) -> List[Tuple[int, str]]:
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        raise RuntimeError(f"No eval checkpoints found in: {run_dir}")

    if which == "end":
        return [ckpts[-1]]
    if which == "all":
        return ckpts[:: max(1, stride)]

    raise ValueError(f"Unknown checkpoint selection: {which}")


def row_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def layer_spectral_metrics(hidden: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    hidden = row_normalize(hidden)
    singular_values = np.linalg.svd(hidden, compute_uv=False)
    energy = singular_values ** 2
    total_energy = float(energy.sum())

    if total_energy <= eps:
        return {
            "top_energy_ratio": 0.0,
            "effective_rank": 0.0,
            "spectral_entropy": 0.0,
        }

    probs = energy / total_energy
    spectral_entropy = float(-np.sum(probs * np.log(probs + eps)))
    return {
        "top_energy_ratio": float(probs[0]),
        "effective_rank": float(np.exp(spectral_entropy)),
        "spectral_entropy": spectral_entropy,
    }


def evaluate_checkpoint(
    ckpt_path: str,
    device: str,
    repetitions: int,
    max_length: int,
) -> Dict[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmp_cache:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, cache_dir=tmp_cache)
        config = AutoConfig.from_pretrained(ckpt_path, cache_dir=tmp_cache)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            config=config,
            cache_dir=tmp_cache,
        )
        model.to(device)
        model.eval()

        per_rep = []
        for random_seed in tqdm(range(repetitions), desc=os.path.basename(ckpt_path), leave=False):
            torch.manual_seed(random_seed)
            text = get_random_long_text(
                "wikipedia",
                random_seed=random_seed,
                min_word_count=1024,
                max_word_count=1280,
            )
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                output = model(**tokens, output_hidden_states=True)

            per_layer = []
            for hidden_state in output.hidden_states:
                hidden = hidden_state.squeeze(0).detach().float().cpu().numpy()
                per_layer.append(layer_spectral_metrics(hidden))
            per_rep.append(per_layer)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    metric_names = ["top_energy_ratio", "effective_rank", "spectral_entropy"]
    stacked = {
        name: np.array([[layer[name] for layer in rep] for rep in per_rep], dtype=np.float64)
        for name in metric_names
    }
    return {name: values.mean(axis=0) for name, values in stacked.items()}


def write_csv(rows: List[Dict[str, float]], csv_path: str) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    fieldnames = [
        "run",
        "step",
        "layer_index",
        "layer_fraction",
        "top_energy_ratio",
        "effective_rank",
        "spectral_entropy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(results: Dict[str, Dict[int, Dict[str, np.ndarray]]], output_path: str) -> None:
    metric_titles = {
        "top_energy_ratio": "Top Energy Ratio",
        "effective_rank": "Effective Rank",
        "spectral_entropy": "Spectral Entropy",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric_name in zip(axes, metric_titles):
        for run_label, by_step in results.items():
            final_step = sorted(by_step.keys())[-1]
            values = by_step[final_step][metric_name]
            x = np.arange(len(values), dtype=np.float64)
            x = x / max(1, len(values) - 1)
            ax.plot(x, values, marker="o", linewidth=2, label=f"{run_label} step {final_step}")

        ax.set_title(metric_titles[metric_name], fontsize=16)
        ax.set_xlabel("Layer Fraction", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel("Metric Value", fontsize=13)
    axes[-1].legend(fontsize=10)
    fig.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare layer-wise spectral metrics for angular_spread and SFA-Peak checkpoints."
    )
    parser.add_argument("--baseline_dir", default=None, help="Optional run directory for CE baseline.")
    parser.add_argument("--angular_dir", required=True, help="Run directory for angular_spread only.")
    parser.add_argument("--sfa_peak_dir", required=True, help="Run directory for angular_spread + SFA-Peak.")
    parser.add_argument("--baseline_label", default="CE baseline")
    parser.add_argument("--angular_label", default="angular_spread")
    parser.add_argument("--sfa_peak_label", default="angular_spread + SFA-Peak")
    parser.add_argument("--output", default="./figures/spectral_metrics_sfa_peak_vs_angular.png")
    parser.add_argument("--csv_output", default="./figures/spectral_metrics_sfa_peak_vs_angular.csv")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--checkpoint", choices=["end", "all"], default="end")
    parser.add_argument("--ckpt_stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dirs = {}
    if args.baseline_dir:
        run_dirs[args.baseline_label] = args.baseline_dir
    run_dirs[args.angular_label] = args.angular_dir
    run_dirs[args.sfa_peak_label] = args.sfa_peak_dir

    results: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    csv_rows: List[Dict[str, float]] = []

    for run_label, run_dir in run_dirs.items():
        results[run_label] = {}
        ckpts = select_checkpoints(run_dir, args.checkpoint, args.ckpt_stride)
        for step, ckpt_path in ckpts:
            print(f"Evaluating {run_label}: step {step}")
            metrics = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                device=device,
                repetitions=args.repetitions,
                max_length=args.max_length,
            )
            results[run_label][step] = metrics

            num_layers = len(next(iter(metrics.values())))
            for layer_index in range(num_layers):
                layer_fraction = layer_index / max(1, num_layers - 1)
                csv_rows.append(
                    {
                        "run": run_label,
                        "step": step,
                        "layer_index": layer_index,
                        "layer_fraction": layer_fraction,
                        "top_energy_ratio": metrics["top_energy_ratio"][layer_index],
                        "effective_rank": metrics["effective_rank"][layer_index],
                        "spectral_entropy": metrics["spectral_entropy"][layer_index],
                    }
                )

    write_csv(csv_rows, args.csv_output)
    plot_results(results, args.output)
    print(f"Saved plot to {args.output}")
    print(f"Saved metrics to {args.csv_output}")


if __name__ == "__main__":
    main()
