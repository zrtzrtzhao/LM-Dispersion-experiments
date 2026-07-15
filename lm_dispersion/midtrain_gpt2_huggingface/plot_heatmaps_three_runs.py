import argparse
import os
import sys
import tempfile
import textwrap
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def add_project_import_path():
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
        if os.path.isdir(os.path.join(root, "key_observations")):
            sys.path.insert(0, os.path.join(root, "key_observations"))
            return

        nested = os.path.join(root, "LM-Dispersion-main")
        if os.path.isdir(os.path.join(nested, "key_observations")):
            sys.path.insert(0, os.path.join(nested, "key_observations"))
            return

    raise RuntimeError(
        "Could not find key_observations. Run this script from the project root "
        "or from inside LM-Dispersion-main."
    )


add_project_import_path()
from utils.text_data import get_random_long_text  # noqa: E402


def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def organize_embeddings(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in embeddings:
        z = z.squeeze(0).detach().float().cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer


def compute_cosine_similarities(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in embeddings:
        z = normalize(z, axis=1)
        cossim_matrix = np.matmul(z, z.T).clip(-1, 1)
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer


def build_hist_stack(
    cossim_matrix_by_layer: List[np.ndarray],
    step: int = 1,
    bins: int = 128,
):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_indices, hist_data = [], []
    denom = max(1, len(cossim_matrix_by_layer) - 1)
    for layer_idx, cossim_matrix in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx / denom)
    return np.array(hist_data), layer_indices


def find_checkpoints(run_folder: str) -> List[Tuple[int, str]]:
    ckpt_dirs = []
    if os.path.isdir(run_folder):
        for name in os.listdir(run_folder):
            path = os.path.join(run_folder, name)
            if os.path.isdir(path) and name.startswith("eval_ckpt_") and "step" in name:
                try:
                    step = int(name.split("step")[-1])
                    ckpt_dirs.append((step, path))
                except ValueError:
                    pass
    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs


def select_checkpoints(run_folder: str, checkpoint: str, stride: int) -> List[Tuple[int, str]]:
    ckpts = find_checkpoints(run_folder)
    if not ckpts:
        raise RuntimeError(f"No eval checkpoints found in: {run_folder}")

    if checkpoint == "end":
        return [ckpts[-1]]
    if checkpoint == "all":
        return ckpts[:: max(1, stride)]

    raise ValueError(f"Unknown checkpoint selection: {checkpoint}")


def compute_histories_for_checkpoint(tokenizer, model, device, repetitions, max_length):
    cossim_matrix_by_layer = None
    for random_seed in range(repetitions):
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
            embeddings_by_layer = organize_embeddings(output.hidden_states)
            curr = compute_cosine_similarities(embeddings_by_layer)

        if cossim_matrix_by_layer is None:
            cossim_matrix_by_layer = [m[None, ...] for m in curr]
        else:
            for i in range(len(cossim_matrix_by_layer)):
                cossim_matrix_by_layer[i] = np.concatenate(
                    (cossim_matrix_by_layer[i], curr[i][None, ...]),
                    axis=0,
                )

    for i in range(len(cossim_matrix_by_layer)):
        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)

    return build_hist_stack(cossim_matrix_by_layer)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cosine-similarity heatmaps for CE, angular_spread, and SFA-Peak runs."
    )
    parser.add_argument("--baseline_dir", required=True, help="Run directory for CE baseline.")
    parser.add_argument("--angular_dir", required=True, help="Run directory for angular_spread only.")
    parser.add_argument("--sfa_peak_dir", required=True, help="Run directory for angular_spread + SFA-Peak.")
    parser.add_argument("--baseline_label", default="CE baseline")
    parser.add_argument("--angular_label", default="angular_spread")
    parser.add_argument("--sfa_peak_label", default="angular_spread + SFA-Peak")
    parser.add_argument("--output", default="./figures/embedding_heatmaps_three_runs.png")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--checkpoint", choices=["end", "all"], default="end")
    parser.add_argument("--ckpt_stride", type=int, default=1)
    parser.add_argument("--title_fontsize", type=int, default=30)
    parser.add_argument("--vmax", type=float, default=10.0)
    return parser.parse_args()


def format_title(label, step):
    wrapped_label = textwrap.fill(label, width=28, break_long_words=False)
    return f"{wrapped_label}\nstep {step}"


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    runs_in_fig = [
        (args.baseline_label, args.baseline_dir),
        (args.angular_label, args.angular_dir),
        (args.sfa_peak_label, args.sfa_peak_dir),
    ]

    plt.rcParams["font.family"] = "sans-serif"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_lists = []
    max_ckpts = 0
    for label, run_folder in runs_in_fig:
        ckpts = select_checkpoints(run_folder, args.checkpoint, args.ckpt_stride)
        ckpt_lists.append(ckpts)
        max_ckpts = max(max_ckpts, len(ckpts))
        print(f"{label}: {len(ckpts)} checkpoint(s) from {run_folder}")

    fig = plt.figure(figsize=(9.5 * max_ckpts, 8 * len(runs_in_fig)))
    for row_idx, (label, _) in enumerate(tqdm(runs_in_fig, desc="runs")):
        ckpts = ckpt_lists[row_idx]
        for col_idx in tqdm(range(max_ckpts), desc=label, leave=False):
            ax = fig.add_subplot(len(runs_in_fig), max_ckpts, row_idx * max_ckpts + col_idx + 1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col_idx >= len(ckpts):
                ax.axis("off")
                continue

            step, ckpt_path = ckpts[col_idx]
            with tempfile.TemporaryDirectory() as tmp_cache:
                tokenizer = AutoTokenizer.from_pretrained(ckpt_path, cache_dir=tmp_cache)
                try:
                    config = AutoConfig.from_pretrained(ckpt_path, cache_dir=tmp_cache)
                    model = AutoModelForCausalLM.from_pretrained(
                        ckpt_path,
                        config=config,
                        cache_dir=tmp_cache,
                    )
                except Exception as e:
                    print(f"[Heatmap] Failed to load checkpoint: {ckpt_path}")
                    print(repr(e))
                    ax.axis("off")
                    continue

                model.to(device)
                model.eval()
                hist_matrix, layer_indices = compute_histories_for_checkpoint(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    repetitions=args.repetitions,
                    max_length=args.max_length,
                )

                if hist_matrix.size == 0:
                    ax.axis("off")
                else:
                    im = ax.imshow(
                        hist_matrix.T,
                        aspect="auto",
                        origin="lower",
                        cmap="Reds",
                        extent=[0, layer_indices[-1], -1, 1],
                        vmin=0,
                        vmax=args.vmax,
                    )
                    ax.set_title(
                        format_title(label, step),
                        pad=24,
                        fontfamily="monospace",
                        fontsize=args.title_fontsize,
                    )
                    ax.set_xlabel("Layer Fraction", fontsize=42)
                    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
                    ax.set_ylim([-0.25, 1])
                    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
                    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
                    if col_idx == 0:
                        ax.set_ylabel("Cosine Similarity", fontsize=42)
                    ax.tick_params(axis="both", which="major", labelsize=30)
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.tick_params(axis="both", which="major", labelsize=28)
                    cbar.ax.set_title("Probability\nDensity", fontsize=20, pad=20)

                del model
                if device == "cuda":
                    torch.cuda.empty_cache()

    fig.tight_layout(pad=2)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"\nDone. Saved to {args.output}")


if __name__ == "__main__":
    main()
