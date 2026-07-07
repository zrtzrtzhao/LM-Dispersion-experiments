from typing import List
import argparse
import os
import sys
import tempfile

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
        if os.path.isdir(os.path.join(root, 'key_observations')):
            sys.path.insert(0, os.path.join(root, 'key_observations'))
            return
        nested = os.path.join(root, 'LM-Dispersion-main')
        if os.path.isdir(os.path.join(nested, 'key_observations')):
            sys.path.insert(0, os.path.join(nested, 'key_observations'))
            return

    raise RuntimeError(
        'Could not find key_observations. Run this script from the project root '
        'or from inside LM-Dispersion-main.'
    )


add_project_import_path()
from utils.text_data import get_random_long_text


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


def build_hist_stack(cossim_matrix_by_layer: List[np.ndarray], step: int = 1, bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_indices, hist_data = [], []
    denom = max(1, len(cossim_matrix_by_layer) - 1)
    for layer_idx, cossim_matrix in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx / denom)
    return np.array(hist_data), layer_indices


def find_checkpoints(run_folder: str):
    ckpt_dirs = []
    if os.path.isdir(run_folder):
        for name in os.listdir(run_folder):
            path = os.path.join(run_folder, name)
            if os.path.isdir(path) and name.startswith('eval_ckpt_') and 'step' in name:
                try:
                    step = int(name.split('step')[-1])
                    ckpt_dirs.append((step, path))
                except ValueError:
                    pass
    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs


def run_dir_from_args(results_dir, model_name, dataset_name, sfa_ce):
    dataset_slug = dataset_name.replace('/', '-')
    base = (
        f'midtrain_{model_name}_{dataset_slug}_lr-5e-05_token-200000000_'
        f'disp-None-1-all-tau_cos-1.0-tau_l2-1.0_'
    )
    if sfa_ce:
        base += 'sface-0.1-k1-alpha0.1-sample128_'
    base += 'fewshot-1_maxsample-200_seed-1'
    return os.path.join(results_dir, base)


def compute_histories_for_checkpoint(tokenizer, model, device, repetitions, max_length):
    cossim_matrix_by_layer = None
    for random_seed in range(repetitions):
        torch.manual_seed(random_seed)
        text = get_random_long_text(
            'wikipedia',
            random_seed=random_seed,
            min_word_count=1024,
            max_word_count=1280,
        )
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
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
                    (cossim_matrix_by_layer[i], curr[i][None, ...]), axis=0
                )

    for i in range(len(cossim_matrix_by_layer)):
        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)

    return build_hist_stack(cossim_matrix_by_layer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2')
    parser.add_argument('--dataset_name', default='Salesforce/wikitext')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--baseline_dir', default=None)
    parser.add_argument('--sfa_ce_dir', default=None)
    parser.add_argument('--baseline_label', default='CE baseline')
    parser.add_argument('--sfa_ce_label', default='SFA-CE only')
    parser.add_argument('--output', default='./figures/embedding_heatmaps_grid_sfa_ce.png')
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--ckpt_stride', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    baseline_dir = args.baseline_dir or run_dir_from_args(
        args.results_dir, args.model_name, args.dataset_name, sfa_ce=False
    )
    sfa_ce_dir = args.sfa_ce_dir or run_dir_from_args(
        args.results_dir, args.model_name, args.dataset_name, sfa_ce=True
    )
    runs_in_fig = [(args.baseline_label, baseline_dir), (args.sfa_ce_label, sfa_ce_dir)]

    plt.rcParams['font.family'] = 'sans-serif'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_lists = []
    max_ckpts = 0
    for label, run_folder in runs_in_fig:
        ckpts = find_checkpoints(run_folder)[::args.ckpt_stride]
        if not ckpts:
            raise RuntimeError(f'No eval checkpoints found for {label}: {run_folder}')
        ckpt_lists.append(ckpts)
        max_ckpts = max(max_ckpts, len(ckpts))

    fig = plt.figure(figsize=(9.5 * max_ckpts, 8 * len(runs_in_fig)))
    for row_idx, (label, _) in enumerate(tqdm(runs_in_fig)):
        ckpts = ckpt_lists[row_idx]
        for col_idx in tqdm(range(max_ckpts)):
            ax = fig.add_subplot(len(runs_in_fig), max_ckpts, row_idx * max_ckpts + col_idx + 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if col_idx >= len(ckpts):
                ax.axis('off')
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
                    print(f'[Heatmap] Failed to load checkpoint: {ckpt_path}')
                    print(repr(e))
                    ax.axis('off')
                    continue

                model.to(device)
                model.eval()
                hist_matrix, layer_indices = compute_histories_for_checkpoint(
                    tokenizer,
                    model,
                    device,
                    args.repetitions,
                    args.max_length,
                )

                if hist_matrix.size == 0:
                    ax.axis('off')
                else:
                    im = ax.imshow(
                        hist_matrix.T,
                        aspect='auto',
                        origin='lower',
                        cmap='Reds',
                        extent=[0, layer_indices[-1], -1, 1],
                        vmin=0,
                        vmax=10,
                    )
                    ax.set_title(f'{label}\nstep {step}', pad=24, fontfamily='monospace', fontsize=42)
                    ax.set_xlabel('Layer Fraction', fontsize=42)
                    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
                    ax.set_ylim([-0.25, 1])
                    ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
                    ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
                    if col_idx == 0:
                        ax.set_ylabel('Cosine Similarity', fontsize=42)
                    ax.tick_params(axis='both', which='major', labelsize=30)
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.tick_params(axis='both', which='major', labelsize=28)
                    cbar.ax.set_title('Probability\nDensity', fontsize=20, pad=20)

                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()

    fig.tight_layout(pad=2)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f'\nDone. Saved to {args.output}')
