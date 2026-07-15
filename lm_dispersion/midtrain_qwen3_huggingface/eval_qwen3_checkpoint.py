import argparse
import json
import os
import time

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


ZERO_SHOT_TASKS = [
    "anli",
    "hellaswag",
    "lambada",
    "openbookqa",
    "paloma_wikitext_103",
    "piqa",
    "truthfulqa_mc2",
    "winogrande",
]

FEW_SHOT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "mmlu",
    "medmcqa",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a saved Qwen3 mid-training checkpoint with lm-eval."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to eval_ckpt_*_step* directory saved by the training callback.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON path. Defaults to <run_dir>/lm_eval_end_fewshot<num_fewshot>.json.",
    )
    parser.add_argument("--cache_dir", default=None, help="HF cache directory.")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Few-shot count for ARC/MMLU/MedMCQA tasks.")
    parser.add_argument("--max_eval_samples", type=int, default=200, help="lm-eval limit per task.")
    parser.add_argument("--max_gen_tokens", type=int, default=1024, help="Generation max tokens for lm-eval.")
    parser.add_argument("--batch_size", type=int, default=1, help="HFLM eval batch size.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--zeroshot_tasks",
        nargs="*",
        default=ZERO_SHOT_TASKS,
        help="Zero-shot lm-eval tasks. Defaults match midtrain_qwen3.py.",
    )
    parser.add_argument(
        "--fewshot_tasks",
        nargs="*",
        default=FEW_SHOT_TASKS,
        help="Few-shot lm-eval tasks. Defaults match midtrain_qwen3.py.",
    )
    return parser.parse_args()


def default_output_path(checkpoint_dir: str, num_fewshot: int) -> str:
    run_dir = os.path.dirname(os.path.abspath(checkpoint_dir.rstrip("/")))
    return os.path.join(run_dir, f"lm_eval_end_fewshot{num_fewshot}.json")


def main() -> None:
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    output_json = args.output_json or default_output_path(checkpoint_dir, args.num_fewshot)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"[Eval] checkpoint: {checkpoint_dir}")
    log(f"[Eval] output: {output_json}")
    log(f"[Eval] device: {device}")
    log(f"[Eval] zero-shot tasks: {args.zeroshot_tasks}")
    log(f"[Eval] few-shot tasks: {args.fewshot_tasks} | num_fewshot={args.num_fewshot}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wrapped_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    gen_kwargs = {"max_gen_toks": args.max_gen_tokens, "do_sample": False}

    with torch.inference_mode():
        log("[Eval] Running zero-shot tasks...")
        res_zeroshot = simple_evaluate(
            model=wrapped_model,
            tasks=args.zeroshot_tasks,
            num_fewshot=0,
            device=device,
            limit=args.max_eval_samples,
            gen_kwargs=gen_kwargs,
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

        log("[Eval] Running few-shot tasks...")
        res_fewshot = simple_evaluate(
            model=wrapped_model,
            tasks=args.fewshot_tasks,
            num_fewshot=args.num_fewshot,
            device=device,
            limit=args.max_eval_samples,
            gen_kwargs=gen_kwargs,
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

    merged = {**res_zeroshot["results"], **res_fewshot["results"]}
    payload = {
        "results": merged,
        "metadata": {
            "checkpoint_dir": checkpoint_dir,
            "num_fewshot": args.num_fewshot,
            "max_eval_samples": args.max_eval_samples,
            "max_gen_tokens": args.max_gen_tokens,
            "seed": args.seed,
            "zeroshot_tasks": args.zeroshot_tasks,
            "fewshot_tasks": args.fewshot_tasks,
            "wall_time_seconds": time.perf_counter() - t0,
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"[Eval] Saved to {output_json}")
    log(f"[Eval] Wall time: {payload['metadata']['wall_time_seconds']:.2f}s")


if __name__ == "__main__":
    main()
