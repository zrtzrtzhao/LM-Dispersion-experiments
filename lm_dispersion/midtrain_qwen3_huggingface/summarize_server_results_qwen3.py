import argparse
import csv
import json
import math
import re
from pathlib import Path


TASKS = [
    ("ANLI R2", "anli_r2", "acc,none"),
    ("LAMBADA", "lambada_openai", "acc,none"),
    ("OpenBookQA", "openbookqa", "acc_norm,none"),
    ("PIQA", "piqa", "acc_norm,none"),
    ("TruthfulQA", "truthfulqa_mc2", "acc,none"),
    ("WinoGrande", "winogrande", "acc,none"),
    ("ARC Easy", "arc_easy", "acc_norm,none"),
    ("ARC Challenge", "arc_challenge", "acc_norm,none"),
    ("MedMCQA", "medmcqa", "acc_norm,none"),
    ("MMLU", "mmlu", "acc,none"),
]


ORDER = [
    "CE baseline",
    "angular_spread only",
    "NEFTune noise 1.0",
    "Active forgetting 1000",
    "SFA-CE only last",
    "angular_spread + SFA-CE last",
    "angular_spread + SFA-CE late_half",
    "angular_spread + SFA-CE no-norm last",
    "angular_spread + SFA-CE no-norm late_half",
    "angular_spread + SFA-Peak last 0.03",
    "angular_spread + SFA-Peak late_half 0.03",
    "angular_spread + SFA-Peak late_half 0.01",
    "angular_spread + SFA-Peak all 0.01",
    "SFA-Peak only late_half 0.01",
    "SFA-Peak only late_half 0.03",
    "angular_spread + LogDet late_half 0.01",
    "angular_spread + LogDet late_half 0.03",
    "Random Quant last 0.03",
]


def get_metric(results, task, metric_name, scale=1.0):
    if task not in results:
        return None
    value = results[task].get(metric_name)
    if value is None or value == "N/A":
        return None
    return float(value) * scale


def parse_step(path):
    match = re.search(r"_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def parse_metadata(run_dir):
    basename = run_dir.name
    info = {
        "model": None,
        "train_tokens": None,
        "maxsample": None,
        "peak_memory_gb": None,
        "total_time_h": None,
    }

    match = re.search(r"^midtrain_(.+?)_Salesforce-wikitext_", basename)
    if match:
        info["model"] = match.group(1)
    match = re.search(r"_token-([0-9]+)", basename)
    if match:
        info["train_tokens"] = int(match.group(1))
    match = re.search(r"_maxsample-([0-9]+)", basename)
    if match:
        info["maxsample"] = int(match.group(1))

    log_path = run_dir / "log.txt"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Training for\s+([0-9]+)\s+tokens", text)
        if match:
            info["train_tokens"] = int(match.group(1))
        match = re.search(r"maxsample-([0-9]+)", text)
        if match:
            info["maxsample"] = int(match.group(1))
        match = re.search(r"Peak memory:\s*([0-9.]+)\s*GB", text)
        if match:
            info["peak_memory_gb"] = float(match.group(1))
        match = re.search(
            r"Training wall time:\s*[0-9.]+s\s*\([0-9.]+\s*min,\s*([0-9.]+)\s*h\)",
            text,
        )
        if match:
            info["total_time_h"] = float(match.group(1))

    return info


def method_name(run_dir):
    name = run_dir.name

    if "_ccnoise-" in name:
        coeff = name.split("_ccnoise-", 1)[1].split("_", 1)[0]
        return f"NEFTune noise {coeff}"
    if "_ccforget-" in name:
        k_value = name.split("_ccforget-", 1)[1].split("_", 1)[0]
        return f"Active forgetting {k_value}"

    if "_rq-" in name:
        tag = name.split("_rq-", 1)[1].split("_fewshot-", 1)[0]
        if tag == "None":
            return base_dispersion_name(name)
        parts = tag.split("-")
        coeff = parts[0] if len(parts) > 0 else "unknown"
        loc = parts[1] if len(parts) > 1 else "unknown"
        return f"Random Quant {loc} {coeff}"

    if "_logdet-" in name:
        tag = name.split("_logdet-", 1)[1].split("_fewshot-", 1)[0]
        parts = tag.split("-")
        coeff = parts[0] if len(parts) > 0 else "unknown"
        loc = parts[1] if len(parts) > 1 else "unknown"
        prefix = "LogDet only" if "_disp-None-" in name else "angular_spread + LogDet"
        return f"{prefix} {loc} {coeff}"

    if "_sfapeak-" in name:
        tag = name.split("_sfapeak-", 1)[1].split("_fewshot-", 1)[0]
        parts = tag.split("-")
        coeff = parts[0] if len(parts) > 0 else "unknown"
        loc = parts[1] if len(parts) > 1 else "unknown"
        prefix = "SFA-Peak only" if "_disp-None-" in name else "angular_spread + SFA-Peak"
        return f"{prefix} {loc} {coeff}"

    if "_sface-nonorm-" in name:
        tag = name.split("_sface-nonorm-", 1)[1].split("_fewshot-", 1)[0]
        loc = "late_half" if "late_half" in tag else "last"
        prefix = "SFA-CE no-norm only" if "_disp-None-" in name else "angular_spread + SFA-CE no-norm"
        return f"{prefix} {loc}"

    if "_sface-" in name:
        tag = name.split("_sface-", 1)[1].split("_fewshot-", 1)[0]
        loc = "late_half" if "late_half" in tag else "last"
        prefix = "SFA-CE only" if "_disp-None-" in name else "angular_spread + SFA-CE"
        return f"{prefix} {loc}"

    return base_dispersion_name(name)


def base_dispersion_name(name):
    if "_disp-None-" in name:
        return "CE baseline"
    if "_disp-angular_spread-" in name:
        return "angular_spread only"
    if "_disp-" in name:
        tag = name.split("_disp-", 1)[1].split("-tau_cos-", 1)[0]
        return f"disp-{tag}"
    return name


def is_formal(row):
    if row["maxsample"] is not None and row["maxsample"] >= 100:
        return True
    if row["train_tokens"] is not None and row["train_tokens"] >= 1_000_000:
        return True
    if row["step"] is not None and row["step"] >= 100:
        return True
    return False


def build_row(json_path, results_root):
    run_dir = json_path.parent
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    meta = parse_metadata(run_dir)

    task_scores = {}
    for label, task, metric_name in TASKS:
        task_scores[label] = get_metric(results, task, metric_name, 100.0)

    values = [v for v in task_scores.values() if v is not None and math.isfinite(v)]
    avg = sum(values) / len(values) if values else None

    row = {
        "model": meta["model"] or "",
        "run": method_name(run_dir),
        "avg10": avg,
        "wikitext_ppl": get_metric(results, "paloma_wikitext_103", "word_perplexity,none"),
        "lambada_ppl": get_metric(results, "lambada_openai", "perplexity,none"),
        "lambada_acc": get_metric(results, "lambada_openai", "acc,none", 100.0),
        "peak_memory_gb": meta["peak_memory_gb"],
        "total_time_h": meta["total_time_h"],
        "step": parse_step(json_path),
        "train_tokens": meta["train_tokens"],
        "maxsample": meta["maxsample"],
        "source": str(json_path.relative_to(results_root)),
        "task_scores": task_scores,
    }
    row["note"] = "" if is_formal(row) else "smoke test / excluded by default"
    return row


def sort_key(row):
    try:
        order = ORDER.index(row["run"])
    except ValueError:
        order = len(ORDER)
    return (row["model"] or "", order, row["run"], row["source"])


def fmt(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text or "unknown").strip("_")


def delta(value, baseline):
    if value is None or baseline is None:
        return None
    return value - baseline


def baseline_by_model(rows):
    baselines = {}
    for row in rows:
        if row["run"] == "CE baseline" and row["model"] not in baselines:
            baselines[row["model"]] = row
    return baselines


def best_by_model(rows):
    best = {}
    for row in rows:
        if row["avg10"] is None:
            continue
        model = row["model"]
        if model not in best or row["avg10"] > best[model]["avg10"]:
            best[model] = row
    return best


def collect_rows(results_root, include_smoke, model_name=None):
    rows = []
    for json_path in sorted(results_root.rglob("lm_eval_end_*.json")):
        row = build_row(json_path, results_root)
        if model_name and row["model"] != model_name:
            continue
        if include_smoke or is_formal(row):
            rows.append(row)
    return sorted(rows, key=sort_key)


def write_summary_csv(rows, output_dir):
    path = output_dir / "qwen3_main_results_summary.csv"
    fields = [
        "Model",
        "Run",
        "10-task avg",
        "Wikitext word ppl",
        "LAMBADA ppl",
        "LAMBADA acc",
        "Peak memory",
        "Total time",
        "Step",
        "Train tokens",
        "Max eval samples",
        "Source",
        "Note",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Run": row["run"],
                    "Model": row["model"],
                    "10-task avg": fmt(row["avg10"], 4),
                    "Wikitext word ppl": fmt(row["wikitext_ppl"], 4),
                    "LAMBADA ppl": fmt(row["lambada_ppl"], 4),
                    "LAMBADA acc": fmt(row["lambada_acc"], 2),
                    "Peak memory": "" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f} GB",
                    "Total time": "" if row["total_time_h"] is None else f"{row['total_time_h']:.4f} h",
                    "Step": row["step"] or "",
                    "Train tokens": row["train_tokens"] or "",
                    "Max eval samples": row["maxsample"] or "",
                    "Source": row["source"],
                    "Note": row["note"],
                }
            )
    return path


def write_task_csv(rows, output_dir):
    path = output_dir / "qwen3_main_results_10task_detail.csv"
    fields = ["Model", "Run"] + [label for label, _, _ in TASKS] + ["Source", "Note"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Run": row["run"],
                    "Model": row["model"],
                    **{label: fmt(row["task_scores"].get(label), 2) for label, _, _ in TASKS},
                    "Source": row["source"],
                    "Note": row["note"],
                }
            )
    return path


def write_markdown(rows, output_dir):
    path = output_dir / "qwen3_main_results_summary.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Main Results Summary\n\n")
        f.write("| Model | Run | 10-task avg | Wikitext word ppl | LAMBADA ppl | LAMBADA acc | Peak memory | Total time | Note |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in rows:
            mem = "" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f} GB"
            time = "" if row["total_time_h"] is None else f"{row['total_time_h']:.4f} h"
            f.write(
                f"| {row['model']} | {row['run']} | {fmt(row['avg10'], 4)} | {fmt(row['wikitext_ppl'], 4)} | "
                f"{fmt(row['lambada_ppl'], 4)} | {fmt(row['lambada_acc'], 2)} | "
                f"{mem} | {time} | {row['note']} |\n"
            )
    return path


def write_latex(rows, output_dir):
    path = output_dir / "qwen3_main_results_summary.tex"
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Run & 10-task avg & Wikitext word ppl & LAMBADA ppl & LAMBADA acc & Peak memory & Total time \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            mem = "" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f} GB"
            time = "" if row["total_time_h"] is None else f"{row['total_time_h']:.4f} h"
            f.write(
                f"{row['model']} & {row['run']} & {fmt(row['avg10'], 4)} & {fmt(row['wikitext_ppl'], 4)} & "
                f"{fmt(row['lambada_ppl'], 4)} & {fmt(row['lambada_acc'], 2)} & {mem} & {time} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return path


def write_best_by_model_csv(rows, output_dir):
    path = output_dir / "qwen3_main_results_best_by_model.csv"
    fields = [
        "Model",
        "Best run",
        "10-task avg",
        "CE baseline avg",
        "Delta avg vs CE",
        "Wikitext word ppl",
        "Delta Wikitext ppl",
        "LAMBADA acc",
        "Delta LAMBADA acc",
        "Peak memory",
        "Total time",
        "Source",
    ]
    baselines = baseline_by_model(rows)
    best = best_by_model(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model in sorted(best):
            row = best[model]
            baseline = baselines.get(model, {})
            writer.writerow(
                {
                    "Model": model,
                    "Best run": row["run"],
                    "10-task avg": fmt(row["avg10"], 4),
                    "CE baseline avg": fmt(baseline.get("avg10"), 4),
                    "Delta avg vs CE": fmt(delta(row["avg10"], baseline.get("avg10")), 4),
                    "Wikitext word ppl": fmt(row["wikitext_ppl"], 4),
                    "Delta Wikitext ppl": fmt(delta(row["wikitext_ppl"], baseline.get("wikitext_ppl")), 4),
                    "LAMBADA acc": fmt(row["lambada_acc"], 2),
                    "Delta LAMBADA acc": fmt(delta(row["lambada_acc"], baseline.get("lambada_acc")), 2),
                    "Peak memory": "" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f} GB",
                    "Total time": "" if row["total_time_h"] is None else f"{row['total_time_h']:.4f} h",
                    "Source": row["source"],
                }
            )
    return path


def write_delta_csv(rows, output_dir):
    path = output_dir / "qwen3_main_results_delta_vs_ce.csv"
    fields = [
        "Model",
        "Run",
        "10-task avg",
        "Delta avg vs CE",
        "Wikitext word ppl",
        "Delta Wikitext ppl",
        "LAMBADA acc",
        "Delta LAMBADA acc",
        "LAMBADA ppl",
        "Delta LAMBADA ppl",
        "Peak memory",
        "Delta peak memory",
        "Total time",
        "Delta total time",
        "Source",
        "Note",
    ]
    baselines = baseline_by_model(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            baseline = baselines.get(row["model"], {})
            writer.writerow(
                {
                    "Model": row["model"],
                    "Run": row["run"],
                    "10-task avg": fmt(row["avg10"], 4),
                    "Delta avg vs CE": fmt(delta(row["avg10"], baseline.get("avg10")), 4),
                    "Wikitext word ppl": fmt(row["wikitext_ppl"], 4),
                    "Delta Wikitext ppl": fmt(delta(row["wikitext_ppl"], baseline.get("wikitext_ppl")), 4),
                    "LAMBADA acc": fmt(row["lambada_acc"], 2),
                    "Delta LAMBADA acc": fmt(delta(row["lambada_acc"], baseline.get("lambada_acc")), 2),
                    "LAMBADA ppl": fmt(row["lambada_ppl"], 4),
                    "Delta LAMBADA ppl": fmt(delta(row["lambada_ppl"], baseline.get("lambada_ppl")), 4),
                    "Peak memory": "" if row["peak_memory_gb"] is None else f"{row['peak_memory_gb']:.2f} GB",
                    "Delta peak memory": fmt(delta(row["peak_memory_gb"], baseline.get("peak_memory_gb")), 2),
                    "Total time": "" if row["total_time_h"] is None else f"{row['total_time_h']:.4f} h",
                    "Delta total time": fmt(delta(row["total_time_h"], baseline.get("total_time_h")), 4),
                    "Source": row["source"],
                    "Note": row["note"],
                }
            )
    return path


def write_delta_latex(rows, output_dir):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / "qwen3_main_results_delta_vs_ce.tex"
    baselines = baseline_by_model(rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Run & 10-task avg & $\\Delta$ avg & Wikitext ppl & LAMBADA acc \\\\\n")
        f.write("\\midrule\n")
        last_model = None
        for row in rows:
            if last_model is not None and row["model"] != last_model:
                f.write("\\midrule\n")
            last_model = row["model"]
            baseline = baselines.get(row["model"], {})
            f.write(
                f"{row['model']} & {row['run']} & {fmt(row['avg10'], 4)} & "
                f"{fmt(delta(row['avg10'], baseline.get('avg10')), 4)} & "
                f"{fmt(row['wikitext_ppl'], 4)} & {fmt(row['lambada_acc'], 2)} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return path


def write_figures(rows, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[Warning] matplotlib is not available; skipped figures: {exc}")
        return []

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model and row["avg10"] is not None]
        if not model_rows:
            continue
        model_rows = sorted(model_rows, key=lambda row: row["avg10"])
        labels = [row["run"] for row in model_rows]
        values = [row["avg10"] for row in model_rows]
        colors = ["#4C78A8"] * len(model_rows)
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        colors[best_index] = "#59A14F"
        for idx, row in enumerate(model_rows):
            if (
                (row["wikitext_ppl"] is not None and row["wikitext_ppl"] > 1000)
                or (row["lambada_ppl"] is not None and row["lambada_ppl"] > 1000)
                or row["lambada_acc"] == 0
            ):
                colors[idx] = "#E15759"

        height = max(4.0, 0.36 * len(model_rows) + 1.8)
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(range(len(model_rows)), values, color=colors)
        ax.set_yticks(range(len(model_rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("10-task average")
        ax.set_title(f"{model} main results")
        ax.grid(axis="x", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(value, idx, f" {value:.4f}", va="center", fontsize=8)
        fig.tight_layout()
        path = figure_dir / f"qwen3_main_results_10task_avg_{safe_name(model)}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description="Summarize server-side Qwen3 midtraining results.")
    parser.add_argument("--results_dir", type=Path, default=Path("./results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional exact model filter, e.g. Qwen-Qwen3-0.6B.",
    )
    parser.add_argument("--include_smoke", action="store_true")
    args = parser.parse_args()

    results_root = args.results_dir.resolve()
    output_dir = (args.output_dir or args.results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_root, args.include_smoke, model_name=args.model_name)
    if not rows:
        raise RuntimeError(f"No formal lm_eval_end_*.json found under {results_root}")

    summary = write_summary_csv(rows, output_dir)
    detail = write_task_csv(rows, output_dir)
    markdown = write_markdown(rows, output_dir)
    latex = write_latex(rows, output_dir)
    best = write_best_by_model_csv(rows, output_dir)
    delta_table = write_delta_csv(rows, output_dir)
    delta_latex = write_delta_latex(rows, output_dir)
    figures = write_figures(rows, output_dir)

    print(f"Rows: {len(rows)}")
    print(f"Summary CSV: {summary}")
    print(f"10-task detail CSV: {detail}")
    print(f"Markdown: {markdown}")
    print(f"LaTeX: {latex}")
    print(f"Best-by-model CSV: {best}")
    print(f"Delta-vs-CE CSV: {delta_table}")
    print(f"Delta-vs-CE LaTeX: {delta_latex}")
    for figure in figures:
        print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
