#!/usr/bin/env python3
"""
Regenerate Adaptive-RAG paper tables from local artifacts.

Outputs one Markdown file per table plus a combined results.md.
Missing local artifacts are shown as em dashes, so partial FLAN-T5-XL-only
experiments are acceptable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASETS = ["squad", "nq", "trivia", "musique", "hotpotqa", "2wikimultihopqa"]
DATASET_DISPLAY = {
    "squad": "SQuAD",
    "nq": "Natural Questions",
    "trivia": "TriviaQA",
    "musique": "MuSiQue",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2WikiMultiHopQA",
}
SYSTEMS = ["nor_qa", "oner_qa", "ircot_qa"]
SYSTEM_DISPLAY = {
    "nor_qa": "No Retrieval",
    "oner_qa": "Single-step Approach",
    "ircot_qa": "Multi-step Approach",
}
LABEL_DISPLAY = {"A": "No (A)", "B": "One (B)", "C": "Multi (C)"}

PAPER_TABLE_1 = [
    ("FLAN-T5-XL (3B)", "Simple", "No Retrieval", 14.87, 21.12, 15.97, 0.00, 0.11),
    ("FLAN-T5-XL (3B)", "Simple", "Single-step Approach", 34.83, 44.31, 38.87, 1.00, 1.00),
    ("FLAN-T5-XL (3B)", "Adaptive", "Adaptive Retrieval", 23.87, 32.24, 26.73, 0.50, 0.56),
    ("FLAN-T5-XL (3B)", "Adaptive", "Self-RAG*", 9.90, 20.79, 31.57, 0.72, 0.43),
    ("FLAN-T5-XL (3B)", "Adaptive", "Adaptive-RAG (Ours)", 37.17, 46.94, 42.10, 2.17, 3.60),
    ("FLAN-T5-XL (3B)", "Complex", "Multi-step Approach", 39.00, 48.85, 43.70, 4.69, 8.81),
    ("FLAN-T5-XL (3B)", "Oracle", "Adaptive-RAG w/ Oracle", 45.00, 56.28, 49.90, 1.28, 2.11),
    ("FLAN-T5-XXL (11B)", "Simple", "No Retrieval", 17.83, 25.14, 19.33, 0.00, 0.08),
    ("FLAN-T5-XXL (11B)", "Simple", "Single-step Approach", 37.87, 47.63, 41.90, 1.00, 1.00),
    ("FLAN-T5-XXL (11B)", "Adaptive", "Adaptive Retrieval", 26.93, 35.67, 29.73, 0.50, 0.54),
    ("FLAN-T5-XXL (11B)", "Adaptive", "Self-RAG*", 10.87, 22.98, 34.13, 0.74, 0.23),
    ("FLAN-T5-XXL (11B)", "Adaptive", "Adaptive-RAG (Ours)", 38.90, 48.62, 43.77, 1.35, 2.00),
    ("FLAN-T5-XXL (11B)", "Complex", "Multi-step Approach", 40.13, 50.09, 45.20, 2.13, 3.80),
    ("FLAN-T5-XXL (11B)", "Oracle", "Adaptive-RAG w/ Oracle", 47.17, 58.60, 52.20, 0.84, 1.10),
    ("GPT-3.5 (Turbo)", "Simple", "No Retrieval", 35.77, 48.56, 44.27, 0.00, 0.71),
    ("GPT-3.5 (Turbo)", "Simple", "Single-step Approach", 34.73, 46.99, 45.27, 1.00, 1.00),
    ("GPT-3.5 (Turbo)", "Adaptive", "Adaptive Retrieval", 35.90, 48.20, 45.30, 0.50, 0.86),
    ("GPT-3.5 (Turbo)", "Adaptive", "Self-RAG*", 10.87, 22.98, 34.13, 0.74, 1.50),
    ("GPT-3.5 (Turbo)", "Adaptive", "Adaptive-RAG (Ours)", 37.97, 50.91, 48.97, 1.03, 1.46),
    ("GPT-3.5 (Turbo)", "Complex", "Multi-step Approach", 38.13, 50.87, 49.70, 2.81, 3.33),
    ("GPT-3.5 (Turbo)", "Oracle", "Adaptive-RAG w/ Oracle", 47.70, 62.80, 58.57, 0.50, 1.03),
]

PAPER_TABLE_2 = [
    ("squad", "Single-step", "No Retrieval", 3.60, 10.50, 5.00, 0.00, 0.11),
    ("squad", "Single-step", "Single-step Approach", 27.80, 39.30, 34.00, 1.00, 1.00),
    ("squad", "Single-step", "Adaptive Retrieval", 13.40, 23.10, 17.60, 0.50, 0.55),
    ("squad", "Single-step", "Self-RAG*", 2.20, 11.20, 18.40, 0.63, 0.50),
    ("squad", "Single-step", "Adaptive-RAG (Ours)", 26.80, 38.30, 33.00, 1.37, 2.02),
    ("squad", "Single-step", "Multi-step Approach", 24.40, 35.60, 29.60, 4.52, 9.03),
    ("squad", "Single-step", "Adaptive-RAG w/ Oracle", 32.00, 45.60, 38.20, 1.24, 1.60),
    ("nq", "Single-step", "No Retrieval", 14.20, 19.00, 15.60, 0.00, 0.13),
    ("nq", "Single-step", "Single-step Approach", 37.80, 47.30, 44.60, 1.00, 1.00),
    ("nq", "Single-step", "Adaptive Retrieval", 28.20, 36.00, 33.00, 0.50, 0.56),
    ("nq", "Single-step", "Self-RAG*", 31.40, 39.00, 33.60, 0.63, 0.17),
    ("nq", "Single-step", "Adaptive-RAG (Ours)", 37.80, 47.30, 44.60, 1.00, 1.00),
    ("nq", "Single-step", "Multi-step Approach", 38.60, 47.80, 44.20, 5.04, 10.18),
    ("nq", "Single-step", "Adaptive-RAG w/ Oracle", 47.40, 57.10, 53.60, 1.10, 1.55),
    ("trivia", "Single-step", "No Retrieval", 25.00, 31.80, 27.00, 0.00, 0.13),
    ("trivia", "Single-step", "Single-step Approach", 53.60, 62.40, 60.20, 1.00, 1.00),
    ("trivia", "Single-step", "Adaptive Retrieval", 38.40, 46.90, 42.60, 0.50, 0.56),
    ("trivia", "Single-step", "Self-RAG*", 12.80, 29.30, 57.00, 0.68, 0.45),
    ("trivia", "Single-step", "Adaptive-RAG (Ours)", 52.20, 60.70, 58.20, 1.23, 1.54),
    ("trivia", "Single-step", "Multi-step Approach", 53.80, 62.40, 60.20, 5.28, 9.22),
    ("trivia", "Single-step", "Adaptive-RAG w/ Oracle", 61.60, 70.20, 66.40, 0.79, 1.10),
    ("musique", "Multi-step", "No Retrieval", 2.40, 10.70, 3.20, 0.00, 0.11),
    ("musique", "Multi-step", "Single-step Approach", 13.80, 22.80, 15.20, 1.00, 1.00),
    ("musique", "Multi-step", "Adaptive Retrieval", 6.40, 15.80, 8.00, 0.50, 0.55),
    ("musique", "Multi-step", "Self-RAG*", 1.60, 8.10, 12.00, 0.73, 0.51),
    ("musique", "Multi-step", "Adaptive-RAG (Ours)", 23.60, 31.80, 26.00, 3.22, 6.61),
    ("musique", "Multi-step", "Multi-step Approach", 23.00, 31.90, 25.80, 3.60, 7.58),
    ("musique", "Multi-step", "Adaptive-RAG w/ Oracle", 24.80, 38.50, 27.00, 1.98, 3.99),
    ("hotpotqa", "Multi-step", "No Retrieval", 16.60, 22.71, 17.20, 0.00, 0.11),
    ("hotpotqa", "Multi-step", "Single-step Approach", 34.40, 46.15, 36.40, 1.00, 1.00),
    ("hotpotqa", "Multi-step", "Adaptive Retrieval", 23.60, 32.22, 25.00, 0.50, 0.55),
    ("hotpotqa", "Multi-step", "Self-RAG*", 6.80, 17.53, 29.60, 0.73, 0.45),
    ("hotpotqa", "Multi-step", "Adaptive-RAG (Ours)", 42.00, 53.82, 44.40, 3.55, 5.99),
    ("hotpotqa", "Multi-step", "Multi-step Approach", 44.60, 56.54, 47.00, 5.53, 9.38),
    ("hotpotqa", "Multi-step", "Adaptive-RAG w/ Oracle", 51.20, 64.00, 54.80, 1.59, 2.77),
    ("2wikimultihopqa", "Multi-step", "No Retrieval", 27.40, 32.04, 27.80, 0.00, 0.10),
    ("2wikimultihopqa", "Multi-step", "Single-step Approach", 41.60, 47.90, 42.80, 1.00, 1.00),
    ("2wikimultihopqa", "Multi-step", "Adaptive Retrieval", 33.20, 39.44, 34.20, 0.50, 0.55),
    ("2wikimultihopqa", "Multi-step", "Self-RAG*", 4.60, 19.59, 38.80, 0.93, 0.49),
    ("2wikimultihopqa", "Multi-step", "Adaptive-RAG (Ours)", 40.60, 49.75, 46.40, 2.63, 4.68),
    ("2wikimultihopqa", "Multi-step", "Multi-step Approach", 49.60, 58.85, 55.40, 4.17, 7.37),
    ("2wikimultihopqa", "Multi-step", "Adaptive-RAG w/ Oracle", 53.00, 62.30, 59.40, 1.01, 1.69),
]

PAPER_TABLE_3 = [("No (A)", 0.35, 8.60), ("One (B)", 3.08, 53.33), ("Multi (C)", 27.18, 38.07)]
PAPER_TABLE_4 = [("Adaptive-RAG (Ours)", 46.94, 1084, 54.52, 30.52, 66.28, 65.45), ("w/o Binary", 43.43, 640, 60.30, 62.19, 65.70, 39.55), ("w/o Silver", 48.79, 1464, 40.00, 0.00, 53.98, 75.91)]
PAPER_TABLE_5 = [
    ("NQ (Single-hop)", "Which famous corporate logo changed to a flat colour/color sans serif font in its first major change since 1999?", "B (Single-step Approach)", "Microsoft", "A (Non Retrieval)", "Google"),
    ("MuSiQue (Multi-hop)", "Who is the child of the Italian navigator who explored the eastern coast of the continent César Gaytan was born in for the English?", "A (Non Retrieval)", "Giovanni Caboto/John Cabot", "C (Multi-step Approach)", "Sebastian Cabot"),
]
PAPER_TABLE_6 = [("Small (60M)", 45.83, 964, 53.48, 26.65, 70.62, 53.18), ("Base (223M)", 45.97, 983, 53.41, 26.42, 69.46, 56.82), ("Large (770M)", 46.94, 1084, 54.52, 30.52, 66.28, 65.45)]


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    return str(x)


def pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return x * 100.0 if abs(x) <= 1.0 else x


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def discover_router_runs(router_root: Path, gen_model: str, split: str) -> List[str]:
    runs = []
    pred_root = router_root / "predictions"
    if pred_root.exists():
        for child in sorted(pred_root.iterdir()):
            if child.is_dir() and (child / gen_model / split).exists():
                runs.append(child.name)
        if (pred_root / gen_model / split).exists():
            runs.append("default")
    return sorted(set(runs))


def router_summary_path(router_root: Path, run: str, gen_model: str, split: str) -> Path:
    p = router_root / "router_predictions" / run / gen_model / split / "router_summary.json"
    return p if p.exists() else router_root / "router_predictions" / gen_model / split / "router_summary.json"


def eval_summary_path(router_root: Path, run: str, gen_model: str, split: str, official: bool = False) -> Path:
    fname = "official_evaluation_summary.json" if official else "evaluation_summary.json"
    p = router_root / "predictions" / run / gen_model / split / fname
    return p if p.exists() else router_root / "predictions" / gen_model / split / fname


def routing_summary_path(router_root: Path, run: str, gen_model: str, split: str) -> Path:
    p = router_root / "predictions" / run / gen_model / split / "routing_summary.json"
    return p if p.exists() else router_root / "predictions" / gen_model / split / "routing_summary.json"


def load_run(router_root: Path, run: str, gen_model: str, split: str) -> Dict[str, Any]:
    return {
        "router_run": run,
        "router_summary": read_json(router_summary_path(router_root, run, gen_model, split)) or {},
        "eval_summary": read_json(eval_summary_path(router_root, run, gen_model, split, False)) or {},
        "official_eval_summary": read_json(eval_summary_path(router_root, run, gen_model, split, True)) or {},
        "routing_summary": read_json(routing_summary_path(router_root, run, gen_model, split)) or {},
    }


def metric(summary: Dict[str, Any], key: str) -> Optional[float]:
    w = summary.get("weighted_average")
    if isinstance(w, dict) and key in w:
        return pct(w[key])
    if key in summary:
        return pct(summary[key])
    if key == "acc" and "accuracy" in summary:
        return pct(summary["accuracy"])
    return None


def per_dataset_metric(summary: Dict[str, Any], dataset: str, key: str) -> Optional[float]:
    per = summary.get("per_dataset")
    if isinstance(per, dict) and dataset in per:
        return pct(per[dataset].get(key))
    return None


def label_counts(run_data: Dict[str, Any]) -> Dict[str, int]:
    for obj_name, key in [("router_summary", "label_counts"), ("routing_summary", "route_counts")]:
        obj = run_data.get(obj_name, {})
        counts = obj.get(key)
        if isinstance(counts, dict):
            return {str(k): int(v) for k, v in counts.items()}
    return {}


def adaptive_step(counts: Dict[str, int], multi_step_value: float) -> Optional[float]:
    total = sum(counts.values())
    if not total:
        return None
    return (counts.get("A", 0) * 0.0 + counts.get("B", 0) * 1.0 + counts.get("C", 0) * multi_step_value) / total


def split_to_repo_dir_suffix(split: str) -> tuple[str, str]:
    if split in {"validation", "test"}:
        return "test", "test_subsampled"
    return "dev_500", "dev_500_subsampled"


def system_dir(system: str, model: str, dataset: str) -> str:
    if system == "ircot_qa":
        return f"ircot_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1"
    if system == "oner_qa":
        return f"oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1"
    if system == "nor_qa":
        return f"nor_qa_{model}_{dataset}____prompt_set_1"
    raise ValueError(system)


def load_strategy_metric(pred_root: Path, gen_model: str, split: str, dataset: str, system: str, official: bool) -> Dict[str, Any]:
    repo_dir, suffix = split_to_repo_dir_suffix(split)
    run_dir = pred_root / repo_dir / system_dir(system, gen_model, dataset)
    prefix = "official_evaluation_metrics" if official else "evaluation_metrics"
    exact = run_dir / f"{prefix}__{dataset}_to_{dataset}__{suffix}.json"
    path = exact if exact.exists() else None
    if path is None and run_dir.exists():
        matches = sorted(run_dir.glob(f"{prefix}__*__{suffix}.json"))
        path = matches[0] if matches else None
    data = read_json(path) if path else None
    return data if isinstance(data, dict) else {}


def aggregate_strategy(pred_root: Path, gen_model: str, split: str, system: str, official: bool) -> Dict[str, Optional[float]]:
    rows = []
    for ds in DATASETS:
        m = load_strategy_metric(pred_root, gen_model, split, ds, system, official)
        if m:
            count = int(m.get("count", 500))
            rows.append((count, pct(m.get("em")), pct(m.get("f1")), pct(m.get("acc") or m.get("accuracy"))))
    out = {"em": None, "f1": None, "acc": None, "count": sum(r[0] for r in rows) if rows else None}
    for i, key in enumerate(["em", "f1", "acc"], start=1):
        vals = [(r[0], r[i]) for r in rows if r[i] is not None]
        if vals:
            out[key] = sum(c * v for c, v in vals) / sum(c for c, _ in vals)
    return out


def get_runs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.router_root)
    names = args.router_runs or discover_router_runs(root, args.gen_model, args.split)
    return [load_run(root, n, args.gen_model, args.split) for n in names]


def reproduce_table_1(args: argparse.Namespace) -> str:
    rows = [[llm, typ, method, fmt(em), fmt(f1), fmt(acc), fmt(step), fmt(time), "Paper"] for llm, typ, method, em, f1, acc, step, time in PAPER_TABLE_1]
    pred_root = Path(args.predictions_root)
    for system in SYSTEMS:
        m = aggregate_strategy(pred_root, args.gen_model, args.split, system, args.official)
        if any(m.get(k) is not None for k in ("em", "f1", "acc")):
            step = {"nor_qa": 0.0, "oner_qa": 1.0, "ircot_qa": args.multi_step_value}[system]
            rows.append([args.gen_model, "Local baseline", SYSTEM_DISPLAY[system], fmt(m["em"]), fmt(m["f1"]), fmt(m["acc"]), fmt(step), "—", "Local"])
    for r in get_runs(args):
        ev = r["official_eval_summary"] if args.official and r["official_eval_summary"] else r["eval_summary"]
        rows.append([args.gen_model, "Local router", f"Adaptive-RAG ({r['router_run']})", fmt(metric(ev, "em")), fmt(metric(ev, "f1")), fmt(metric(ev, "acc")), fmt(adaptive_step(label_counts(r), args.multi_step_value)), "—", "Local"])
    return md_table(["LLM", "Type", "Method", "EM", "F1", "Acc", "Step", "Time", "Source"], rows)


def reproduce_table_2(args: argparse.Namespace) -> str:
    rows = [[DATASET_DISPLAY[d], typ, method, fmt(em), fmt(f1), fmt(acc), fmt(step), fmt(time), "Paper"] for d, typ, method, em, f1, acc, step, time in PAPER_TABLE_2]
    for r in get_runs(args):
        ev = r["official_eval_summary"] if args.official and r["official_eval_summary"] else r["eval_summary"]
        for ds in DATASETS:
            rows.append([DATASET_DISPLAY[ds], "Local", f"Adaptive-RAG ({r['router_run']})", fmt(per_dataset_metric(ev, ds, "em")), fmt(per_dataset_metric(ev, ds, "f1")), fmt(per_dataset_metric(ev, ds, "acc")), "—", "—", "Local"])
    return md_table(["Dataset", "Type", "Method", "EM", "F1", "Acc", "Step", "Time", "Source"], rows)


def reproduce_table_3(args: argparse.Namespace) -> str:
    rows = [[label, fmt(t), fmt(p), "Paper"] for label, t, p in PAPER_TABLE_3]
    for r in get_runs(args):
        counts = label_counts(r)
        total = sum(counts.values())
        for lab in ["A", "B", "C"]:
            perc = counts.get(lab, 0) / total * 100 if total else None
            rows.append([f"{LABEL_DISPLAY[lab]} — {r['router_run']}", "—", fmt(perc), "Local"])
    return md_table(["Labels", "Time/Query (Sec.)", "Percentage (%)", "Source"], rows)


def reproduce_table_4(args: argparse.Namespace) -> str:
    rows = [[s, fmt(f1), fmt(step, 0), fmt(all_), fmt(no), fmt(one), fmt(multi), "Paper"] for s, f1, step, all_, no, one, multi in PAPER_TABLE_4]
    for r in get_runs(args):
        ev = r["official_eval_summary"] if args.official and r["official_eval_summary"] else r["eval_summary"]
        acc = r["router_summary"].get("accuracy") or r["router_summary"].get("val_accuracy")
        rows.append([f"Adaptive-RAG ({r['router_run']})", fmt(metric(ev, "f1")), fmt(adaptive_step(label_counts(r), args.multi_step_value), 0), fmt(pct(acc)), "—", "—", "—", "Local"])
    return md_table(["Training Strategy", "QA F1", "Step", "Cls. All", "Cls. No", "Cls. One", "Cls. Multi", "Source"], rows)


def reproduce_table_5(args: argparse.Namespace) -> str:
    rows = [[*r, "Paper"] for r in PAPER_TABLE_5]
    return md_table(["Dataset", "Question", "Adaptive Retrieval Type", "Adaptive Retrieval Answer", "Adaptive-RAG Type", "Adaptive-RAG Answer", "Source"], rows)


def reproduce_table_6(args: argparse.Namespace) -> str:
    rows = [[size, fmt(f1), fmt(step, 0), fmt(all_), fmt(no), fmt(one), fmt(multi), "Paper"] for size, f1, step, all_, no, one, multi in PAPER_TABLE_6]
    for r in get_runs(args):
        ev = r["official_eval_summary"] if args.official and r["official_eval_summary"] else r["eval_summary"]
        acc = r["router_summary"].get("accuracy") or r["router_summary"].get("val_accuracy")
        rows.append([r["router_run"], fmt(metric(ev, "f1")), fmt(adaptive_step(label_counts(r), args.multi_step_value), 0), fmt(pct(acc)), "—", "—", "—", "Local"])
    return md_table(["Classifier / Router", "QA F1", "Step", "Cls. All", "Cls. No", "Cls. One", "Cls. Multi", "Source"], rows)


def reproduce_all_tables(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "table_1.md": "# Table 1 — Average QA Results\n\n" + reproduce_table_1(args),
        "table_2.md": "# Table 2 — Per-Dataset FLAN-T5-XL Results\n\n" + reproduce_table_2(args),
        "table_3.md": "# Table 3 — Predicted Label Distribution and Elapsed Time\n\n" + reproduce_table_3(args),
        "table_4.md": "# Table 4 — Training-Data Ablation\n\n" + reproduce_table_4(args),
        "table_5.md": "# Table 5 — Case Study\n\n" + reproduce_table_5(args),
        "table_6.md": "# Table 6 — Classifier Size / Router Improvement\n\n" + reproduce_table_6(args),
    }


def build_results_md(args: argparse.Namespace, tables: Dict[str, str]) -> str:
    parts = [
        "# Adaptive-RAG Reproduction Results\n",
        "## Scope\n",
        "This run is configured for locally generated `flan_t5_xl` strategy outputs. Paper-reference rows for unrun settings are retained and local values are shown where artifacts exist.\n",
        "## Improvement\n",
        "Improvement idea: replace the paper's generative T5 router with a discriminative encoder router such as DeBERTa-v3. Hypothesis: query complexity routing is a short-text classification problem, so an encoder classifier can match or improve routing while avoiding generative decoding overhead.\n",
        "## Generated Tables\n",
    ]
    for name in sorted(tables):
        parts.append(f"- [{name.replace('.md', '').replace('_', ' ').title()}]({name})")
    parts.append("\n## Main Local Router Comparison\n")
    parts.append(tables["table_6.md"].split("\n\n", 1)[1])
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gen-model", default="flan_t5_xl")
    p.add_argument("--split", default="validation", choices=["validation", "test", "train", "dev"])
    p.add_argument("--router-root", default="router")
    p.add_argument("--predictions-root", default="predictions")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--router-runs", nargs="*", default=None)
    p.add_argument("--official", action="store_true")
    p.add_argument("--multi-step-value", type=float, default=4.69)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = reproduce_all_tables(args)
    for name, text in tables.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    (out_dir / "results.md").write_text(build_results_md(args, tables), encoding="utf-8")
    print(f"Wrote tables to {out_dir}")
    for name in sorted(tables):
        print(out_dir / name)
    print(out_dir / "results.md")


if __name__ == "__main__":
    main()
