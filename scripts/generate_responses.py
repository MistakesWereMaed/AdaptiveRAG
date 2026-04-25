import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm.auto import tqdm

from src.data.preprocessing import extract_qa_records, load_records
from src.llm import LocalLLM
from src.pipeline import AdaptiveRAGPipeline
from src.retriever import FaissIVFRetriever
from src.utils.config import load_yaml_config


# ------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------
def _prepare_records(source: str) -> List[Dict[str, Any]]:
    records = load_records(source)

    structured = []

    for i, r in enumerate(records):
        if isinstance(r, dict):
            q = r.get("question") or r.get("query")
            if not q:
                continue

            structured.append({
                "id": r.get("_id", i),
                "question": q.strip(),
                "gold": r.get("answer", None)
            })

        elif isinstance(r, str) and r.strip():
            structured.append({
                "id": i,
                "question": r.strip(),
                "gold": None
            })

    # fallback
    if not structured:
        qa_records = extract_qa_records(records)
        for i, r in enumerate(qa_records):
            structured.append({
                "id": r.get("_id", i),
                "question": r["question"],
                "gold": r.get("answer", None)
            })

    return structured


# ------------------------------------------------------------
# IO utilities
# ------------------------------------------------------------
def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _require_index_dir(index_dir: str) -> Path:
    p = Path(index_dir)
    if not (p / "documents.json").exists():
        raise FileNotFoundError(f"Missing index at {p}. Run build_index.py first.")
    return p


def _stage_output_path(base_output: Path, stage: str) -> Path:
    return base_output.with_name(f"{base_output.stem}-{stage}{base_output.suffix}")


def _write_stage(
    base_output: Path,
    stage: str,
    records: List[Dict[str, Any]],
    predictions: List[str],
):
    path = _stage_output_path(base_output, stage)
    path.parent.mkdir(parents=True, exist_ok=True)

    assert len(records) == len(predictions)

    output = []
    for r, pred in zip(records, predictions):
        output.append({
            "id": r["id"],
            "question": r["question"],
            "prediction": pred.strip(),
            "gold": r["gold"],
        })

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ------------------------------------------------------------
# Stage execution
# ------------------------------------------------------------
def _run_stage(
    pipeline: AdaptiveRAGPipeline,
    stage: str,
    questions: List[str],
) -> List[str]:

    if stage == "no-rag":
        return pipeline.no_retrieval(questions)

    if stage == "single":
        return pipeline.single_step(questions)

    if stage == "multi":
        return pipeline.multi_step(questions)

    raise ValueError(f"Unknown strategy: {stage}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run Adaptive RAG (single process)")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    # ---- Load configs ----
    cfg = load_yaml_config(args.config)
    llm_cfg = load_yaml_config(cfg["llm_config"])
    retr_cfg = load_yaml_config(cfg["retriever_config"])

    # ---- Normalize paths ----
    output_path = _as_path(cfg["output"])
    index_dir = _require_index_dir(cfg["index_dir"])

    # ---- Prepare data ----
    records = _prepare_records(cfg["questions"])
    questions = [r["question"] for r in records]

    # ---- Init components ----
    llm = LocalLLM(llm_cfg)

    retriever = FaissIVFRetriever(
        encoder_name=retr_cfg["encoder_name"]
    )
    retriever.load(index_dir)

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    # ---- Determine stages ----
    strategy = cfg["strategy"]

    if strategy == "all":
        stages = ["no-rag", "single", "multi"]
    elif isinstance(strategy, list):
        stages = strategy
    else:
        stages = [strategy]

    # ---- Run stages sequentially ----
    for stage in stages:
        print(f"\n=== Running stage: {stage} ===")

        predictions = _run_stage(pipeline, stage, questions)

        # ---- Save structured outputs immediately ----
        _write_stage(output_path, stage, records, predictions)

        print(f"Saved {stage} → {_stage_output_path(output_path, stage)}")


if __name__ == "__main__":
    main()