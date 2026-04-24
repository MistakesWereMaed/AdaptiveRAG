import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import extract_qa_records, load_records
from src.llm import LocalLLM
from src.pipeline import AdaptiveRAGPipeline
from src.retriever import FaissIVFRetriever
from src.utils.config import load_yaml_config


def _prepare_questions(source):
    records = load_records(source)
    questions = []
    for record in records:
        if isinstance(record, str):
            text = record.strip()
            if text:
                questions.append(text)
            continue

        if isinstance(record, dict):
            question = record.get("question") or record.get("query")
            if isinstance(question, str) and question.strip():
                questions.append(question.strip())

    if not questions:
        questions = [record["question"] for record in extract_qa_records(records)]

    return questions


def _shard_questions(questions, rank: int, world_size: int):
    indexed_questions = list(enumerate(questions))
    return indexed_questions[rank::world_size]


def _require_index_dir(index_dir: str | None) -> Path:
    if not index_dir:
        raise ValueError(
            "A prebuilt index directory is required. Provide --index-dir with documents.json. "
            "Run scripts/build_index.py first."
        )

    index_path = Path(index_dir)
    documents_path = index_path / "documents.json"
    if not documents_path.exists():
        raise FileNotFoundError(
            f"Prebuilt index not found at {index_path}. Expected {documents_path.name}. "
            "Run scripts/build_index.py first."
        )
    return index_path


def _strategy_output_path(base_output: Path, strategy_name: str) -> Path:
    return base_output.with_name(f"{base_output.stem}.{strategy_name}{base_output.suffix}")


def _write_strategy_outputs(result: dict, base_output: Path) -> None:
    base_output.parent.mkdir(parents=True, exist_ok=True)
    for strategy_name, values in result.items():
        strategy_path = _strategy_output_path(base_output, strategy_name)
        with strategy_path.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2)


def _run_shard(
    questions_path: str,
    strategy: str,
    output_path: Path,
    index_dir: str,
    encoder_name: str,
    model_name: str,
    rank: int,
    world_size: int,
    shard_dir: Path,
):
    questions = _prepare_questions(questions_path)
    sharded_questions = _shard_questions(questions, rank=rank, world_size=world_size)
    local_indices = [index for index, _ in sharded_questions]
    local_questions = [question for _, question in sharded_questions]

    llm = LocalLLM(model_name=model_name)
    retriever = FaissIVFRetriever(encoder_name=encoder_name)
    retriever.load(_require_index_dir(index_dir))

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    if strategy == "no":
        local_result = {"no": pipeline.no_retrieval(local_questions)}
    elif strategy == "single":
        local_result = {"single": pipeline.single_step(local_questions)}
    elif strategy == "multi":
        local_result = {"multi": pipeline.multi_step(local_questions)}
    else:
        local_result = pipeline.run_all(local_questions)

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_payload = {
        "indices": local_indices,
        "result": local_result,
    }
    shard_file = shard_dir / f"{output_path.stem}.rank{rank}.json"
    with shard_file.open("w", encoding="utf-8") as handle:
        json.dump(shard_payload, handle, indent=2)


def _launch_workers(
    args,
    questions_path: str,
    output_path: Path,
    strategy: str,
    index_dir: str,
    encoder_name: str,
    model_name: str,
    shard_dir: Path,
):
    questions = _prepare_questions(questions_path)
    shard_dir.mkdir(parents=True, exist_ok=True)

    for old_shard in shard_dir.glob(f"{output_path.stem}.rank*.json"):
        old_shard.unlink()

    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [item.strip() for item in raw_visible_devices.split(",") if item.strip()]

    worker_processes = []
    for rank in range(args.num_procs):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            args.config,
            "--questions",
            questions_path,
            "--output",
            str(output_path),
            "--strategy",
            strategy,
            "--index-dir",
            index_dir,
            "--encoder",
            encoder_name,
            "--model-name",
            model_name,
            "--worker-rank",
            str(rank),
            "--world-size",
            str(args.num_procs),
            "--num-procs",
            "1",
            "--shard-dir",
            str(shard_dir),
        ]

        env = os.environ.copy()
        if visible_devices and rank < len(visible_devices):
            env["CUDA_VISIBLE_DEVICES"] = visible_devices[rank]

        worker_processes.append(subprocess.Popen(cmd, env=env))

    for process in tqdm(worker_processes, desc="Waiting for worker shards", unit="worker"):
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Worker process failed with exit code {return_code}")

    merged = {}
    for rank in range(args.num_procs):
        shard_file = shard_dir / f"{output_path.stem}.rank{rank}.json"
        if not shard_file.exists():
            raise FileNotFoundError(f"Expected shard output missing: {shard_file}")

        with shard_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        indices = payload["indices"]
        result = payload["result"]
        for key in result.keys():
            if key not in merged:
                merged[key] = [None] * len(questions)
        for key, values in result.items():
            for idx, value in zip(indices, values):
                merged[key][idx] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive RAG pipelines")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to runtime config")
    parser.add_argument("--questions", default=None, help="Path to questions or QA file")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--strategy", choices=["no", "single", "multi", "all"], default=None)
    parser.add_argument("--index-dir", default=None, help="Directory containing saved retriever metadata (documents.json)")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--num-procs", type=int, default=1, help="Number of local worker processes for dataset sharding")
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--shard-dir", default=None, help="Directory for temporary shard outputs")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    questions_path = args.questions or config.get("questions")
    output_path = Path(args.output or config.get("output", "outputs/predictions.json"))
    strategy = args.strategy or config.get("strategy", "all")
    index_dir = args.index_dir or config.get("index_dir")
    encoder_name = args.encoder or config.get("encoder_name", "all-MiniLM-L6-v2")
    model_name = args.model_name or config.get("model_name", "mistralai/Mistral-7B-v0.1")
    shard_dir = Path(args.shard_dir or output_path.parent / f"{output_path.stem}_shards")

    if questions_path is None:
        raise ValueError("A questions path must be provided via --questions or the config file")
    _require_index_dir(index_dir)

    should_cleanup_shards = args.worker_rank is None

    try:
        # Launcher mode: spawn local workers with plain python and merge shard files.
        if args.worker_rank is None and args.num_procs > 1:
            _launch_workers(
                args=args,
                questions_path=questions_path,
                output_path=output_path,
                strategy=strategy,
                index_dir=index_dir,
                encoder_name=encoder_name,
                model_name=model_name,
                shard_dir=shard_dir,
            )
            merged_payload_path = output_path
            with merged_payload_path.open("r", encoding="utf-8") as handle:
                merged_payload = json.load(handle)
            _write_strategy_outputs(merged_payload, output_path)
            return

        # Single-process mode or worker mode.
        rank = 0 if args.worker_rank is None else args.worker_rank
        world_size = 1 if args.worker_rank is None else args.world_size

        _run_shard(
            questions_path=questions_path,
            strategy=strategy,
            output_path=output_path,
            index_dir=index_dir,
            encoder_name=encoder_name,
            model_name=model_name,
            rank=rank,
            world_size=world_size,
            shard_dir=shard_dir,
        )

        if args.worker_rank is None:
            shard_file = shard_dir / f"{output_path.stem}.rank0.json"
            with shard_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            _write_strategy_outputs(payload["result"], output_path)
    finally:
        if should_cleanup_shards:
            shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()