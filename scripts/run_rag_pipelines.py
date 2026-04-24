import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import extract_documents, extract_qa_records, load_records
from src.llm import LocalLLM
from src.pipeline import AdaptiveRAGPipeline
from src.retrieval.retriever import Retriever
from src.utils.config import load_yaml_config


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


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


def _init_distributed_if_needed(backend: str = "nccl"):
    if not is_distributed():
        return False

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return True


def _shard_questions(questions):
    world_size = get_world_size()
    rank = get_rank()
    indexed_questions = list(enumerate(questions))
    return indexed_questions[rank::world_size]


def main():
    parser = argparse.ArgumentParser(description="Run Adaptive RAG pipelines")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to runtime config")
    parser.add_argument("--questions", default=None, help="Path to questions or QA file")
    parser.add_argument("--corpus", default=None, help="Path to retrieval corpus")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--strategy", choices=["no", "single", "multi", "all"], default=None)
    parser.add_argument("--index-dir", default=None, help="Directory containing index.faiss and documents.json")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the index even if saved files exist")
    parser.add_argument("--save-index", action="store_true", help="Save a newly built index to --index-dir")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--ddp-backend", default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    distributed = _init_distributed_if_needed(args.ddp_backend)

    config = load_yaml_config(args.config)
    questions_path = args.questions or config.get("questions")
    corpus_path = args.corpus or config.get("corpus")
    output_path = Path(args.output or config.get("output", "outputs/predictions.json"))
    strategy = args.strategy or config.get("strategy", "all")
    index_dir = args.index_dir or config.get("index_dir")
    rebuild_index = args.rebuild_index or bool(config.get("rebuild_index", False))
    save_index = args.save_index or bool(config.get("save_index", False))
    encoder_name = args.encoder or config.get("encoder_name", "all-MiniLM-L6-v2")
    model_name = args.model_name or config.get("model_name", "mistralai/Mistral-7B-v0.1")

    if questions_path is None:
        raise ValueError("A questions path must be provided via --questions or the config file")
    if corpus_path is None:
        raise ValueError("A corpus path must be provided via --corpus or the config file")

    questions = _prepare_questions(questions_path)
    sharded_questions = _shard_questions(questions) if distributed else list(enumerate(questions))
    local_indices = [index for index, _ in sharded_questions]
    local_questions = [question for _, question in sharded_questions]

    llm = LocalLLM(model_name=model_name)
    retriever = Retriever(encoder_name=encoder_name)

    loaded_index = False
    if index_dir and not rebuild_index:
        index_path = Path(index_dir)
        if (index_path / "index.faiss").exists() and (index_path / "documents.json").exists():
            retriever.load_index(index_path)
            loaded_index = True

    if not loaded_index:
        if distributed and index_dir:
            if is_main_process():
                corpus_documents = extract_documents(load_records(corpus_path))
                retriever.build_index(corpus_documents)
                retriever.save_index(index_dir)
            dist.barrier()
            retriever.load_index(index_dir)
        else:
            corpus_documents = extract_documents(load_records(corpus_path))
            retriever.build_index(corpus_documents)

            if index_dir and save_index:
                retriever.save_index(index_dir)

    pipeline = AdaptiveRAGPipeline(llm, retriever)

    if strategy == "no":
        local_result = {"no": pipeline.no_retrieval(local_questions)}
    elif strategy == "single":
        local_result = {"single": pipeline.single_step(local_questions)}
    elif strategy == "multi":
        local_result = {"multi": pipeline.multi_step(local_questions)}
    else:
        local_result = pipeline.run_all(local_questions)

    if distributed:
        gathered = [None for _ in range(get_world_size())]
        payload = {
            "indices": local_indices,
            "result": local_result,
        }
        dist.all_gather_object(gathered, payload)

        if is_main_process():
            merged = {}
            keys = gathered[0]["result"].keys() if gathered else []
            for key in keys:
                merged[key] = [None] * len(questions)

            for item in gathered:
                indices = item["indices"]
                result = item["result"]
                for key, values in result.items():
                    for idx, value in zip(indices, values):
                        merged[key][idx] = value

            result = merged
        else:
            result = None
    else:
        result = local_result

    if not distributed or is_main_process():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()