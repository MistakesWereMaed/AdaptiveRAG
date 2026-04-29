from typing import Any

import torch

from cs6263_template.src.myproject.src.rag.pipeline import AdaptiveRAGPipeline


def _to_int(strategy: Any) -> int:
    if isinstance(strategy, torch.Tensor):
        return int(strategy.item())
    if isinstance(strategy, (list, tuple)) and strategy:
        return _to_int(strategy[0])
    return int(strategy)


def route(
    question,
    classifier,
    llm,
    retriever,
    single_k: int = 5,
    multi_k: int = 3,
    steps: int = 2,
):
    print("[router] Routing a question", flush=True)
    if hasattr(classifier, "predict"):
        strategy = classifier.predict(question)
    else:
        strategy = classifier(question)
    strategy_index = _to_int(strategy)
    pipeline = AdaptiveRAGPipeline(llm=llm, retriever=retriever)

    if strategy_index == 0:
        return pipeline.no_retrieval([question])[0]
    if strategy_index == 1:
        return pipeline.single_step([question], k=single_k)[0]
    return pipeline.multi_step([question], steps=steps, k=multi_k)[0]