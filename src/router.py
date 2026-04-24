from typing import Any

import torch

from pipeline import AdaptiveRAGPipeline


def _to_int(strategy: Any) -> int:
    if isinstance(strategy, torch.Tensor):
        return int(strategy.item())
    if isinstance(strategy, (list, tuple)) and strategy:
        return _to_int(strategy[0])
    return int(strategy)


def route(question, classifier, llm, retriever, k: int = 5, steps: int = 2):
    if hasattr(classifier, "predict"):
        strategy = classifier.predict(question)
    else:
        strategy = classifier(question)
    strategy_index = _to_int(strategy)
    pipeline = AdaptiveRAGPipeline(llm=llm, retriever=retriever)

    if strategy_index == 0:
        return pipeline.no_retrieval([question])[0]
    if strategy_index == 1:
        return pipeline.single_step([question], k=k)[0]
    return pipeline.multi_step([question], steps=steps, k=max(1, k // 2))[0]