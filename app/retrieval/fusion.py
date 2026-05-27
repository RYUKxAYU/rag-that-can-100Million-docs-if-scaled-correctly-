from typing import List, Sequence

from app.retrieval.schema import RetrievalResult


def rrf_fusion(
    bm25_results: Sequence[RetrievalResult],
    vector_results: Sequence[RetrievalResult],
    top_k: int = 10,
    k: int = 60,
) -> List[RetrievalResult]:
    scores = {}
    items = {}

    for rank, result in enumerate(bm25_results):
        if result.item.id not in scores:
            scores[result.item.id] = 0.0
            items[result.item.id] = result.item
        scores[result.item.id] += 1.0 / (k + rank + 1)

    for rank, result in enumerate(vector_results):
        if result.item.id not in scores:
            scores[result.item.id] = 0.0
            items[result.item.id] = result.item
        scores[result.item.id] += 1.0 / (k + rank + 1)

    fused = [
        RetrievalResult(
            item=items[item_id],
            score=scores[item_id],
            rank=0,
            source="hybrid",
        )
        for item_id in scores
    ]
    fused.sort(key=lambda result: (-result.score, result.item.id))
    return [
        RetrievalResult(item=result.item, score=result.score, rank=index + 1, source=result.source)
        for index, result in enumerate(fused[:top_k])
    ]
