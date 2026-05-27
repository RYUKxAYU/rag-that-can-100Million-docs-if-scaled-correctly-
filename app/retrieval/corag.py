from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.retrieval.orchestrator import HybridRetrievalOrchestrator
from app.retrieval.schema import RetrievalResult


@dataclass(frozen=True)
class RetrievalCriticResult:
    confidence_score: float
    is_sufficient: bool
    reason: str
    top_score: float
    average_score: float


class RetrievalCritic:
    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold

    def evaluate(self, results: List[RetrievalResult], query: str) -> RetrievalCriticResult:
        if not results:
            return RetrievalCriticResult(
                confidence_score=0.0,
                is_sufficient=False,
                reason="no_results",
                top_score=0.0,
                average_score=0.0,
            )

        top_score = results[0].score
        average_score = sum(result.score for result in results) / len(results)
        query_length = len(str(query).split())
        richness = min(1.0, len(results) / 10.0)
        confidence_score = min(1.0, (top_score * 0.45) + (average_score * 0.35) + (richness * 0.2))
        is_sufficient = confidence_score >= self.threshold and top_score > 0.05
        reason = "sufficient" if is_sufficient else "low_confidence"

        return RetrievalCriticResult(
            confidence_score=round(confidence_score, 3),
            is_sufficient=is_sufficient,
            reason=reason,
            top_score=round(top_score, 3),
            average_score=round(average_score, 3),
        )


class QueryRewriter:
    def rewrite(self, query: str, critique: RetrievalCriticResult) -> str:
        normalized = str(query).strip()
        if critique.is_sufficient:
            return normalized

        if not normalized:
            return "Please provide a more detailed and focused query."

        lower = normalized.lower()
        if "relationship" in lower or "connect" in lower or "link" in lower:
            return f"{normalized} Provide the causal chain and supporting evidence for each connection."

        if "why" in lower or "how" in lower or "compare" in lower:
            return f"{normalized} Include step-by-step reasoning and citations."

        return f"{normalized} Clarify the intent and emphasize the most relevant supporting facts."


class IterativeRetrievalLoop:
    def __init__(
        self,
        retriever: HybridRetrievalOrchestrator,
        critic: Optional[RetrievalCritic] = None,
        rewriter: Optional[QueryRewriter] = None,
        max_iterations: int = 3,
    ) -> None:
        self.retriever = retriever
        self.critic = critic or RetrievalCritic()
        self.rewriter = rewriter or QueryRewriter()
        self.max_iterations = max_iterations

    async def run(
        self,
        query: str,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        query_text = query_text or query
        history: List[Dict[str, Any]] = []
        final_results: List[RetrievalResult] = []
        current_query = query_text

        for iteration in range(1, self.max_iterations + 1):
            results = await self.retriever.retrieve(
                query_embedding=query_embedding,
                query_text=current_query,
                top_k=top_k,
                use_bm25=True,
                use_vector=True,
            )
            critique = self.critic.evaluate(results, current_query)
            history.append(
                {
                    "iteration": iteration,
                    "query": current_query,
                    "results": results,
                    "critique": critique,
                }
            )
            if critique.is_sufficient or iteration == self.max_iterations:
                final_results = results
                break
            current_query = self.rewriter.rewrite(current_query, critique)

        return {
            "query": query,
            "final_query": current_query,
            "iterations": len(history),
            "history": history,
            "final_results": final_results,
            "sufficient": history[-1]["critique"].is_sufficient,
        }


class RecursiveRetrievalOrchestrator:
    def __init__(
        self,
        iterative_loop: IterativeRetrievalLoop,
        max_depth: int = 2,
    ) -> None:
        self.iterative_loop = iterative_loop
        self.max_depth = max_depth

    async def recursive_retrieve(
        self,
        query: str,
        query_embedding: List[float],
        classification: str = "factual",
        query_text: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        query_text = query_text or query
        base_run = await self.iterative_loop.run(query, query_embedding, query_text=query_text, top_k=top_k)
        recursive_path: List[Dict[str, Any]] = [
            {
                "depth": 0,
                "query": base_run["final_query"],
                "iterations": base_run["iterations"],
                "sufficient": base_run["sufficient"],
            }
        ]

        if classification == "multi-hop" and not base_run["sufficient"] and self.max_depth > 0:
            follow_up = self._construct_recursive_query(base_run["final_query"], base_run["final_results"])
            child_run = await self.iterative_loop.run(follow_up, query_embedding, query_text=follow_up, top_k=top_k)
            recursive_path.append(
                {
                    "depth": 1,
                    "query": follow_up,
                    "iterations": child_run["iterations"],
                    "sufficient": child_run["sufficient"],
                }
            )
            return {
                "root": base_run,
                "recursive_steps": recursive_path,
                "child": child_run,
            }

        return {
            "root": base_run,
            "recursive_steps": recursive_path,
            "child": None,
        }

    def _construct_recursive_query(self, query: str, results: List[RetrievalResult]) -> str:
        sources = ", ".join(result.item.id for result in results[:3])
        if sources:
            return f"{query} Use the retrieved items {sources} to trace the next retrieval step and explain the relationship."
        return f"{query} Provide an additional focused retrieval step for deeper context."
