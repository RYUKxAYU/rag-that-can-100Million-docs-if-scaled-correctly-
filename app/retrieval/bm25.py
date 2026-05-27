import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from app.retrieval.schema import RetrievalItem, RetrievalResult

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text) if token]


class BM25Retriever:
    def __init__(
        self,
        documents: Sequence[RetrievalItem],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_count = len(self.documents)
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0
        self._index_documents()

    def _index_documents(self) -> None:
        total_length = 0
        for item in self.documents:
            tokens = _tokenize(item.text)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            tf = Counter(tokens)
            self.doc_term_freqs.append(dict(tf))
            for term in tf.keys():
                self.document_frequencies[term] += 1
        self.avg_doc_len = total_length / max(1, self.doc_count)

    def _idf(self, term: str) -> float:
        doc_freq = self.document_frequencies.get(term, 0)
        return math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        query_freq = Counter(query_terms)
        scores: Dict[str, float] = {}
        for idx, item in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            for term, freq in query_freq.items():
                tf = self.doc_term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_len))
                score += idf * numerator / denominator
            if score > 0:
                scores[item.id] = score
        ranked = sorted(
            [RetrievalResult(item=item, score=scores[item.id], rank=rank + 1, source="bm25")
             for rank, item in enumerate(self.documents) if item.id in scores],
            key=lambda result: (-result.score, result.item.id),
        )
        return ranked[:top_k]
