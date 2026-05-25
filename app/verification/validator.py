import re
from typing import List, Sequence, Set, Tuple

from app.verification.citation_parser import CitationParser
from app.retrieval.schema import RetrievalResult

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class SentenceValidator:
    def __init__(self, candidates: Sequence[RetrievalResult]):
        self.candidate_ids: Set[str] = {result.item.id for result in candidates}

    def split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
        return sentences

    def validate(self, text: str) -> Tuple[List[str], List[str]]:
        valid_sentences: List[str] = []
        invalid_sentences: List[str] = []

        for sentence in self.split_sentences(text):
            citations = CitationParser.extract_citation_tokens(sentence)
            if not citations:
                invalid_sentences.append(sentence)
                continue

            if not set(citations).issubset(self.candidate_ids):
                invalid_sentences.append(sentence)
                continue

            valid_sentences.append(sentence)

        return valid_sentences, invalid_sentences
