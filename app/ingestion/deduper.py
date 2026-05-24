import hashlib
import re
from typing import List, Set, Tuple

from app.ingestion.schema import Chunk

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [token for token in _WORD_PATTERN.findall(text.lower()) if token]


def _shingles(tokens: List[str], size: int = 5) -> Set[str]:
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _minhash_signature(shingles: Set[str], num_perm: int = 64) -> Tuple[int, ...]:
    if not shingles:
        return tuple([0] * num_perm)

    signature: List[int] = []
    for idx in range(num_perm):
        seed = idx.to_bytes(4, "little", signed=False)
        values = []
        for shingle in shingles:
            digest = hashlib.blake2b(seed + shingle.encode("utf-8"), digest_size=8).digest()
            values.append(int.from_bytes(digest, "big"))
        signature.append(min(values))
    return tuple(signature)


def _minhash_similarity(sig_a: Tuple[int, ...], sig_b: Tuple[int, ...]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matching = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matching / len(sig_a)


class MinHashDeduplicator:
    def __init__(self, threshold: float = 0.85, num_perm: int = 64, shingle_size: int = 5):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._signatures: List[Tuple[int, ...]] = []

    def is_duplicate(self, chunk: Chunk) -> bool:
        signature = self._signature(chunk.text)
        for existing in self._signatures:
            if _minhash_similarity(existing, signature) >= self.threshold:
                return True
        self._signatures.append(signature)
        return False

    def _signature(self, text: str) -> Tuple[int, ...]:
        tokens = _tokenize(text)
        shingles = _shingles(tokens, size=self.shingle_size)
        return _minhash_signature(shingles, num_perm=self.num_perm)
