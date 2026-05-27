import asyncio
import hashlib
import math
from typing import List, Optional, Sequence


class EmbeddingService:
    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> List[float]:
        raise NotImplementedError


class BGEM3EmbeddingService(EmbeddingService):
    def __init__(
        self,
        embedding_dim: int = 1536,
        batch_size: int = 16,
        model_path: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.model_path = model_path

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings.extend(await self._embed_batch(batch))
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        result = await self._embed_batch([query])
        return result[0]

    async def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        await asyncio.sleep(0)
        return [self._embed_text(text) for text in batch]

    def _embed_text(self, text: str) -> List[float]:
        if text is None:
            text = ""
        base_digest = hashlib.blake2b(text.encode("utf-8"), digest_size=64).digest()
        vector: List[float] = []
        for idx in range(self.embedding_dim):
            seed = hashlib.blake2b(base_digest + idx.to_bytes(2, "little"), digest_size=4).digest()
            value = int.from_bytes(seed, "big", signed=False) / 0xFFFFFFFF
            vector.append(value * 2.0 - 1.0)
        length = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / length for value in vector]
