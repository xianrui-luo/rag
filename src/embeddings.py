from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class LocalEmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        encoded = self._model.encode(list(texts), normalize_embeddings=True)
        return encoded.tolist()

    def embed_query(self, text: str) -> List[float]:
        encoded = self._model.encode([text], normalize_embeddings=True)
        return encoded[0].tolist()


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)
