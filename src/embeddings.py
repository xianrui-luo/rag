from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Sequence

from sentence_transformers import CrossEncoder, SentenceTransformer


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


class LocalReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_cross_encoder(model_name)

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)
