from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.api.models.Collection import Collection


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "documents"):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_file(self, kb_name: str, relative_path: str) -> None:
        self._collection.delete(
            where={
                "$and": [
                    {"kb_name": kb_name},
                    {"relative_path": relative_path},
                ]
            }
        )

    def delete_by_kb(self, kb_name: str) -> None:
        self._collection.delete(where={"kb_name": kb_name})

    def query(self, kb_name: str, query_embedding: List[float], top_k: int) -> Dict[str, Any]:
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"kb_name": kb_name},
            include=["documents", "metadatas", "distances"],
        )

    def get_chunks(self, kb_name: str) -> Dict[str, Any]:
        return self._collection.get(
            where={"kb_name": kb_name},
            include=["documents", "metadatas"],
        )
