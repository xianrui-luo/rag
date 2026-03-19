from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.config import Settings
from src.embeddings import LocalEmbeddingService
from src.llm_client import LLMClient
from src.vectorstore import VectorStore


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = LocalEmbeddingService(settings.embedding_model)
        self.vectorstore = VectorStore(settings.chroma_dir)
        self.llm = LLMClient(settings)

    def ask(self, kb_name: str, question: str, top_k: int | None = None) -> Tuple[str, List[Dict[str, Any]]]:
        if not kb_name.strip():
            raise ValueError("Knowledge base name is required")
        if not question.strip():
            raise ValueError("Question is required")

        query_embedding = self.embeddings.embed_query(question)
        results = self.vectorstore.query(kb_name=kb_name.strip(), query_embedding=query_embedding, top_k=top_k or self.settings.top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return "No indexed content found for this knowledge base. Refresh or rebuild the index first.", []

        context_blocks: List[Dict[str, Any]] = []
        for content, metadata, distance in zip(documents, metadatas, distances):
            context_blocks.append(
                {
                    "content": content,
                    "relative_path": metadata["relative_path"],
                    "chunk_index": metadata["chunk_index"],
                    "distance": round(float(distance), 4),
                }
            )

        answer = self.llm.generate_answer(question, context_blocks)
        return answer, context_blocks
