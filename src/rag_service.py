from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.config import Settings
from src.embeddings import LocalEmbeddingService, LocalReranker
from src.llm_client import LLMClient
from src.metadata_store import MetadataStore
from src.vectorstore import VectorStore


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = LocalEmbeddingService(settings.embedding_model)
        self.vectorstore = VectorStore(settings.chroma_dir)
        self.store = MetadataStore(settings.sqlite_path)
        self.llm = LLMClient(settings)
        self.reranker: LocalReranker | None = None

    def ask(
        self,
        kb_name: str,
        question: str,
        top_k: int | None = None,
        history: List[Dict[str, Any]] | None = None,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        if not kb_name.strip():
            raise ValueError("Knowledge base name is required")
        if not question.strip():
            raise ValueError("Question is required")

        final_top_k = top_k or self.settings.top_k
        retrieval_question = self._rewrite_question(question, history)
        candidates = self._retrieve_candidates(kb_name.strip(), retrieval_question, final_top_k)
        if not candidates:
            return (
                "No indexed content found for this knowledge base. Refresh or rebuild the index first.",
                [],
                retrieval_question,
            )

        context_blocks = self._expand_context_blocks(kb_name.strip(), candidates, final_top_k)
        answer = self.llm.generate_answer(question, context_blocks, history)
        return answer, context_blocks, retrieval_question

    def _rewrite_question(self, question: str, history: List[Dict[str, Any]] | None) -> str:
        if not self.settings.enable_query_rewrite or not history:
            return question
        rewritten = self.llm.rewrite_query(question, history)
        return rewritten.strip() or question

    def _retrieve_candidates(self, kb_name: str, question: str, top_k: int) -> List[Dict[str, Any]]:
        vector_candidates = self._vector_candidates(kb_name, question)
        if self.settings.enable_hybrid_retrieval:
            lexical_candidates = self._lexical_candidates(kb_name, question)
            if not vector_candidates and not lexical_candidates:
                return []
            candidates = self._fuse_candidates(vector_candidates, lexical_candidates)
        else:
            if not vector_candidates:
                return []
            candidates = vector_candidates

        reranker = self._get_reranker()
        if reranker is not None:
            rerank_limit = min(len(candidates), max(top_k, self.settings.retrieval_candidates))
            reranked = self._rerank(question, candidates[:rerank_limit], reranker)
            return reranked + candidates[rerank_limit:]
        return candidates

    def _vector_candidates(self, kb_name: str, question: str) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(question)
        results = self.vectorstore.query(
            kb_name=kb_name,
            query_embedding=query_embedding,
            top_k=self.settings.retrieval_candidates,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        candidates: List[Dict[str, Any]] = []
        for idx, (content, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            candidate = self._candidate_from_record(
                content=content,
                metadata=metadata,
                source_id=ids[idx] if idx < len(ids) else None,
            )
            candidate["distance"] = round(float(distance), 4)
            candidate["vector_rank"] = idx + 1
            candidate["retrieval_modes"] = ["vector"]
            candidates.append(candidate)
        return candidates

    def _lexical_candidates(self, kb_name: str, question: str) -> List[Dict[str, Any]]:
        match_query = _build_fts_query(question)
        if not match_query:
            return []
        rows = self.store.search_chunks_fts(kb_name, match_query, self.settings.lexical_candidates)
        if not rows:
            return []
        ranked: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            candidate = self._candidate_from_record(
                content=row.get("content") or "",
                metadata=row,
                source_id=row.get("chunk_id"),
            )
            candidate["lexical_score"] = round(float(-row.get("score", 0.0)), 4)
            candidate["lexical_rank"] = idx + 1
            candidate["retrieval_modes"] = ["lexical"]
            ranked.append(candidate)
        return ranked

    def _fuse_candidates(
        self,
        vector_candidates: List[Dict[str, Any]],
        lexical_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        for candidates, label in ((vector_candidates, "vector"), (lexical_candidates, "lexical")):
            for rank, candidate in enumerate(candidates, start=1):
                key = str(candidate["source_id"])
                merged = fused.setdefault(key, dict(candidate))
                merged.setdefault("retrieval_modes", [])
                if label not in merged["retrieval_modes"]:
                    merged["retrieval_modes"].append(label)
                merged["fusion_score"] = merged.get("fusion_score", 0.0) + (1.0 / (60 + rank))
                if label == "vector":
                    merged["vector_rank"] = candidate.get("vector_rank", rank)
                    merged["distance"] = candidate.get("distance")
                if label == "lexical":
                    merged["lexical_rank"] = candidate.get("lexical_rank", rank)
                    merged["lexical_score"] = candidate.get("lexical_score")
        return sorted(fused.values(), key=lambda item: item.get("fusion_score", 0.0), reverse=True)

    def _rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        reranker: LocalReranker,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        passages = [candidate["content"] for candidate in candidates]
        scores = reranker.score(question, passages)
        reranked = []
        for candidate, score in zip(candidates, scores):
            updated = dict(candidate)
            updated["rerank_score"] = round(float(score), 4)
            reranked.append(updated)
        return sorted(reranked, key=lambda item: item.get("rerank_score", 0.0), reverse=True)

    def _get_reranker(self) -> LocalReranker | None:
        if not self.settings.enable_reranker:
            return None
        if self.reranker is None:
            self.reranker = LocalReranker(self.settings.reranker_model)
        return self.reranker

    def _candidate_from_record(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_id: str | None,
    ) -> Dict[str, Any]:
        return {
            "source_id": source_id or f"{metadata['relative_path']}:{metadata['chunk_index']}",
            "content": content,
            "relative_path": metadata["relative_path"],
            "chunk_index": metadata["chunk_index"],
            "distance": None,
            "section_title": metadata.get("section_title") or "Body",
            "section_path": metadata.get("section_path") or metadata.get("section_title") or "Body",
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "block_type": metadata.get("block_type") or "body",
            "paper_title": metadata.get("paper_title") or metadata.get("relative_path"),
        }

    def _expand_context_blocks(
        self,
        kb_name: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        max_blocks = max(top_k, self.settings.max_context_blocks)
        primary_candidates = candidates[:top_k]
        expanded: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for candidate in primary_candidates:
            self._append_candidate(expanded, seen, candidate)
            neighbors = self.store.get_neighbor_chunks(
                kb_name,
                candidate["relative_path"],
                int(candidate["chunk_index"]),
                candidate.get("section_title") or "Body",
                self.settings.neighbor_expansion_window,
            )
            for neighbor in neighbors:
                neighbor_candidate = self._candidate_from_record(
                    content=neighbor.get("content") or "",
                    metadata=neighbor,
                    source_id=neighbor.get("chunk_id"),
                )
                neighbor_candidate["distance"] = candidate.get("distance")
                neighbor_candidate["retrieval_modes"] = sorted(
                    set(candidate.get("retrieval_modes", [])) | {"neighbor"}
                )
                self._append_candidate(expanded, seen, neighbor_candidate)
                if len(expanded) >= max_blocks:
                    return expanded
            if len(expanded) >= max_blocks:
                break
        return expanded

    @staticmethod
    def _append_candidate(
        target: List[Dict[str, Any]],
        seen: set[str],
        candidate: Dict[str, Any],
    ) -> None:
        key = str(candidate["source_id"])
        if key in seen:
            return
        seen.add(key)
        target.append(candidate)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w-]+", text.lower())


def _build_fts_query(text: str) -> str:
    terms = []
    for token in _tokenize(text):
        escaped = token.replace('"', '""').strip()
        if escaped:
            terms.append(f'"{escaped}"')
    return " OR ".join(terms)
