from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _as_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    embedding_model: str
    reranker_model: str
    chroma_dir: Path
    sqlite_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_candidates: int
    lexical_candidates: int
    history_turns: int
    max_context_blocks: int
    neighbor_expansion_window: int
    enable_hybrid_retrieval: bool
    enable_reranker: bool
    enable_query_rewrite: bool
    exclude_references: bool
    supported_extensions: tuple[str, ...]


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    supported_extensions = tuple(
        ext.strip().lower()
        for ext in os.getenv("SUPPORTED_EXTENSIONS", ".pdf,.docx,.md,.txt").split(",")
        if ext.strip()
    )
    chroma_dir = _as_path(os.getenv("CHROMA_DIR", "./data/chroma"), project_root)
    sqlite_path = _as_path(os.getenv("SQLITE_PATH", "./data/sqlite/index.db"), project_root)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=project_root,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        chroma_dir=chroma_dir,
        sqlite_path=sqlite_path,
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "5")),
        retrieval_candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
        lexical_candidates=int(os.getenv("LEXICAL_CANDIDATES", "20")),
        history_turns=int(os.getenv("HISTORY_TURNS", "4")),
        max_context_blocks=int(os.getenv("MAX_CONTEXT_BLOCKS", "8")),
        neighbor_expansion_window=int(os.getenv("NEIGHBOR_EXPANSION_WINDOW", "1")),
        enable_hybrid_retrieval=_as_bool(os.getenv("ENABLE_HYBRID_RETRIEVAL", "true"), True),
        enable_reranker=_as_bool(os.getenv("ENABLE_RERANKER", "true"), True),
        enable_query_rewrite=_as_bool(os.getenv("ENABLE_QUERY_REWRITE", "true"), True),
        exclude_references=_as_bool(os.getenv("EXCLUDE_REFERENCES", "true"), True),
        supported_extensions=supported_extensions,
    )
