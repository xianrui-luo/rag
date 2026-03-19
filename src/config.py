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
    chroma_dir: Path
    sqlite_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    supported_extensions: tuple[str, ...]


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
        chroma_dir=chroma_dir,
        sqlite_path=sqlite_path,
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "5")),
        supported_extensions=supported_extensions,
    )
