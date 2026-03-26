from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from src.chunking import chunk_document
from src.config import Settings
from src.embeddings import LocalEmbeddingService
from src.loaders import load_document
from src.metadata_store import MetadataStore
from src.vectorstore import VectorStore


class FileScanInfo(TypedDict):
    abs_path: str
    mtime: float
    size: int
    file_hash: str


@dataclass
class IndexReport:
    kb_name: str
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: int = 0
    file_count: int = 0
    chunk_count: int = 0
    messages: List[str] = field(default_factory=list)


class IndexManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MetadataStore(settings.sqlite_path)
        self.vectorstore = VectorStore(settings.chroma_dir)
        self.embeddings = LocalEmbeddingService(settings.embedding_model)

    def refresh_index(self, kb_name: str, root_path: str) -> IndexReport:
        report = IndexReport(kb_name=kb_name)
        root = self._validate_root(root_path)
        self._validate_kb_config(kb_name, root)
        self.store.upsert_knowledge_base(
            kb_name,
            str(root),
            self.settings.embedding_model,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        existing_files = self.store.get_files(kb_name)
        current_files = self._scan_files(root, existing_files)

        for relative_path, info in current_files.items():
            existing = existing_files.get(relative_path)
            try:
                if existing is None:
                    self._index_file(kb_name, root, relative_path, info)
                    report.added += 1
                elif existing["file_hash"] != info["file_hash"]:
                    self._delete_file_records(kb_name, relative_path)
                    self._index_file(kb_name, root, relative_path, info)
                    report.updated += 1
                else:
                    report.unchanged += 1
            except Exception as exc:
                report.failed += 1
                report.messages.append(f"{relative_path}: {exc}")

        deleted_paths = sorted(set(existing_files) - set(current_files))
        for relative_path in deleted_paths:
            self._delete_file_records(kb_name, relative_path)
            report.deleted += 1

        stats = self.store.get_stats(kb_name)
        report.file_count = stats["file_count"]
        report.chunk_count = stats["chunk_count"]
        return report

    def rebuild_index(self, kb_name: str, root_path: str) -> IndexReport:
        root = self._validate_root(root_path)
        self.vectorstore.delete_by_kb(kb_name)
        self.store.delete_knowledge_base(kb_name)
        return self.refresh_index(kb_name, str(root))

    def get_stats(self, kb_name: str) -> Dict[str, int]:
        return self.store.get_stats(kb_name)

    def _validate_root(self, root_path: str) -> Path:
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("Folder path does not exist or is not a directory")
        return root

    def _validate_kb_config(self, kb_name: str, root: Path) -> None:
        existing = self.store.get_knowledge_base(kb_name)
        if not existing:
            return
        config_changed = any(
            [
                existing["embedding_model"] != self.settings.embedding_model,
                int(existing["chunk_size"]) != self.settings.chunk_size,
                int(existing["chunk_overlap"]) != self.settings.chunk_overlap,
            ]
        )
        if config_changed:
            raise ValueError(
                "Embedding or chunk settings changed for this knowledge base. Use Rebuild Index."
            )
        if Path(existing["root_path"]).resolve() != root:
            raise ValueError("Knowledge base name is already bound to a different folder")

    def _scan_files(
        self, root: Path, existing_files: Dict[str, Dict[str, Any]] | None = None
    ) -> Dict[str, FileScanInfo]:
        if existing_files is None:
            existing_files = {}
        discovered: Dict[str, FileScanInfo] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.settings.supported_extensions:
                continue
            stat = path.stat()
            relative_path = str(path.relative_to(root)).replace("\\", "/")
            mtime = stat.st_mtime
            size = stat.st_size

            existing = existing_files.get(relative_path)
            if existing and existing["mtime"] == mtime and existing["size"] == size:
                file_hash = existing["file_hash"]
            else:
                file_hash = self._hash_file(path)

            discovered[relative_path] = {
                "abs_path": str(path),
                "mtime": mtime,
                "size": size,
                "file_hash": file_hash,
            }
        return discovered

    def _index_file(self, kb_name: str, root: Path, relative_path: str, info: FileScanInfo) -> None:
        path = root / relative_path
        document = load_document(path)
        chunks = chunk_document(
            document,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.exclude_references,
        )
        if not chunks:
            raise ValueError("No readable text extracted")

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embeddings.embed_documents(chunk_texts)
        chunk_ids: List[str] = []
        metadatas = []
        chunk_rows = []

        file_hash = str(info["file_hash"])
        for index, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
            chunk_id = hashlib.sha256(
                f"{kb_name}:{relative_path}:{file_hash}:{index}".encode("utf-8")
            ).hexdigest()
            chunk_ids.append(chunk_id)
            metadatas.append(
                {
                    "kb_name": kb_name,
                    "relative_path": relative_path,
                    "abs_path": str(path),
                    "file_hash": file_hash,
                    "chunk_index": index,
                    "paper_title": chunk.paper_title,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "block_type": chunk.block_type,
                }
            )
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "content_hash": content_hash,
                    "content": chunk.text,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "block_type": chunk.block_type,
                    "paper_title": chunk.paper_title,
                }
            )

        self.vectorstore.upsert_chunks(chunk_ids, chunk_texts, embeddings, metadatas)
        self.store.upsert_file(
            kb_name,
            relative_path,
            str(path),
            file_hash,
            info["mtime"],
            info["size"],
        )
        self.store.replace_chunks(kb_name, relative_path, chunk_rows)

    def _delete_file_records(self, kb_name: str, relative_path: str) -> None:
        self.vectorstore.delete_by_file(kb_name, relative_path)
        self.store.delete_file(kb_name, relative_path)

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
