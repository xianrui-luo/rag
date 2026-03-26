from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetadataStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    name TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS files (
                    kb_name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    abs_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (kb_name, relative_path)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    kb_name TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    kb_name UNINDEXED,
                    relative_path UNINDEXED,
                    chunk_index UNINDEXED,
                    section_title,
                    section_path UNINDEXED,
                    page_start UNINDEXED,
                    page_end UNINDEXED,
                    block_type UNINDEXED,
                    paper_title,
                    content,
                    tokenize='unicode61'
                )
                """
            )
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
            }
            for column_name, column_type in [
                ("content", "TEXT"),
                ("section_title", "TEXT"),
                ("section_path", "TEXT"),
                ("page_start", "INTEGER"),
                ("page_end", "INTEGER"),
                ("block_type", "TEXT"),
                ("paper_title", "TEXT"),
            ]:
                if column_name not in existing_columns:
                    conn.execute(f"ALTER TABLE chunks ADD COLUMN {column_name} {column_type}")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute(
                """
                INSERT INTO chunks_fts (
                    chunk_id,
                    kb_name,
                    relative_path,
                    chunk_index,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_type,
                    paper_title,
                    content
                )
                SELECT
                    chunk_id,
                    kb_name,
                    relative_path,
                    chunk_index,
                    COALESCE(section_title, ''),
                    COALESCE(section_path, ''),
                    COALESCE(page_start, ''),
                    COALESCE(page_end, ''),
                    COALESCE(block_type, ''),
                    COALESCE(paper_title, ''),
                    COALESCE(content, '')
                FROM chunks
                """
            )

    def upsert_knowledge_base(
        self,
        name: str,
        root_path: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bases (name, root_path, embedding_model, chunk_size, chunk_overlap, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    root_path=excluded.root_path,
                    embedding_model=excluded.embedding_model,
                    chunk_size=excluded.chunk_size,
                    chunk_overlap=excluded.chunk_overlap,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (name, root_path, embedding_model, chunk_size, chunk_overlap),
            )

    def get_knowledge_base(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM knowledge_bases WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_files(self, kb_name: str) -> Dict[str, Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM files WHERE kb_name = ?", (kb_name,)).fetchall()
        return {row["relative_path"]: dict(row) for row in rows}

    def upsert_file(
        self,
        kb_name: str,
        relative_path: str,
        abs_path: str,
        file_hash: str,
        mtime: float,
        size: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (kb_name, relative_path, abs_path, file_hash, mtime, size, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(kb_name, relative_path) DO UPDATE SET
                    abs_path=excluded.abs_path,
                    file_hash=excluded.file_hash,
                    mtime=excluded.mtime,
                    size=excluded.size,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (kb_name, relative_path, abs_path, file_hash, mtime, size),
            )

    def replace_chunks(
        self,
        kb_name: str,
        relative_path: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE kb_name = ? AND relative_path = ?",
                (kb_name, relative_path),
            )
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    kb_name,
                    relative_path,
                    chunk_index,
                    content_hash,
                    content,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_type,
                    paper_title
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        kb_name,
                        relative_path,
                        chunk["chunk_index"],
                        chunk["content_hash"],
                        chunk["content"],
                        chunk["section_title"],
                        chunk["section_path"],
                        chunk["page_start"],
                        chunk["page_end"],
                        chunk["block_type"],
                        chunk["paper_title"],
                    )
                    for chunk in chunks
                ],
            )
            conn.execute(
                "DELETE FROM chunks_fts WHERE kb_name = ? AND relative_path = ?",
                (kb_name, relative_path),
            )
            conn.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id,
                    kb_name,
                    relative_path,
                    chunk_index,
                    section_title,
                    section_path,
                    page_start,
                    page_end,
                    block_type,
                    paper_title,
                    content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["chunk_id"],
                        kb_name,
                        relative_path,
                        chunk["chunk_index"],
                        chunk["section_title"],
                        chunk["section_path"],
                        chunk["page_start"],
                        chunk["page_end"],
                        chunk["block_type"],
                        chunk["paper_title"],
                        chunk["content"],
                    )
                    for chunk in chunks
                ],
            )

    def delete_file(self, kb_name: str, relative_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE kb_name = ? AND relative_path = ?",
                (kb_name, relative_path),
            )
            conn.execute(
                "DELETE FROM chunks_fts WHERE kb_name = ? AND relative_path = ?",
                (kb_name, relative_path),
            )
            conn.execute(
                "DELETE FROM files WHERE kb_name = ? AND relative_path = ?",
                (kb_name, relative_path),
            )

    def delete_knowledge_base(self, kb_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE kb_name = ?", (kb_name,))
            conn.execute("DELETE FROM chunks WHERE kb_name = ?", (kb_name,))
            conn.execute("DELETE FROM files WHERE kb_name = ?", (kb_name,))
            conn.execute("DELETE FROM knowledge_bases WHERE name = ?", (kb_name,))

    def get_stats(self, kb_name: str) -> Dict[str, int]:
        with self._connect() as conn:
            file_count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE kb_name = ?", (kb_name,)
            ).fetchone()[0]
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE kb_name = ?", (kb_name,)
            ).fetchone()[0]
        return {"file_count": file_count, "chunk_count": chunk_count}

    def get_chunks(self, kb_name: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE kb_name = ? ORDER BY relative_path, chunk_index",
                (kb_name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_chunks_fts(self, kb_name: str, match_query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        chunk_id,
                        kb_name,
                        relative_path,
                        chunk_index,
                        section_title,
                        section_path,
                        page_start,
                        page_end,
                        block_type,
                        paper_title,
                        content,
                        bm25(chunks_fts, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.2, 2.0) AS score
                    FROM chunks_fts
                    WHERE kb_name = ? AND chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (kb_name, match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(row) for row in rows]

    def get_neighbor_chunks(
        self,
        kb_name: str,
        relative_path: str,
        center_index: int,
        section_title: str,
        window: int,
    ) -> List[Dict[str, Any]]:
        if window <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE kb_name = ?
                  AND relative_path = ?
                  AND section_title = ?
                  AND chunk_index BETWEEN ? AND ?
                  AND chunk_index != ?
                ORDER BY chunk_index
                """,
                (
                    kb_name,
                    relative_path,
                    section_title,
                    center_index - window,
                    center_index + window,
                    center_index,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """Return all knowledge bases with their metadata."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, root_path, embedding_model, chunk_size, chunk_overlap, updated_at "
                "FROM knowledge_bases ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def kb_exists(self, kb_name: str) -> bool:
        """Check if a knowledge base exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE name = ?", (kb_name,)
            ).fetchone()
        return row is not None
