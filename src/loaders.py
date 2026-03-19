from __future__ import annotations

from pathlib import Path

import docx2txt
from pypdf import PdfReader


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        return docx2txt.process(str(path)) or ""
    raise ValueError(f"Unsupported file type: {path.suffix}")
