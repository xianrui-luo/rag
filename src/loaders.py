from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import docx2txt
from pypdf import PdfReader


REFERENCE_HEADINGS = {
    "reference",
    "references",
    "bibliography",
}

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "approaches",
    "experiment",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "appendix",
    "acknowledgements",
    "acknowledgments",
}


@dataclass(frozen=True)
class DocumentBlock:
    text: str
    page_start: int
    page_end: int
    section_title: str
    section_level: int
    block_type: str


@dataclass(frozen=True)
class LoadedDocument:
    title: str
    blocks: List[DocumentBlock]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


def load_text(path: Path) -> str:
    return load_document(path).text


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return _load_text_document(text, fallback_title=path.stem)
    if suffix == ".pdf":
        return _load_pdf_document(path)
    if suffix == ".docx":
        text = docx2txt.process(str(path)) or ""
        return _load_text_document(text, fallback_title=path.stem)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_pdf_document(path: Path) -> LoadedDocument:
    reader = PdfReader(str(path))
    pages = [_clean_pdf_text(page.extract_text() or "") for page in reader.pages]
    page_blocks: List[tuple[int, str]] = []
    for page_number, page_text in enumerate(pages, start=1):
        for paragraph in _extract_pdf_paragraphs(page_text):
            page_blocks.append((page_number, paragraph))
    if not page_blocks:
        return LoadedDocument(title=path.stem, blocks=[])

    title = _detect_title([paragraph for _, paragraph in page_blocks], path.stem)
    blocks = _assign_sections(page_blocks, title)
    return LoadedDocument(title=title, blocks=blocks)


def _load_text_document(text: str, fallback_title: str) -> LoadedDocument:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    page_blocks: List[tuple[int, str]] = []
    current = ""
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                page_blocks.append((1, _normalize_whitespace(current)))
                current = ""
            continue
        if _looks_like_heading(line):
            if current:
                page_blocks.append((1, _normalize_whitespace(current)))
                current = ""
            page_blocks.append((1, line))
            continue
        current = f"{current} {line}".strip() if current else line
    if current:
        page_blocks.append((1, _normalize_whitespace(current)))
    title = _detect_title([paragraph for _, paragraph in page_blocks], fallback_title)
    return LoadedDocument(title=title, blocks=_assign_sections(page_blocks, title))


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines: List[str] = []
    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d+", line):
            cleaned_lines.append("")
            continue
        if _looks_like_running_header(line):
            cleaned_lines.append("")
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _extract_pdf_paragraphs(page_text: str) -> List[str]:
    paragraphs: List[str] = []
    current = ""
    for raw_line in page_text.split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            continue
        if _looks_like_heading(line):
            if current:
                paragraphs.append(current.strip())
                current = ""
            paragraphs.append(line)
            continue
        if not current:
            current = line
            continue
        if current.endswith("-"):
            current = current[:-1] + line
        elif _should_join_pdf_lines(current, line):
            current = f"{current} {line}"
        else:
            paragraphs.append(current.strip())
            current = line
    if current:
        paragraphs.append(current.strip())
    return [_normalize_whitespace(paragraph) for paragraph in paragraphs if paragraph.strip()]


def _assign_sections(page_blocks: List[tuple[int, str]], title: str) -> List[DocumentBlock]:
    blocks: List[DocumentBlock] = []
    current_section = "Front Matter"
    current_level = 0
    reference_mode = False
    for page_number, paragraph in page_blocks:
        heading_info = _parse_heading(paragraph)
        if heading_info is not None:
            current_section, current_level = heading_info
            reference_mode = _normalize_heading_name(current_section) in REFERENCE_HEADINGS
            continue
        block_type = "reference" if reference_mode else _section_to_block_type(current_section)
        blocks.append(
            DocumentBlock(
                text=paragraph,
                page_start=page_number,
                page_end=page_number,
                section_title=current_section,
                section_level=current_level,
                block_type=block_type,
            )
        )
    if not blocks and page_blocks:
        first_page, paragraph = page_blocks[0]
        blocks.append(
            DocumentBlock(
                text=paragraph,
                page_start=first_page,
                page_end=first_page,
                section_title="Body",
                section_level=1,
                block_type="body",
            )
        )
    return blocks


def _detect_title(paragraphs: List[str], fallback_title: str) -> str:
    for paragraph in paragraphs[:8]:
        line = _normalize_whitespace(paragraph)
        if not line:
            continue
        if _parse_heading(line) is not None:
            continue
        if len(line) > 220:
            continue
        return line
    return fallback_title


def _looks_like_running_header(line: str) -> bool:
    if len(line) > 120:
        return False
    if "arxiv" in line.lower():
        return True
    if re.fullmatch(r"[A-Z][A-Za-z\s,&-]{2,80}\d{4}", line):
        return True
    return False


def _should_join_pdf_lines(current: str, line: str) -> bool:
    if current.endswith(":"):
        return True
    if re.search(r"[.!?]$", current):
        return False
    if line[:1].islower():
        return True
    if len(line) < 40 and line.isupper():
        return False
    return len(current) < 180


def _looks_like_heading(line: str) -> bool:
    return _parse_heading(line) is not None


def _parse_heading(line: str) -> tuple[str, int] | None:
    stripped = _normalize_whitespace(line)
    if not stripped or len(stripped) > 160:
        return None
    if stripped.startswith("#"):
        level = len(stripped) - len(stripped.lstrip("#"))
        return stripped.lstrip("# "), level
    normalized = _normalize_heading_name(stripped)
    if normalized in KNOWN_HEADINGS or normalized in REFERENCE_HEADINGS:
        return stripped, 1
    numbered = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+)$", stripped)
    if numbered:
        return stripped, numbered.group(1).count(".") + 1
    roman = re.match(r"^(?:[IVXLC]+)[.)]?\s+(.+)$", stripped)
    if roman and len(stripped.split()) <= 10:
        return stripped, 1
    if stripped.isupper() and len(stripped.split()) <= 8:
        return stripped.title(), 1
    return None


def _normalize_heading_name(line: str) -> str:
    lowered = line.lower().strip().rstrip(":")
    lowered = re.sub(r"^(\d+(?:\.\d+)*)[.)]?\s+", "", lowered)
    lowered = re.sub(r"^(?:[ivxlc]+)[.)]?\s+", "", lowered)
    return lowered


def _section_to_block_type(section_title: str) -> str:
    normalized = _normalize_heading_name(section_title)
    if normalized == "abstract":
        return "abstract"
    if normalized in REFERENCE_HEADINGS:
        return "reference"
    if normalized in {"conclusion", "conclusions"}:
        return "conclusion"
    if normalized in {"appendix", "acknowledgements", "acknowledgments", "limitations"}:
        return normalized
    return "body"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
