from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.loaders import DocumentBlock, LoadedDocument


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    section_title: str
    section_path: str
    page_start: int
    page_end: int
    block_type: str
    paper_title: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    document = LoadedDocument(
        title="Untitled",
        blocks=[
            DocumentBlock(
                text=text,
                page_start=1,
                page_end=1,
                section_title="Body",
                section_level=1,
                block_type="body",
            )
        ],
    )
    return [chunk.text for chunk in chunk_document(document, chunk_size, chunk_overlap, False)]


def chunk_document(
    document: LoadedDocument,
    chunk_size: int,
    chunk_overlap: int,
    exclude_references: bool,
) -> List[ChunkRecord]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = [
        block
        for block in document.blocks
        if block.text.strip() and not (exclude_references and block.block_type == "reference")
    ]
    if not blocks:
        return []

    chunks: List[ChunkRecord] = []
    current_group: List[DocumentBlock] = []
    current_length = 0

    for block in blocks:
        block_length = len(block.text)
        if block_length > chunk_size:
            if current_group:
                chunks.append(_build_chunk(current_group, document.title))
                current_group = []
                current_length = 0
            chunks.extend(_split_long_block(block, document.title, chunk_size, chunk_overlap))
            continue

        if not current_group:
            current_group = [block]
            current_length = block_length
            continue

        same_section = block.section_title == current_group[-1].section_title
        projected_length = current_length + 2 + block_length
        if same_section and projected_length <= chunk_size:
            current_group.append(block)
            current_length = projected_length
            continue

        chunks.append(_build_chunk(current_group, document.title))
        current_group = _tail_overlap_blocks(current_group, chunk_overlap)
        current_length = _group_length(current_group)

        if not current_group or block.section_title != current_group[-1].section_title:
            current_group = []
            current_length = 0

        current_group.append(block)
        current_length = _group_length(current_group)

    if current_group:
        chunks.append(_build_chunk(current_group, document.title))
    return chunks


def _split_long_block(
    block: DocumentBlock,
    paper_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    text = block.text.strip()
    if not text:
        return []
    step = chunk_size - chunk_overlap
    chunks: List[ChunkRecord] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text_value = text[start:end].strip()
        if chunk_text_value:
            chunks.append(
                ChunkRecord(
                    text=chunk_text_value,
                    section_title=block.section_title,
                    section_path=block.section_title,
                    page_start=block.page_start,
                    page_end=block.page_end,
                    block_type=block.block_type,
                    paper_title=paper_title,
                )
            )
        if end >= len(text):
            break
        start += step
    return chunks


def _build_chunk(blocks: Iterable[DocumentBlock], paper_title: str) -> ChunkRecord:
    block_list = list(blocks)
    first = block_list[0]
    return ChunkRecord(
        text="\n\n".join(block.text for block in block_list),
        section_title=first.section_title,
        section_path=first.section_title,
        page_start=min(block.page_start for block in block_list),
        page_end=max(block.page_end for block in block_list),
        block_type=first.block_type,
        paper_title=paper_title,
    )


def _tail_overlap_blocks(blocks: List[DocumentBlock], chunk_overlap: int) -> List[DocumentBlock]:
    if chunk_overlap <= 0:
        return []
    overlap_blocks: List[DocumentBlock] = []
    overlap_length = 0
    for block in reversed(blocks):
        overlap_blocks.append(block)
        overlap_length += len(block.text) + 2
        if overlap_length >= chunk_overlap:
            break
    return list(reversed(overlap_blocks))


def _group_length(blocks: List[DocumentBlock]) -> int:
    if not blocks:
        return 0
    return sum(len(block.text) for block in blocks) + max(0, len(blocks) - 1) * 2
