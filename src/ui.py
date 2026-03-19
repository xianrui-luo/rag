from __future__ import annotations

from typing import List, Tuple

import gradio as gr

from src.config import get_settings
from src.index_manager import IndexManager
from src.metadata_store import MetadataStore
from src.rag_service import RAGService


def build_app() -> gr.Blocks:
    settings = get_settings()
    index_manager = IndexManager(settings)
    rag_service = RAGService(settings)
    meta_store = MetadataStore(settings.sqlite_path)

    def get_kb_choices() -> List[str]:
        kbs = meta_store.list_knowledge_bases()
        return [kb["name"] for kb in kbs] or ["default"]

    def get_kb_info(kb_name: str) -> str:
        if not kb_name:
            return ""
        kb = meta_store.get_knowledge_base(kb_name)
        if not kb:
            return "Knowledge base not found / 未找到知识库"
        stats = meta_store.get_stats(kb_name)
        return (
            f"Path / 路径: {kb['root_path']}\n"
            f"Model / 模型: {kb['embedding_model']}\n"
            f"Chunk / 分块: {kb['chunk_size']} / {kb['chunk_overlap']}\n"
            f"Files / 文件: {stats['file_count']} | Chunks / 分块数: {stats['chunk_count']}\n"
            f"Updated / 更新时间: {kb['updated_at']}"
        )

    def refresh_index(kb_name: str, folder_path: str) -> Tuple[str, str, gr.Dropdown, str]:
        report = index_manager.refresh_index(kb_name.strip(), folder_path.strip())
        summary = (
            f"KB / 知识库: {report.kb_name}\n"
            f"Added / 新增: {report.added}\n"
            f"Updated / 更新: {report.updated}\n"
            f"Deleted / 删除: {report.deleted}\n"
            f"Unchanged / 未变化: {report.unchanged}\n"
            f"Failed / 失败: {report.failed}\n"
            f"Files / 文件: {report.file_count}\n"
            f"Chunks / 分块: {report.chunk_count}"
        )
        details = "\n".join(report.messages or []) or "Refresh completed / 刷新完成"
        return summary, details, gr.Dropdown(choices=get_kb_choices(), value=kb_name), get_kb_info(kb_name)

    def rebuild_index(kb_name: str, folder_path: str) -> Tuple[str, str, gr.Dropdown, str]:
        report = index_manager.rebuild_index(kb_name.strip(), folder_path.strip())
        summary = (
            f"KB / 知识库: {report.kb_name}\n"
            f"Rebuild completed / 重建完成\n"
            f"Added / 新增: {report.added}\n"
            f"Updated / 更新: {report.updated}\n"
            f"Deleted / 删除: {report.deleted}\n"
            f"Failed / 失败: {report.failed}\n"
            f"Files / 文件: {report.file_count}\n"
            f"Chunks / 分块: {report.chunk_count}"
        )
        details = "\n".join(report.messages or []) or "Rebuild completed / 重建完成"
        return summary, details, gr.Dropdown(choices=get_kb_choices(), value=kb_name), get_kb_info(kb_name)

    def delete_kb(kb_name: str) -> Tuple[str, str, gr.Dropdown, str]:
        if not kb_name or not meta_store.kb_exists(kb_name):
            return (
                "Error / 错误",
                f"Knowledge base '{kb_name}' not found / 未找到知识库 '{kb_name}'",
                gr.Dropdown(choices=get_kb_choices()),
                "",
            )
        index_manager.vectorstore.delete_by_kb(kb_name)
        meta_store.delete_knowledge_base(kb_name)
        choices = get_kb_choices()
        return (
            "Deleted / 已删除",
            f"Knowledge base '{kb_name}' deleted / 知识库 '{kb_name}' 已删除",
            gr.Dropdown(choices=choices, value=choices[0] if choices else ""),
            "",
        )

    def ask_question(kb_name: str, question: str, top_k: int):
        answer, sources = rag_service.ask(kb_name.strip(), question.strip(), top_k)
        rows = [
            [item["relative_path"], item["chunk_index"], item["distance"], item["content"]]
            for item in sources
        ]
        return answer, rows

    def on_kb_select(kb_name: str) -> Tuple[str, str]:
        if not kb_name:
            return "", ""
        kb = meta_store.get_knowledge_base(kb_name)
        folder = kb["root_path"] if kb else ""
        return folder, get_kb_info(kb_name)

    with gr.Blocks(title="Folder RAG Local") as demo:
        gr.Markdown("# Folder RAG Local / 本地文件夹 RAG")
        gr.Markdown(
            "Local folder RAG with multi-KB management, incremental refresh, and cloud generation. / 支持多知识库管理、增量刷新和云端生成的本地文件夹 RAG。"
        )

        with gr.Row():
            with gr.Column(scale=2):
                kb_dropdown = gr.Dropdown(
                    label="Knowledge Base / 知识库",
                    choices=get_kb_choices(),
                    value=get_kb_choices()[0],
                    allow_custom_value=True,
                    interactive=True,
                )
            with gr.Column(scale=3):
                folder_path = gr.Textbox(
                    label="Folder Path / 文件夹路径",
                    placeholder="/path/to/your/documents",
                )

        kb_info = gr.Textbox(label="KB Info / 知识库信息", lines=5, interactive=False)

        with gr.Row():
            refresh_button = gr.Button("Refresh Index / 增量刷新", variant="primary")
            rebuild_button = gr.Button("Rebuild Index / 重建索引")
            delete_button = gr.Button("Delete KB / 删除知识库", variant="stop")

        with gr.Row():
            index_summary = gr.Textbox(label="Index Summary / 索引摘要", lines=8)
            index_details = gr.Textbox(label="Index Details / 索引详情", lines=8)

        gr.Markdown("## Ask / 提问")
        gr.Markdown(
            "Refresh the index before asking if you added or changed files. / 如果你刚新增或修改了文件，请先刷新索引再提问。"
        )
        with gr.Row():
            question = gr.Textbox(label="Question / 问题", lines=4, scale=4)
            top_k = gr.Slider(label="Top K / 召回数量", minimum=1, maximum=10, step=1, value=settings.top_k)

        ask_button = gr.Button("Ask / 提问", variant="primary")
        answer = gr.Textbox(label="Answer / 回答", lines=10)
        sources = gr.Dataframe(
            headers=["file / 文件", "chunk / 分块", "distance / 距离", "content / 内容"],
            datatype=["str", "number", "number", "str"],
            label="Retrieved Sources / 检索到的来源",
            wrap=True,
        )

        kb_dropdown.change(on_kb_select, inputs=[kb_dropdown], outputs=[folder_path, kb_info])
        refresh_button.click(
            refresh_index,
            inputs=[kb_dropdown, folder_path],
            outputs=[index_summary, index_details, kb_dropdown, kb_info],
        )
        rebuild_button.click(
            rebuild_index,
            inputs=[kb_dropdown, folder_path],
            outputs=[index_summary, index_details, kb_dropdown, kb_info],
        )
        delete_button.click(
            delete_kb,
            inputs=[kb_dropdown],
            outputs=[index_summary, index_details, kb_dropdown, kb_info],
        )
        ask_button.click(ask_question, inputs=[kb_dropdown, question, top_k], outputs=[answer, sources])

    return demo
