from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

from src.config import get_settings
from src.index_manager import IndexManager, IndexReport
from src.metadata_store import MetadataStore
from src.rag_service import RAGService


def _human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def build_app() -> gr.Blocks:
    settings = get_settings()
    index_manager = IndexManager(settings)
    rag_service = RAGService(settings)
    meta_store = MetadataStore(settings.sqlite_path)
    home_dir = Path.home().resolve()
    desktop_dir = home_dir / "Desktop"
    browser_shortcuts = {
        "Desktop / 桌面": desktop_dir if desktop_dir.exists() else home_dir,
        "D:/": Path("D:/"),
        "E:/": Path("E:/"),
    }
    explorer_root = next(
        (path.resolve() for path in [desktop_dir, Path("D:/"), Path("E:/"), home_dir] if path.exists()),
        home_dir,
    )

    def normalize_browser_root(target: str | Path) -> Path:
        candidate = Path(target).expanduser().resolve()
        if candidate.is_file():
            return candidate.parent
        if candidate.exists() and candidate.is_dir():
            return candidate
        return explorer_root

    def list_directory_entries(target: str | Path) -> List[Dict[str, Any]]:
        browser_root = normalize_browser_root(target)
        entries: List[Dict[str, Any]] = []
        for path in sorted(browser_root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            prefix = "[DIR]" if path.is_dir() else "[FILE]"
            size = "" if path.is_dir() else _human_size(path.stat().st_size)
            suffix = "" if not size else f" ({size})"
            entries.append(
                {
                    "label": f"{prefix} {path.name}{suffix}",
                    "path": str(path.resolve()),
                    "is_dir": path.is_dir(),
                }
            )
        return entries

    def browser_view(
        target: str | Path,
        message: str,
    ) -> Tuple[str, str, List[Dict[str, Any]], str, str, str, gr.Radio]:
        browser_root = normalize_browser_root(target)
        entries = list_directory_entries(browser_root)
        status_message = message.format(path=browser_root)
        if not entries:
            status_message = f"{status_message}\nFolder is empty / 当前文件夹为空"
        return (
            str(browser_root),
            str(browser_root),
            entries,
            "",
            "No entry selected / 未选择条目",
            status_message,
            gr.Radio(choices=[entry["label"] for entry in entries], value=None),
        )

    def reset_new_kb_panel(
        message: str = "Ready to choose a folder for the new knowledge base / 可以为新知识库选择文件夹",
        visible: bool = False,
    ):
        browser_root, browser_path, entries, selected_state, selected_display, status_message, selector_update = browser_view(
            explorer_root,
            message,
        )
        return (
            gr.Column(visible=visible),
            "",
            "",
            browser_root,
            browser_path,
            entries,
            selected_state,
            selected_display,
            status_message,
            selector_update,
        )

    def get_kb_choices() -> List[str]:
        return [kb["name"] for kb in meta_store.list_knowledge_bases()]

    def get_kb_record(kb_name: str | None) -> Dict[str, Any] | None:
        if not kb_name:
            return None
        return meta_store.get_knowledge_base(kb_name)

    def get_kb_path(kb_name: str | None) -> str:
        kb = get_kb_record(kb_name)
        return kb["root_path"] if kb else ""

    def get_kb_info(kb_name: str | None) -> str:
        kb = get_kb_record(kb_name)
        if not kb:
            return "No knowledge base selected / 未选择知识库"
        stats = meta_store.get_stats(kb["name"])
        return (
            f"Path / 路径: {kb['root_path']}\n"
            f"Model / 模型: {kb['embedding_model']}\n"
            f"Chunk / 分块: {kb['chunk_size']} / {kb['chunk_overlap']}\n"
            f"Files / 文件: {stats['file_count']} | Chunks / 分块数: {stats['chunk_count']}\n"
            f"Updated / 更新时间: {kb['updated_at']}"
        )

    def format_report(report: IndexReport, rebuilt: bool = False) -> Tuple[str, str]:
        summary_lines = [f"KB / 知识库: {report.kb_name}"]
        if rebuilt:
            summary_lines.append("Rebuild completed / 重建完成")
        summary_lines.extend(
            [
                f"Added / 新增: {report.added}",
                f"Updated / 更新: {report.updated}",
                f"Deleted / 删除: {report.deleted}",
                f"Unchanged / 未变化: {report.unchanged}",
                f"Failed / 失败: {report.failed}",
                f"Files / 文件: {report.file_count}",
                f"Chunks / 分块: {report.chunk_count}",
            ]
        )
        details = "\n".join(report.messages or []) or (
            "Rebuild completed / 重建完成" if rebuilt else "Refresh completed / 刷新完成"
        )
        return "\n".join(summary_lines), details

    def refresh_index(kb_name: str | None) -> Tuple[str, str, str, str]:
        kb = get_kb_record(kb_name)
        if not kb:
            return (
                "Error / 错误",
                "Please select an existing knowledge base first / 请先选择已有知识库",
                "",
                "No knowledge base selected / 未选择知识库",
            )
        report = index_manager.refresh_index(kb["name"], kb["root_path"])
        summary, details = format_report(report)
        return summary, details, kb["root_path"], get_kb_info(kb["name"])

    def rebuild_index(kb_name: str | None) -> Tuple[str, str, str, str]:
        kb = get_kb_record(kb_name)
        if not kb:
            return (
                "Error / 错误",
                "Please select an existing knowledge base first / 请先选择已有知识库",
                "",
                "No knowledge base selected / 未选择知识库",
            )
        report = index_manager.rebuild_index(kb["name"], kb["root_path"])
        summary, details = format_report(report, rebuilt=True)
        return summary, details, kb["root_path"], get_kb_info(kb["name"])

    def clear_chat() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str, List[List[Any]], str]:
        return [], [], "", [], ""

    def ask_question(
        kb_name: str | None,
        question: str,
        top_k: int,
        conversation_history: List[Dict[str, str]] | None,
    ):
        history = list(conversation_history or [])
        original_question = question.strip()
        answer, sources, retrieval_question = rag_service.ask(
            (kb_name or "").strip(),
            original_question,
            top_k,
            history,
        )
        rows = [
            [
                item["relative_path"],
                item.get("section_title", "Body"),
                f"{item.get('page_start', '?')}-{item.get('page_end', '?')}",
                item["chunk_index"],
                item.get("rerank_score") or item.get("fusion_score") or item.get("distance"),
                item["content"],
            ]
            for item in sources
        ]
        if retrieval_question.strip() and retrieval_question.strip() != original_question:
            user_message = (
                f"Question / 原始问题:\n{original_question}\n\n"
                f"> Rewritten Query / 改写检索问题:\n"
                f"> *{retrieval_question.strip()}*"
            )
        else:
            user_message = f"Question / 原始问题:\n{original_question}"
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, updated_history, answer, rows, ""

    def on_kb_select(kb_name: str | None) -> Tuple[str, str, List[Dict[str, str]], List[Dict[str, str]], str, List[List[Any]], str]:
        return get_kb_path(kb_name), get_kb_info(kb_name), [], [], "", [], ""

    def show_new_kb_panel() -> gr.Column:
        return gr.Column(visible=True)

    def hide_new_kb_panel():
        return reset_new_kb_panel(visible=False)

    def create_kb(
        new_kb_name: str,
        new_kb_folder_path: str,
    ):
        kb_name = new_kb_name.strip()
        folder_path = new_kb_folder_path.strip()
        if not kb_name:
            return (
                "Error / 错误",
                "Knowledge base name is required / 请输入知识库名称",
                gr.Dropdown(choices=get_kb_choices()),
                "",
                get_kb_info(None),
                gr.Column(visible=True),
                new_kb_name,
                new_kb_folder_path,
                *browser_view(explorer_root, "Please choose a folder for the new knowledge base / 请为新知识库选择文件夹: {path}"),
                [],
                [],
                "",
                [],
                "",
            )
        if not folder_path:
            return (
                "Error / 错误",
                "Folder path is required / 请输入或选择文件夹路径",
                gr.Dropdown(choices=get_kb_choices()),
                "",
                get_kb_info(None),
                gr.Column(visible=True),
                new_kb_name,
                new_kb_folder_path,
                *browser_view(explorer_root, "Please choose a folder for the new knowledge base / 请为新知识库选择文件夹: {path}"),
                [],
                [],
                "",
                [],
                "",
            )

        try:
            report = index_manager.refresh_index(kb_name, folder_path)
        except Exception as exc:
            return (
                "Error / 错误",
                str(exc),
                gr.Dropdown(choices=get_kb_choices()),
                "",
                get_kb_info(None),
                gr.Column(visible=True),
                new_kb_name,
                new_kb_folder_path,
                *browser_view(explorer_root, "Please choose a folder for the new knowledge base / 请为新知识库选择文件夹: {path}"),
                [],
                [],
                "",
                [],
                "",
            )
        summary, details = format_report(report)
        kb_choices = get_kb_choices()
        panel_update = reset_new_kb_panel(
            message="New knowledge base created / 新知识库已创建: {path}",
            visible=False,
        )
        return (
            summary,
            details,
            gr.Dropdown(choices=kb_choices, value=kb_name),
            get_kb_path(kb_name),
            get_kb_info(kb_name),
            *panel_update,
            [],
            [],
            "",
            [],
            "",
        )

    def delete_kb(
        kb_name: str | None,
    ) -> Tuple[str, str, gr.Dropdown, str, str, List[Dict[str, str]], List[Dict[str, str]], str, List[List[Any]], str]:
        if not kb_name or not meta_store.kb_exists(kb_name):
            return (
                "Error / 错误",
                "Please select an existing knowledge base first / 请先选择已有知识库",
                gr.Dropdown(choices=get_kb_choices()),
                get_kb_path(None),
                get_kb_info(None),
                [],
                [],
                "",
                [],
                "",
            )
        index_manager.vectorstore.delete_by_kb(kb_name)
        meta_store.delete_knowledge_base(kb_name)
        kb_choices = get_kb_choices()
        next_kb = kb_choices[0] if kb_choices else None
        return (
            "Deleted / 已删除",
            f"Knowledge base '{kb_name}' deleted / 知识库 '{kb_name}' 已删除",
            gr.Dropdown(choices=kb_choices, value=next_kb),
            get_kb_path(next_kb),
            get_kb_info(next_kb),
            [],
            [],
            "",
            [],
            "",
        )

    def on_entry_select(selected_label: str | None, entries: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        if not selected_label:
            return "", "No entry selected / 未选择条目", "No entry selected / 未选择条目"
        entry = next((item for item in entries if item["label"] == selected_label), None)
        if entry is None:
            return "", "Entry not found / 未找到条目", "Entry not found / 未找到条目"
        selected_path = entry["path"]
        entry_kind = "Folder / 文件夹" if entry["is_dir"] else "File / 文件"
        return selected_path, selected_path, f"Selected {entry_kind}: {selected_path}"

    def enter_selected(
        selected_entry_path: str,
        current_browser_root: str,
    ):
        if not selected_entry_path:
            return browser_view(
                current_browser_root,
                "Please select a folder or file first / 请先选择一个文件夹或文件: {path}",
            )
        selected_path = Path(selected_entry_path)
        if not selected_path.exists():
            return browser_view(
                current_browser_root,
                "Selected entry no longer exists / 选中的条目已不存在: {path}",
            )
        if not selected_path.is_dir():
            return browser_view(
                current_browser_root,
                "Selected entry is a file and cannot be entered / 当前选择是文件，无法进入: {path}",
            )
        return browser_view(selected_path, "Entered folder / 已进入文件夹: {path}")

    def go_up(current_browser_root: str):
        current_root = normalize_browser_root(current_browser_root)
        parent = current_root.parent if current_root.parent != current_root else current_root
        return browser_view(parent, "Moved to parent folder / 已切换到上一级目录: {path}")

    def jump_to_shortcut(target: str):
        return browser_view(target, "Jumped to / 已跳转到: {path}")

    def use_current_folder(current_browser_root: str) -> Tuple[str, str]:
        current_root = normalize_browser_root(current_browser_root)
        return str(current_root), f"Using current folder for new knowledge base / 已将当前文件夹用于新知识库: {current_root}"

    initial_entries = list_directory_entries(explorer_root)
    initial_kb_choices = get_kb_choices()
    initial_kb = initial_kb_choices[0] if initial_kb_choices else None

    with gr.Blocks(title="Folder RAG Local") as demo:
        gr.Markdown("# Folder RAG Local / 本地文件夹 RAG")
        gr.Markdown(
            "Local folder RAG with multi-KB management, incremental refresh, and cloud generation. / 支持多知识库管理、增量刷新和云端生成的本地文件夹 RAG。"
        )
        gr.Markdown(
            "Academic paper mode uses section-aware chunking, reference suppression, hybrid retrieval, and reranking. / 学术论文模式默认启用章节感知分块、参考文献抑制、混合检索和重排。"
        )

        with gr.Row():
            kb_dropdown = gr.Dropdown(
                label="Knowledge Base / 知识库",
                choices=initial_kb_choices,
                value=initial_kb,
                allow_custom_value=False,
                interactive=True,
                scale=2,
            )
            current_kb_path = gr.Textbox(
                label="Current KB Folder / 当前知识库目录",
                value=get_kb_path(initial_kb),
                interactive=False,
                scale=3,
            )

        with gr.Row():
            new_kb_button = gr.Button("New KB / 新增知识库")
            refresh_button = gr.Button("Refresh Index / 增量刷新", variant="primary")
            rebuild_button = gr.Button("Rebuild Index / 重建索引")
            delete_button = gr.Button("Delete KB / 删除知识库", variant="stop")

        browser_root_state = gr.State(str(explorer_root))
        browser_entries_state = gr.State(initial_entries)
        selected_entry_state = gr.State("")
        conversation_state = gr.State([])

        with gr.Column(visible=False) as new_kb_panel:
            gr.Markdown(
                "### New Knowledge Base / 新增知识库\nSelect a folder only when creating a new KB. / 只有在创建新知识库时才需要选择文件夹。"
            )
            with gr.Row():
                new_kb_name = gr.Textbox(label="New KB Name / 新知识库名称")
                new_kb_folder_path = gr.Textbox(label="New KB Folder / 新知识库目录")
            browser_path_display = gr.Textbox(
                label="Current Browser Folder / 当前浏览目录",
                value=str(explorer_root),
                interactive=False,
            )
            with gr.Row():
                up_button = gr.Button("Up One Level / 上一级")
                desktop_button = gr.Button("Desktop / 桌面")
                d_drive_button = gr.Button("D:/")
                e_drive_button = gr.Button("E:/")
            browser_entry_selector = gr.Radio(
                label="Folder Contents / 当前目录内容",
                choices=[entry["label"] for entry in initial_entries],
                value=None,
            )
            with gr.Row():
                enter_selected_button = gr.Button("Enter Selected / 进入选中项")
                use_current_folder_button = gr.Button(
                    "Use Current Folder / 使用当前目录",
                    variant="primary",
                )
                create_kb_button = gr.Button("Create KB / 创建知识库", variant="primary")
                cancel_new_kb_button = gr.Button("Cancel / 取消")
            selected_entry_display = gr.Textbox(
                label="Selected Entry / 当前选中项",
                value="No entry selected / 未选择条目",
                interactive=False,
            )
            browser_status = gr.Textbox(
                label="Browser Status / 浏览器状态",
                value=f"Ready / 就绪: {explorer_root}",
                lines=2,
                interactive=False,
            )

        kb_info = gr.Textbox(
            label="KB Info / 知识库信息",
            value=get_kb_info(initial_kb),
            lines=5,
            interactive=False,
        )

        with gr.Row():
            index_summary = gr.Textbox(label="Index Summary / 索引摘要", lines=8)
            index_details = gr.Textbox(label="Index Details / 索引详情", lines=8)

        gr.Markdown("## Ask / 提问")
        gr.Markdown(
            "Switching the knowledge base clears the current chat automatically. / 切换知识库会自动清空当前对话。"
        )
        chat_history = gr.Chatbot(
            label="Conversation / 对话记录",
            value=[],
            height=320,
            placeholder="Ask a question to start the conversation. / 提问后将在这里显示当前页面会话记录。",
        )
        with gr.Row():
            question = gr.Textbox(label="Question / 问题", lines=4, scale=4)
            top_k = gr.Slider(label="Top K / 召回数量", minimum=1, maximum=10, step=1, value=settings.top_k)

        with gr.Row():
            ask_button = gr.Button("Ask / 提问", variant="primary")
            clear_chat_button = gr.Button("Clear Chat / 清空对话")
        answer = gr.Textbox(label="Answer / 回答", lines=10)
        sources = gr.Dataframe(
            headers=[
                "file / 文件",
                "section / 章节",
                "pages / 页码",
                "chunk / 分块",
                "score / 分数",
                "content / 内容",
            ],
            datatype=["str", "str", "str", "number", "number", "str"],
            label="Retrieved Sources / 检索到的来源",
            wrap=True,
        )

        kb_dropdown.change(
            on_kb_select,
            inputs=[kb_dropdown],
            outputs=[current_kb_path, kb_info, conversation_state, chat_history, answer, sources, question],
        )
        new_kb_button.click(show_new_kb_panel, outputs=[new_kb_panel])
        cancel_new_kb_button.click(
            hide_new_kb_panel,
            outputs=[
                new_kb_panel,
                new_kb_name,
                new_kb_folder_path,
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        browser_entry_selector.change(
            on_entry_select,
            inputs=[browser_entry_selector, browser_entries_state],
            outputs=[selected_entry_state, selected_entry_display, browser_status],
        )
        enter_selected_button.click(
            enter_selected,
            inputs=[selected_entry_state, browser_root_state],
            outputs=[
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        up_button.click(
            go_up,
            inputs=[browser_root_state],
            outputs=[
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        desktop_button.click(
            lambda: jump_to_shortcut(str(browser_shortcuts["Desktop / 桌面"])),
            outputs=[
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        d_drive_button.click(
            lambda: jump_to_shortcut(str(browser_shortcuts["D:/"])),
            outputs=[
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        e_drive_button.click(
            lambda: jump_to_shortcut(str(browser_shortcuts["E:/"])),
            outputs=[
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
            ],
        )
        use_current_folder_button.click(
            use_current_folder,
            inputs=[browser_root_state],
            outputs=[new_kb_folder_path, browser_status],
        )
        create_kb_button.click(
            create_kb,
            inputs=[new_kb_name, new_kb_folder_path],
            outputs=[
                index_summary,
                index_details,
                kb_dropdown,
                current_kb_path,
                kb_info,
                new_kb_panel,
                new_kb_name,
                new_kb_folder_path,
                browser_root_state,
                browser_path_display,
                browser_entries_state,
                selected_entry_state,
                selected_entry_display,
                browser_status,
                browser_entry_selector,
                conversation_state,
                chat_history,
                answer,
                sources,
                question,
            ],
        )
        refresh_button.click(
            refresh_index,
            inputs=[kb_dropdown],
            outputs=[index_summary, index_details, current_kb_path, kb_info],
        )
        rebuild_button.click(
            rebuild_index,
            inputs=[kb_dropdown],
            outputs=[index_summary, index_details, current_kb_path, kb_info],
        )
        delete_button.click(
            delete_kb,
            inputs=[kb_dropdown],
            outputs=[index_summary, index_details, kb_dropdown, current_kb_path, kb_info, conversation_state, chat_history, answer, sources, question],
        )
        ask_button.click(
            ask_question,
            inputs=[kb_dropdown, question, top_k, conversation_state],
            outputs=[conversation_state, chat_history, answer, sources, question],
        )
        clear_chat_button.click(
            clear_chat,
            outputs=[conversation_state, chat_history, answer, sources, question],
        )

    return demo
