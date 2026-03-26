from __future__ import annotations

from typing import Any, List

from openai import OpenAI

from src.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def generate_answer(
        self,
        question: str,
        context_blocks: list[dict],
        history: List[dict[str, Any]] | None = None,
    ) -> str:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        context_text = "\n\n".join(
            f"[Source {i}] file={item['relative_path']} chunk={item['chunk_index']} "
            f"section={item.get('section_title', 'Body')} pages={item.get('page_start', '?')}-{item.get('page_end', '?')}\n"
            f"{item['content']}"
            for i, item in enumerate(context_blocks, start=1)
        )
        system_prompt = (
            "You are a precise RAG assistant. Answer only from the provided context. "
            "Use recent conversation only to resolve references like pronouns or follow-up questions. "
            "Do not use prior conversation as factual evidence. "
            "If the context is insufficient, say so clearly. End factual claims with source tags like [S1]."
        )
        messages: List[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._recent_history_messages(history))
        messages.append(
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nContext:\n{context_text}",
            }
        )
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.2,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def rewrite_query(self, question: str, history: List[dict[str, Any]] | None = None) -> str:
        if not self.settings.openai_api_key or not history:
            return question
        rewrite_prompt = (
            "Rewrite the latest user question into a standalone retrieval query for academic-paper search. "
            "Use the recent conversation only to resolve references like pronouns, omitted subjects, or follow-up wording. "
            "Do not answer the question. Do not add facts not present in the conversation. "
            "Return only the rewritten query."
        )
        messages: List[dict[str, str]] = [{"role": "system", "content": rewrite_prompt}]
        messages.extend(self._recent_history_messages(history))
        messages.append(
            {
                "role": "user",
                "content": f"Latest question to rewrite:\n{question}",
            }
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.0,
                messages=messages,
            )
        except Exception:
            return question
        rewritten = (response.choices[0].message.content or "").strip()
        if not rewritten:
            return question
        first_line = next((line.strip() for line in rewritten.splitlines() if line.strip()), rewritten)
        cleaned = first_line.removeprefix("Rewritten query:").strip().strip('"')
        return cleaned or question

    def _recent_history_messages(self, history: List[dict[str, Any]] | None) -> List[dict[str, str]]:
        if not history:
            return []
        recent_messages = history[-(self.settings.history_turns * 2) :]
        messages: List[dict[str, str]] = []
        for item in recent_messages:
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                continue
            messages.append({"role": role, "content": content})
        return messages
