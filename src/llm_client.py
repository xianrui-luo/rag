from __future__ import annotations

from openai import OpenAI

from src.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def generate_answer(self, question: str, context_blocks: list[dict]) -> str:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        context_text = "\n\n".join(
            f"[Source {i}] file={item['relative_path']} chunk={item['chunk_index']}\n{item['content']}"
            for i, item in enumerate(context_blocks, start=1)
        )
        system_prompt = (
            "You are a precise RAG assistant. Answer only from the provided context. "
            "If the context is insufficient, say so clearly. End factual claims with source tags like [S1]."
        )
        user_prompt = f"Question:\n{question}\n\nContext:\n{context_text}"
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
