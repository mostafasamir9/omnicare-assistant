from typing import Any, Optional

import numpy as np
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.app.config import get_settings

_settings = get_settings()

# gemini-embedding-2 only accepts ONE input per embed_content call.
# Sending more silently returns only the first embedding.
EMBED_BATCH_SIZE = 1


class LLMClient:
    """Thin wrapper around Gemini so the rest of the app doesn't depend on the SDK."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=_settings.gemini_api_key)
        self._chat_model = _settings.gemini_model
        self._embed_model = _settings.gemini_embed_model
        self._embed_dim = _settings.embed_dim

    # ---------- Embeddings ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        output_dim: Optional[int] = None,
    ) -> list[np.ndarray]:
        if not texts:
            return []

        dim = output_dim or self._embed_dim
        all_vectors: list[np.ndarray] = []

        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]

            resp = self._client.models.embed_content(
                model=self._embed_model,
                contents=batch,
                config={
                    "task_type": task_type,
                    "output_dimensionality": dim,
                },
            )

            embeddings = getattr(resp, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Batch embed mismatch at offset {start}: "
                    f"sent {len(batch)} texts, got {len(embeddings)} embeddings. "
                    f"Model: {self._embed_model}"
                )

            for i, e in enumerate(embeddings):
                values = getattr(e, "values", None)
                if values is None:
                    raise RuntimeError(
                        f"Embedding {start + i} returned no values."
                    )
                vec = np.asarray(values, dtype=np.float32).ravel()
                if vec.size != dim:
                    raise RuntimeError(
                        f"Embedding {start + i} has size {vec.size}, expected {dim}."
                    )
                all_vectors.append(vec)

        if len(all_vectors) != len(texts):
            raise RuntimeError(
                f"Total embeddings ({len(all_vectors)}) != input count ({len(texts)})"
            )

        return all_vectors

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed_query(self, text: str, output_dim: Optional[int] = None) -> np.ndarray:
        dim = output_dim or self._embed_dim
        return self.embed([text], task_type="RETRIEVAL_QUERY", output_dim=dim)[0]

    # ---------- Chat ----------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def generate(
        self,
        contents: list[types.Content],
        system_instruction: str,
        tools: Optional[list[types.Tool]] = None,
        temperature: float = 0.2,
    ) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            tools=tools if tools else None,
        )
        return self._client.models.generate_content(
            model=self._chat_model,
            contents=contents,
            config=config,
        )

    # ---------- Helpers ----------
    @staticmethod
    def text_from_response(response: types.GenerateContentResponse) -> str:
        try:
            parts = response.candidates[0].content.parts
        except (AttributeError, IndexError):
            return ""
        return "".join(p.text for p in parts if getattr(p, "text", None))

    @staticmethod
    def function_calls_from_response(
        response: types.GenerateContentResponse,
    ) -> list[Any]:
        try:
            parts = response.candidates[0].content.parts
        except (AttributeError, IndexError):
            return []
        return [p.function_call for p in parts if getattr(p, "function_call", None)]


llm = LLMClient()