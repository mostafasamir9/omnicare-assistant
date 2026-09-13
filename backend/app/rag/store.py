import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from backend.app.config import get_settings

_settings = get_settings()


@dataclass
class Chunk:
    doc_id: str
    policy_type: str
    state: str
    effective_date: str
    section: str
    page: int
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("---", 3)
    if end == -1:
        return {}, raw
    fm_text = raw[3:end].strip()
    body = raw[end + 3:].strip()
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def _split_sections(body: str) -> list[tuple[str, int, str]]:
    lines = body.splitlines()
    sections: list[tuple[str, int, str]] = []
    current_heading = "General"
    current_lines: list[str] = []
    page = 1
    for line in lines:
        if line.strip().startswith("#"):
            if current_lines:
                sections.append(
                    (current_heading, page, "\n".join(current_lines).strip())
                )
                current_lines = []
            current_heading = line.strip("# ").strip()
        else:
            current_lines.append(line)
            if len(current_lines) % 40 == 0:
                page += 1
    if current_lines:
        sections.append((current_heading, page, "\n".join(current_lines).strip()))
    return sections


class InMemoryStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None

    def clear(self) -> None:
        self.chunks = []
        self._bm25 = None

    def add(self, chunk: Chunk) -> None:
        """Append, but replace if (doc_id, section) already exists."""
        for i, existing in enumerate(self.chunks):
            if existing.doc_id == chunk.doc_id and existing.section == chunk.section:
                self.chunks[i] = chunk
                return
        self.chunks.append(chunk)

    def build_bm25(self) -> None:
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None or not self.chunks:
            return np.zeros(len(self.chunks))
        return np.array(self._bm25.get_scores(self._tokenize(query)))

    def _validate_embeddings(self) -> None:
        expected = _settings.embed_dim
        bad = []
        for i, c in enumerate(self.chunks):
            if c.embedding is None:
                bad.append((i, c.doc_id, c.section, "None"))
                continue
            if c.embedding.ndim != 1:
                bad.append((i, c.doc_id, c.section, f"ndim={c.embedding.ndim}"))
                continue
            if c.embedding.size != expected:
                bad.append((i, c.doc_id, c.section, f"size={c.embedding.size}"))
        if bad:
            raise RuntimeError(
                f"Invalid embeddings in {len(bad)} chunks (expected dim={expected}): "
                f"{bad[:5]}. Run build_index() before serving requests."
            )

    def dense_scores(self, query_emb: np.ndarray) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0)
        self._validate_embeddings()
        mat = np.vstack([c.embedding for c in self.chunks])
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        return m @ q


store = InMemoryStore()


def load_policies(directory: str) -> int:
    store.clear()
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Policy dir not found: {directory}")
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = _parse_frontmatter(raw)
        sections = _split_sections(body)
        for heading, page, text in sections:
            if not text.strip():
                continue
            store.add(
                Chunk(
                    doc_id=meta.get("doc_id", fname),
                    policy_type=meta.get("policy_type", "unknown"),
                    state=meta.get("state", "NA"),
                    effective_date=meta.get("effective_date", "1970-01-01"),
                    section=heading,
                    page=page,
                    text=text,
                )
            )
    store.build_bm25()
    return len(store.chunks)