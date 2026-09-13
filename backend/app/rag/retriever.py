import numpy as np

from backend.app.config import get_settings
from backend.app.llm.client import llm
from backend.app.rag.store import store
from backend.app.schemas import Citation

_settings = get_settings()


def _normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_search(
    query: str, top_k: int | None = None
) -> list[tuple[float, Citation]]:
    top_k = top_k or _settings.retrieval_top_k
    if not store.chunks:
        return []

    q_emb = llm.embed_query(query, output_dim=_settings.embed_dim)
    dense = store.dense_scores(q_emb)
    sparse = store.bm25_scores(query)

    combined = 0.6 * _normalize(dense) + 0.4 * _normalize(sparse)

    idx = np.argsort(-combined)[:top_k]
    results: list[tuple[float, Citation]] = []
    for i in idx:
        c = store.chunks[int(i)]
        results.append(
            (
                float(combined[int(i)]),
                Citation(
                    doc_id=c.doc_id,
                    section=c.section,
                    page=c.page,
                    snippet=c.text[:500],
                ),
            )
        )
    return results


def retrieve_with_confidence(query: str) -> tuple[list[Citation], float]:
    hits = hybrid_search(query)
    if not hits:
        return [], 0.0
    top_score = hits[0][0]
    threshold = _settings.retrieval_min_score
    citations = [c for s, c in hits if s >= threshold]
    return citations, top_score