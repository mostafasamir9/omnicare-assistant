from backend.app.config import get_settings
from backend.app.llm.client import llm
from backend.app.rag.store import load_policies, store

_settings = get_settings()


def build_index() -> int:
    n = load_policies(_settings.policy_dir)
    if n == 0:
        print("[ingest] no chunks found")
        return 0

    dim = _settings.embed_dim
    print(
        f"[ingest] embedding {n} chunks with {_settings.gemini_embed_model} "
        f"(dim={dim})..."
    )
    texts = [c.text for c in store.chunks]
    embeddings = llm.embed(texts, task_type="RETRIEVAL_DOCUMENT", output_dim=dim)

    if len(embeddings) != len(store.chunks):
        raise RuntimeError(
            f"Got {len(embeddings)} embeddings for {len(store.chunks)} chunks."
        )

    for chunk, emb in zip(store.chunks, embeddings):
        chunk.embedding = emb

    store._validate_embeddings()

    print(f"[ingest] indexed {n} chunks (dim={dim})")
    return n