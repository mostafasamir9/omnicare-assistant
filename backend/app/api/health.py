from fastapi import APIRouter

from backend.app.rag.store import store

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": len(store.chunks),
    }