import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.chat import router as chat_router
from backend.app.api.health import router as health_router
from backend.app.rag.ingest import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnicare")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — building RAG index...")
    try:
        n = build_index()
        logger.info(f"RAG index ready with {n} chunks")
    except Exception as e:
        logger.exception(f"Failed to build index: {e}")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="OmniCare Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)