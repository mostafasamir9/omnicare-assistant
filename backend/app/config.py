from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "gemini-embedding-2"

    embed_dim: int = 768

    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.35
    max_tokens_per_session: int = 50_000
    max_tool_calls_per_session: int = 20

    policy_dir: str = "backend/data/policies"


@lru_cache
def get_settings() -> Settings:
    return Settings()