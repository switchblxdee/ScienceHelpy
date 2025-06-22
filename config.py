from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PATH_TO_PDFS_URL: str
    PATH_TO_PDFS: Path
    CHROMA_DB_DIR: Path
    MODEL_NAME: str
    GROQ_API: str
    TAVILY_API: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
