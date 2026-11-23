from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path
    paper_list: Path
    events_db: Path
    model: str
    groq_key: str
    tavily_key: str
    ollama_model: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    public_url: str = ""

    @property
    def ai_enabled(self) -> bool:
        return bool(self.groq_key or self.ollama_model)

    @classmethod
    def from_env(cls):
        return cls(
            pdf_dir=Path(os.getenv("PATH_TO_PDFS", ROOT / "pdfs")),
            paper_list=Path(os.getenv("PATH_TO_PDFS_URL", ROOT / "papers.txt")),
            events_db=Path(os.getenv("SCIENCEHELPY_EVENTS_DB", ROOT / ".sciencehelpy/events.sqlite3")),
            model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
            groq_key=os.getenv("GROQ_API", ""),
            tavily_key=os.getenv("TAVILY_API", ""),
            ollama_model=os.getenv("SCIENCEHELPY_OLLAMA_MODEL", "").strip(),
            ollama_url=os.getenv("SCIENCEHELPY_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            public_url=os.getenv("SCIENCEHELPY_PUBLIC_URL", "").rstrip("/"),
        )
