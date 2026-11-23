from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


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
    openrouter_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL

    @property
    def ai_enabled(self) -> bool:
        return bool(self.openrouter_key or self.groq_key or self.ollama_model)

    @classmethod
    def from_env(cls):
        shared_path = os.getenv("OPENROUTER_ENV_FILE", "").strip()
        shared = dotenv_values(Path(shared_path).expanduser(), interpolate=False) if shared_path else {}

        def openrouter_setting(name: str) -> str:
            return os.getenv(name, "").strip() or (shared.get(name) or "").strip()

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
            openrouter_key=openrouter_setting("OPENROUTER_API_KEY"),
            openrouter_model=openrouter_setting("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL,
        )
