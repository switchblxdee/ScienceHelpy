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

    @classmethod
    def from_env(cls):
        return cls(
            pdf_dir=Path(os.getenv("PATH_TO_PDFS", ROOT / "pdfs")),
            paper_list=Path(os.getenv("PATH_TO_PDFS_URL", ROOT / "papers.txt")),
            events_db=Path(os.getenv("SCIENCEHELPY_EVENTS_DB", ROOT / ".sciencehelpy/events.sqlite3")),
            model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
            groq_key=os.getenv("GROQ_API", ""),
            tavily_key=os.getenv("TAVILY_API", ""),
        )
