import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import settings
from emb_model import create_embeddings
from pdf_parser import PDFChunker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHROMA_DB_DIR = settings.CHROMA_DB_DIR


class ChromaDB:
    def __init__(self, persist_directory: Path = CHROMA_DB_DIR):
        self.persist_directory = persist_directory

    def create_vector_storage(
        self, docs: list[Document], embeddings: Embeddings
    ) -> Chroma:
        if self.persist_directory.exists():
            return Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=embeddings,
            )

        db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=str(self.persist_directory),
        )
        return db


def prepare_RAG_system(pdf_directory: str) -> Chroma:
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        logger.error("Директория с PDF не найдена: %s", pdf_dir)
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_chunker = PDFChunker()
    documents = pdf_chunker.load_pdfs(str(pdf_dir))
    embeddings = create_embeddings()
    chromadb = ChromaDB()

    vector_storage = chromadb.create_vector_storage(
        docs=documents, embeddings=embeddings
    )

    return vector_storage
