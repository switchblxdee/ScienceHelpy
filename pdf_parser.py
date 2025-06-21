import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

PDF_DIR = settings.PATH_TO_PDFS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdfs(self, pdf_dir: Path = PDF_DIR) -> list:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        docs = []

        for file_path in pdf_dir.glob("*.pdf"):
            pdf_loader = PyMuPDFLoader(str(file_path))

            try:
                pdf_documents = pdf_loader.load()
                docs.extend(pdf_documents)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not docs:
            logger.warning("No PDF documents found in the specified directory.")
            return []

        texts_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ".", ""],
        )

        try:
            split_docs = texts_splitter.split_documents(docs)
            logger.info("Documents split into chunks successfully.")
            return split_docs
        except Exception as ex:
            logger.error("Error splitting documents: %s", ex)
            raise RuntimeError(f"Failed to split documents: {ex}")
