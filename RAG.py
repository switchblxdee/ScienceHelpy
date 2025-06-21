import logging

from config import settings
from database import prepare_RAG_system

PATH_TO_PDFS = settings.PATH_TO_PDFS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def RAG_answer(query: str) -> list[str]:
    vector_storage = prepare_RAG_system(PATH_TO_PDFS)
    retriever = vector_storage.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    try:
        rel_docs = retriever.invoke(query)
    except Exception as ex:
        logger.error("Error during retrieval: %s", ex)
        return []

    return [doc.page_content for doc in rel_docs]
