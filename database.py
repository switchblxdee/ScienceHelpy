from langchain_chroma import Chroma
from pdf_parser import PDFChunker
from emb_model import create_embeddings
import os

class ChromaDB:
    def __init__(self):
        pass

    def create_vectorestorage(self, docs, embeddings):
        if os.path.exists("./chroma_db"):
            return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        else:
            vectorestorage = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            return vectorestorage
        
def prepare_RAG_system(pdf_directory):
    pdf_chunker = PDFChunker()
    chromadb = ChromaDB()

    documents = pdf_chunker.load_pdfs(pdf_directory)
    embeddings = create_embeddings()
    vectorstorage = chromadb.create_vectorestorage(docs=documents, embeddings=embeddings)

    return vectorstorage