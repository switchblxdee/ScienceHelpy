from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"

def create_embeddings(model_name=model_name):
    return HuggingFaceEmbeddings(model_name=model_name)
