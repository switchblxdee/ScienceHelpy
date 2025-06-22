from langchain_huggingface import HuggingFaceEmbeddings

# settings.MODEL_NAME mb?
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(name: str | None = None) -> HuggingFaceEmbeddings:
    name = name or MODEL_NAME
    return HuggingFaceEmbeddings(model_name=name)
