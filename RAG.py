from database import prepare_RAG_system

from config import PATH_TO_PDFS

def RAG_answer(query):
    vectorestorage = prepare_RAG_system(PATH_TO_PDFS)
    retriever = vectorestorage.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 1}
    )

    rel_docs = retriever.invoke(query)
    clear_list_of_docs = []
    for doc in rel_docs:
        clear_list_of_docs.append(doc.page_content) 

    return clear_list_of_docs