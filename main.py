from download_pdf import download_all_papers
from graph import GraphLangGraph

def main():
    PATH_TO_PDFS_URL = "C:/Users/Егор/Documents/vs_code/ML/Science_Helpy_0.2/papers.txt"
    download_all_papers(PATH_TO_PDFS_URL)
    model = GraphLangGraph()
    query = "Привет, какая ты модель?"
    while query != "0":
        print(model.generate(user_prompt=query)["messages"])
        print("Ваше сообщение: \n")
        query=input()

if __name__ == "__main__":
    main()
