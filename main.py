from download_pdf import download_all_papers
from graph import Graph

def main():
    PATH_TO_PDFS_URL = "C:/Users/Егор/Documents/vs_code/ML/Science_Helpy_0.2/papers.txt"
    download_all_papers(PATH_TO_PDFS_URL)
    model = Graph()
    query = input("Давай начнем диалог: ")
    while query != "0":
        print(model.run(user_prompt=query)["messages"][-1].content)
        print("Ваше сообщение: ")
        query=input()

if __name__ == "__main__":
    main()
