from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from config import TAVILY_API

search_tool = TavilySearchResults(tavily_api_key=TAVILY_API, max_results=3)

@tool
def internet_search(query: str) -> str:
    """
    Perform an internet search and return combined results.
    """
    results = search_tool.invoke(query)
    combined_content = " ".join([res["content"] for res in results])
    
    return combined_content