from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults()

def search_web(query: str) -> str:
    """
    run web search on the question
    """
    web_results = web_search_tool.invoke({"query": query})
    
    return [
        Document(page_content=d["content"], metadata={"url": d["url"]})
        for d in web_results
    ]