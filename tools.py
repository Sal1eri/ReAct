
# add the wiki search tool instead of the tavily search tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def search(query: str) :
    """Search for general web results.

    This function performs a search using the Wiki search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    search_res = wikipedia.run(query)

    print(f"Search results for query: '{query}'\n{search_res}")
    return search_res

search("Openai")


