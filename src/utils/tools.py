from typing import List,Callable,Dict
from rank_bm25 import BM25Okapi
import re

class LocalSearchTool:
    def __init__(self, context:str):

        self.documents = self._parse_documents(context)
        self.corpus = [
            self._simple_tokenize(doc["title"] + " " + doc["text"])
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus)

    def _parse_documents(self, context: str) -> List[Dict]:
        pattern = r"Document \[(\d+)\] \(Titile: (.*?)\)\s*\n(.*?)(?=\n\nDocument \[\d+\] \(Titile:|\Z)"
        matches = re.findall(pattern, context, flags=re.S)

        docs = []
        for doc_id, title, text in matches:
            docs.append({
                "doc_id": int(doc_id),
                "title": title.strip(),
                "text": text.strip()
            })
        return docs

    def _simple_tokenize(self,text: str):
        text = text.lower()
        return re.findall(r"\w+", text)


    def search(self, query: str, top_k: int = 5) -> str:
        """
        Search for information relevant to the query.

        Args:
            query (str): The search query.

        Returns:
            str: Relevant documents retrieved from the context.
        """
        tokenized_query = self._simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        scored_docs = list(zip(self.documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_docs = scored_docs[:top_k]

        if not top_docs:
            return "No relevant documents found."

        results = []
        for i, (doc, _) in enumerate(top_docs, start=1):
            results.append(
                f"[{i}] {doc['title']}\n{doc['text']}"
            )

        res = "\n\n".join(results)
        return res
    
    def __call__(self, query: str) -> str:
        return self.search(query)


def get_current_weather(location: str, format: str):
    """
    Get the current weather

    Args:
        location: The city and state, e.g. San Francisco, CA
        format: The temperature unit to use. Infer this from the users location. (choices: ["celsius", "fahrenheit"])
    """
    return 'it is currently 20 degrees celsius in Paris.'


def make_search_tool(context:str):
    search_engine= LocalSearchTool(context)
    def search(query: str) -> str:
        """
        Search for information relevant to the query.

        Args:
            query: The search query.

        Returns:
            str: Relevant documents retrieved from the context.
        """
        return search_engine.search(query)
    return search

