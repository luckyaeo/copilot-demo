from typing import Any, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from graph.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---web_search---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(d["content"] for d in docs)
    web_results = Document(page_content=web_results)
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    state = GraphState({"question": "What is the capital of France?", "documents": None, "generation": None, "web_search": None})
    print(web_search(state))