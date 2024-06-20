from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---retrieve documents based on the question---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}
