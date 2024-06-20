# create a LangGraph Graph State object

from typing import List, Optional, TypedDict
from langchain.schema import Document

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question input by the user
        generation: LLM generated answer
        web_search: a boolean flag to tell us whether to add web search
        documents: list of documents that could help answer the question
    """
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
