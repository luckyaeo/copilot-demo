from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from graph.state import GraphState

llm = ChatOpenAI()

# And this is a very standard ReAct prompt that Lance Martin from the LangChain team wrote
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()

