from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class GradeDocuments(BaseModel):
    """Binary classification for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Binary score for the document's relevance to the question.",
    )


llm = ChatOpenAI(temperature=0)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved Document: \n\n {document} \n\n User Question: \n\n {question}"),
    ]
)

llm_grader = llm.with_structured_output(schema=GradeDocuments)

retrieval_grader = (grade_prompt | llm_grader)
