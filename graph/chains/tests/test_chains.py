from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from graph.chains.generation import generation_chain
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "Convertible Bond"
    expected_output = "yes" 

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    result: GradeDocuments = retrieval_grader.invoke({"document": doc_txt, "question": question})
    assert result.binary_score == expected_output

def test_retrieval_grader_answer_no() -> None:
    question = "Nvidia stock price"
    expected_output = "no" 

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    result: GradeDocuments = retrieval_grader.invoke({"document": doc_txt, "question": question})
    assert result.binary_score == expected_output

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:
def test_generation_chain() ->  None:
    question = "convertible arbitrage"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)














