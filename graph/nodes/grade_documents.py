from typing import Any, Dict, List
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from langchain.schema import Document

def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("---grade documents---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
