from dotenv import load_dotenv

load_dotenv()

from graph.consts import WEBSEARCH, GENERATE, GRADE_DOCUMENTS, RETRIEVE
from langgraph.graph import END, StateGraph

from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState


def decide_to_websearch_or_generate(state: GraphState) -> str:
    print("---access graded documents---")

    if state["web_search"]:
        print("---Decision: not all documents are relevant to the question, include web search")
        return WEBSEARCH
    else:
        print("---Decision: all documents are relevant to the question, generate answer")
        return GENERATE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    decide_to_websearch_or_generate, 
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")




