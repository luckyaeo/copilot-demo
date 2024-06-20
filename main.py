from dotenv import load_dotenv
load_dotenv()

from graph.graph import app


if __name__ == "__main__":
    print("Advanced RAG")
#    print(app.invoke(input={"question": "What is the capital of France?"}))
#    print(app.invoke(input={"question": "What is convertible arbitrage?"}))
    print(app.invoke(input={"question": "What risks does convertible arbitrage have?"}))
