from langchain_core.messages import HumanMessage, AIMessage

from app.loader import load_and_split_transcript
from app.retriever import build_retriever
from app.graph_builder import build_graph
from app.config import COURSE_NAME, CHAPTERS


def main():
    docx_file, split_docs = load_and_split_transcript()
    retriever = build_retriever(split_docs)
    graph = build_graph(retriever)

    print("=" * 70)
    print("Transcript Chatbot Ready")
    print(f"Loaded DOCX: {docx_file}")
    print(f"Course: {COURSE_NAME}")
    print("Chapters:")
    for chapter in CHAPTERS:
        print(f" - {chapter}")
    print("=" * 70)
    print(graph.get_graph(xray=True).draw_ascii())

    state = {
        "messages": [],
        "summary": "",
        "retrieved_context": "",
        "has_context": False,
    }

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=question)]

        result = graph.invoke(state)

        last_ai = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break

        print("\n" + "=" * 70)
        print("ANSWER:")
        print("-" * 70)
        print(last_ai.content if last_ai else "No answer generated.")

        print("\nRETRIEVED TRANSCRIPT CONTEXT:")
        print("-" * 70)
        retrieved_context = result.get("retrieved_context", "").strip()

        if retrieved_context:
            first_chunk = retrieved_context.split("\n\n---\n\n")[0]
            print(first_chunk)
        else:
            print("No relevant transcript context found.")

        print("=" * 70)

        state = {
            "messages": result["messages"],
            "summary": result.get("summary", state["summary"]),
            "retrieved_context": "",
            "has_context": False,
        }


if __name__ == "__main__":
    main()