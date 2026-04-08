from langchain_core.messages import HumanMessage, AIMessage

from app.loader import load_and_split_transcript
from app.retriever import build_retriever_with_logging
from app.graph_builder import build_graph


def main():
    docx_files, split_docs = load_and_split_transcript()
    retriever, docs_added, docs_skipped = build_retriever_with_logging(split_docs)
    graph = build_graph(retriever)

    print("=" * 70)
    print("Transcript Chatbot")
    print(f"Loaded DOCX files: {len(docx_files)}")
    for docx_file in docx_files:
        print(f" - {docx_file}")

    if docs_added:
        print(f"\nNew documents indexed: {len(docs_added)}")
        for doc_name in docs_added:
            print(f"  + {doc_name}")

    if docs_skipped:
        print(f"\nDocuments already indexed: {len(docs_skipped)}")
        for doc_name in docs_skipped:
            print(f"  ~ {doc_name}")

    print("=" * 70)
    print(graph.get_graph(xray=True).draw_ascii())

    state = {
        "messages": [],
        "summary": "",
        "retrieved_context": "",
        "retrieved_sources": [],
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

        retrieved_sources = result.get("retrieved_sources", [])
        if retrieved_sources:
            print("\nSOURCE DOCUMENTS:")
            print("-" * 70)
            for source_name in retrieved_sources:
                print(f" - {source_name}")

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
            "retrieved_sources": [],
            "has_context": False,
        }


if __name__ == "__main__":
    main()