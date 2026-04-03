import os
import requests
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever


# ============================================================
# Environment
# ============================================================
#load_dotenv()

SUBSCRIPTION_KEY = os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY")
if not SUBSCRIPTION_KEY:
    raise ValueError(
        "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY is not set. Add it in your .env file."
    )

BOSCH_URL = (
    "https://aoai-farm.bosch-temp.com/api/openai/deployments/"
    "askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/chat/completions"
    "?api-version=2024-08-01-preview"
)

PROXIES = {
    "http": "http://127.0.0.1:3128",
    "https": "http://127.0.0.1:3128",
}


# ============================================================
# Bosch client
# ============================================================
def ask_bosch(messages: list[dict], api_key: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "genaiplatform-farm-subscription-key": api_key,
    }

    payload = {
        "messages": messages,
        "max_tokens": 1200,   # enough for answer/summaries; keeps cost reasonable
        "temperature": 0.2,
        "stream": False,
    }

    response = requests.post(
        BOSCH_URL,
        headers=headers,
        json=payload,
        proxies=PROXIES,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if content is None:
        return ""
    return content


# ============================================================
# Helpers
# ============================================================
def to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    converted = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            converted.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            converted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            converted.append({"role": "assistant", "content": msg.content or ""})
        else:
            converted.append({"role": "user", "content": str(msg.content)})
    return converted


def find_docx_file() -> str:
    files = [f for f in os.listdir(".") if f.lower().endswith(".docx")]
    if not files:
        raise FileNotFoundError("No .docx file found in the current directory.")
    preferred = [f for f in files if "data" in f.lower()]
    return preferred[0] if preferred else files[0]


# ============================================================
# Load + split + retriever
# ============================================================
DOCX_FILE = find_docx_file()

loader = Docx2txtLoader(DOCX_FILE)
documents = loader.load()

# Token-aware splitter.
# Since transcript is ~8260 chars, these chunk settings are a good fit:
# roughly moderate chunks with overlap for continuity.
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=80,
)

split_docs = splitter.split_documents(documents)

retriever = TFIDFRetriever.from_documents(split_docs)
retriever.k = 4


# ============================================================
# State
# ============================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    retrieved_context: str
    has_context: bool


# ============================================================
# Nodes
# ============================================================
def retrieve_context(state: ChatState) -> dict:
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {
            "retrieved_context": "",
            "has_context": False,
        }

    docs = retriever.invoke(last_user_msg)

    # Small relevance guard for TF-IDF retrieval:
    # if nothing meaningful is returned, treat as no context.
    useful_docs = []
    for d in docs:
        text = d.page_content.strip()
        if text and len(text) > 40:
            useful_docs.append(text)

    context = "\n\n---\n\n".join(useful_docs[:4])

    return {
        "retrieved_context": context,
        "has_context": bool(useful_docs),
    }


def answer_from_context(state: ChatState) -> dict:
    system_prompt = (
        "You are a helpful course chatbot for the transcript document of the course "
        "'Introduction to Data_and_Data_Science'. "
        "The transcript covers these chapters:\n"
        "1. Analysis vs Analytics\n"
        "2. Programming Languages & Software Employed in Data Science - All the Tools You Need\n\n"
        "Rules:\n"
        "- Answer only from the retrieved transcript context.\n"
        "- Use the conversation summary when useful for continuity.\n"
        "- If the answer is not clearly supported by the context, say: "
        "'I could not find that in the provided transcript.'\n"
        "- Keep the response clear, direct, and student-friendly.\n"
        "- End with one short follow-up line: 'What more would you like to know?'"
    )

    # Keep only the most recent few raw messages, plus the running summary.
    recent_messages = state["messages"][-4:]

    user_prompt = (
        f"Conversation summary so far:\n{state['summary'] or 'No summary yet.'}\n\n"
        f"Retrieved transcript context:\n{state['retrieved_context']}\n\n"
        "Answer the user's latest question using only the retrieved transcript context."
    )

    bosch_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        *to_openai_messages(recent_messages),
    ]

    answer = ask_bosch(bosch_messages, SUBSCRIPTION_KEY)
    return {"messages": [AIMessage(content=answer)]}


def answer_no_context(state: ChatState) -> dict:
    system_prompt = (
        "You are a helpful course chatbot. "
        "If no transcript support is available, say so clearly and do not invent facts. "
        "End with: 'What more would you like to know?'"
    )

    recent_messages = state["messages"][-2:]

    user_prompt = (
        f"Conversation summary so far:\n{state['summary'] or 'No summary yet.'}\n\n"
        "There was no reliable transcript context retrieved for the latest question. "
        "Please answer safely by saying the information could not be found in the provided transcript."
    )

    bosch_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        *to_openai_messages(recent_messages),
    ]

    answer = ask_bosch(bosch_messages, SUBSCRIPTION_KEY)
    return {"messages": [AIMessage(content=answer)]}


def summarize_history(state: ChatState) -> dict:
    """
    Compress older conversation into a running summary, and keep only the latest turns.
    """
    existing_summary = state["summary"] or "No summary yet."

    # Summarize everything except the latest 2 messages
    # (typically latest Human + latest AI).
    old_messages = state["messages"][:-2]
    latest_messages = state["messages"][-2:]

    if not old_messages:
        return {}

    summary_prompt = (
        "Create a concise running summary of the conversation for future context.\n"
        "Keep important facts, user intent, unresolved points, and references to the transcript.\n"
        "Do not include unnecessary wording.\n\n"
        f"Existing summary:\n{existing_summary}\n\n"
        "Older messages to compress:\n"
    )

    bosch_messages = [
        {"role": "system", "content": "You summarize conversations compactly and accurately."},
        {"role": "user", "content": summary_prompt},
        *to_openai_messages(old_messages),
    ]

    new_summary = ask_bosch(bosch_messages, SUBSCRIPTION_KEY).strip()

    # Replace history with only the newest messages
    return {
        "summary": new_summary,
        "messages": latest_messages,
    }


# ============================================================
# Conditional routing
# ============================================================
def route_after_retrieval(state: ChatState) -> Literal["answer_from_context", "answer_no_context"]:
    if state["has_context"]:
        return "answer_from_context"
    return "answer_no_context"


def route_after_answer(state: ChatState) -> Literal["summarize_history", END]:
    # If messages are growing, summarize.
    # Since we store only short sessions, 6 is a good threshold.
    if len(state["messages"]) > 6:
        return "summarize_history"
    return END


# ============================================================
# Build graph
# ============================================================
builder = StateGraph(ChatState)

builder.add_node("retrieve_context", retrieve_context)
builder.add_node("answer_from_context", answer_from_context)
builder.add_node("answer_no_context", answer_no_context)
builder.add_node("summarize_history", summarize_history)

builder.add_edge(START, "retrieve_context")
builder.add_conditional_edges("retrieve_context", route_after_retrieval)
builder.add_conditional_edges("answer_from_context", route_after_answer)
builder.add_conditional_edges("answer_no_context", route_after_answer)
builder.add_edge("summarize_history", END)

graph = builder.compile()


# ============================================================
# CLI
# ============================================================
def main():
    print("=" * 70)
    print("Transcript Chatbot Ready")
    print(f"Loaded DOCX: {DOCX_FILE}")
    print("Course: Introduction to Data_and_Data_Science")
    print("Chapters:")
    print(" - Analysis vs Analytics")
    print(" - Programming Languages & Software Employed in Data Science - All the Tools You Need")
    print("=" * 70)
    print(graph.get_graph().draw_ascii())
    state: ChatState = {
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

        state["messages"] = add_messages(state["messages"], [HumanMessage(content=question)])

        result = graph.invoke(state)

        last_ai = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                last_ai = msg
                break

        print("\nAnswer:\n")
        print(last_ai.content if last_ai else "No answer generated.")

        # carry forward compact state
        state = {
            "messages": result["messages"],
            "summary": result.get("summary", state["summary"]),
            "retrieved_context": "",
            "has_context": False,
        }


if __name__ == "__main__":
    main()