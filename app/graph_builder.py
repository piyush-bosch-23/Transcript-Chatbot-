from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from app.bosch_client import ask_bosch
from app.config import (
    SUMMARY_TRIGGER_MESSAGE_COUNT,
    KEEP_LAST_MESSAGES,
    ANSWER_RECENT_MESSAGES,
)
from app.retriever import get_relevant_context
from app.utils import to_openai_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    retrieved_context: str
    retrieved_sources: list[str]
    retrieved_context_chunks: list[dict[str, str]]
    has_context: bool


def build_graph(retriever):
    def retrieve_context(state: ChatState) -> dict:
        print("\n[Node] retrieve_context")
        last_user_msg = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return {
                "retrieved_context": "",
                "retrieved_sources": [],
                "retrieved_context_chunks": [],
                "has_context": False,
            }

        context, has_context, source_names, context_chunks = get_relevant_context(retriever, last_user_msg)

        return {
            "retrieved_context": context,
            "retrieved_sources": source_names,
            "retrieved_context_chunks": context_chunks,
            "has_context": has_context,
        }

    def answer_from_context(state: ChatState) -> dict:
        print("\n[Node] answer_from_context")
        system_prompt = (
            "You are a helpful chatbot for the provided transcript documents.\n"
            "Rules:\n"
            "- Answer only from the retrieved transcript context.\n"
            "- Use the conversation summary when useful for continuity.\n"
            "- If the answer is not clearly supported by the context, say: "
            "'I could not find that in the provided transcript.'\n"
            "- Keep the response clear, direct, and student-friendly.\n"
            "- End with one short follow-up line: 'What more would you like to know?'"
        )

        recent_messages = state["messages"][-ANSWER_RECENT_MESSAGES:]

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

        answer = ask_bosch(bosch_messages)
        return {"messages": [AIMessage(content=answer)]}

    def answer_no_context(state: ChatState) -> dict:
        print("\n[Node] answer_no_context")
        system_prompt = (
            "You are a helpful transcript chatbot. "
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

        answer = ask_bosch(bosch_messages)
        return {"messages": [AIMessage(content=answer)]}

    def summarize_history(state: ChatState) -> dict:
        print("\n[Node] summarize_history")
        existing_summary = state["summary"] or "No summary yet."

        old_messages = state["messages"][:-KEEP_LAST_MESSAGES]
        latest_messages = state["messages"][-KEEP_LAST_MESSAGES:]

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

        new_summary = ask_bosch(bosch_messages).strip()

        return {
            "summary": new_summary,
            "messages": latest_messages,
        }

    def route_after_retrieval(state: ChatState) -> Literal["answer_from_context", "answer_no_context"]:
        if state["has_context"]:
            return "answer_from_context"
        return "answer_no_context"

    def route_after_answer(state: ChatState) -> Literal["summarize_history", "__end__"]:
        if len(state["messages"]) > SUMMARY_TRIGGER_MESSAGE_COUNT:
            return "summarize_history"
        return "__end__"

    builder = StateGraph(ChatState)

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("answer_from_context", answer_from_context)
    builder.add_node("answer_no_context", answer_no_context)
    builder.add_node("summarize_history", summarize_history)

    builder.add_edge(START, "retrieve_context")
    builder.add_conditional_edges("retrieve_context", route_after_retrieval)
    builder.add_conditional_edges(
        "answer_from_context",
        route_after_answer,
        {
            "summarize_history": "summarize_history",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "answer_no_context",
        route_after_answer,
        {
            "summarize_history": "summarize_history",
            "__end__": END,
        },
    )
    builder.add_edge("summarize_history", END)

    return builder.compile()