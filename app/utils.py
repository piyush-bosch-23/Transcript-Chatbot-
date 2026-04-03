import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage


def find_docx_file() -> str:
    files = [f for f in os.listdir(".") if f.lower().endswith(".docx")]
    if not files:
        raise FileNotFoundError("No .docx file found in the current directory.")

    preferred = [f for f in files if "data" in f.lower()]
    return preferred[0] if preferred else files[0]


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