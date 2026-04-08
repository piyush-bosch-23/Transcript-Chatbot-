import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage


def find_docx_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Transcript folder not found: {folder}")

    files = [
        os.path.join(folder, file_name)
        for file_name in os.listdir(folder)
        if file_name.lower().endswith(".docx")
    ]

    files.sort(key=lambda p: os.path.basename(p).lower())

    if not files:
        raise FileNotFoundError(f"No .docx files found in folder: {folder}")

    return files


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