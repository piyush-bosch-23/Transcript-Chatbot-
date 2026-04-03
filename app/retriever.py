from langchain_community.retrievers import TFIDFRetriever

from app.config import RETRIEVER_TOP_K


def build_retriever(split_docs):
    retriever = TFIDFRetriever.from_documents(split_docs)
    retriever.k = RETRIEVER_TOP_K
    return retriever


def get_relevant_context(retriever, query: str) -> tuple[str, bool]:
    docs = retriever.invoke(query)

    useful_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if text and len(text) > 40:
            useful_docs.append(text)

    context = "\n\n---\n\n".join(useful_docs[:RETRIEVER_TOP_K])
    has_context = bool(useful_docs)

    return context, has_context