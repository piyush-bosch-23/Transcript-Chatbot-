import os

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from app.loader import load_and_split_transcript
from app.retriever import build_retriever_with_logging, get_chroma_stats
from app.graph_builder import build_graph
from app.config import TRANSCRIPTS_FOLDER
from app.utils import find_docx_files


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Analysis Chatbot",
    page_icon="📚",
    layout="wide",
)

UPLOADS_DIR = os.path.join(TRANSCRIPTS_FOLDER, "_ui_uploads")


def reset_chat_state():
    st.session_state.chat_messages = []
    st.session_state.graph_state = {
        "messages": [],
        "summary": "",
        "retrieved_context": "",
        "retrieved_sources": [],
        "retrieved_context_chunks": [],
        "has_context": False,
    }


def list_uploaded_docx_files() -> list[str]:
    if not os.path.isdir(UPLOADS_DIR):
        return []
    try:
        return find_docx_files(UPLOADS_DIR)
    except FileNotFoundError:
        return []


def get_all_docx_files() -> list[str]:
    docx_files = []

    if os.path.isdir(TRANSCRIPTS_FOLDER):
        try:
            for path in find_docx_files(TRANSCRIPTS_FOLDER):
                if os.path.dirname(path) != os.path.abspath(UPLOADS_DIR):
                    docx_files.append(path)
        except FileNotFoundError:
            pass

    docx_files.extend(list_uploaded_docx_files())
    return docx_files


def save_uploaded_files(uploaded_files) -> list[str]:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    saved_files = []
    progress_bar = st.progress(0, text="Uploading documents...")
    status_placeholder = st.empty()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        file_name = os.path.basename(uploaded_file.name)
        target_path = os.path.join(UPLOADS_DIR, file_name)
        already_exists = os.path.exists(target_path)

        status_placeholder.write(f"Saving {file_name} ({index}/{len(uploaded_files)})")

        with open(target_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())

        saved_files.append({
            "name": file_name,
            "path": target_path,
            "status": "updated" if already_exists else "uploaded",
        })
        progress_value = int(index / len(uploaded_files) * 100)
        progress_bar.progress(progress_value, text=f"Processed {index} of {len(uploaded_files)} document(s)")

    progress_bar.empty()
    status_placeholder.empty()
    return saved_files


# ── Load graph once per session ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading transcript documents and building index…")
def load_graph(docx_files: tuple[str, ...]):
    docx_files, split_docs = load_and_split_transcript(list(docx_files))
    retriever, docs_added, docs_skipped = build_retriever_with_logging(split_docs)
    graph = build_graph(retriever)
    return graph, docx_files, docs_added, docs_skipped


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "graph_state" not in st.session_state:
    reset_chat_state()

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "chroma_status" not in st.session_state:
    st.session_state.chroma_status = {"added": [], "skipped": []}

all_docx_files = get_all_docx_files()

if all_docx_files:
    graph, docx_files, docs_added, docs_skipped = load_graph(tuple(all_docx_files))
    st.session_state.chroma_status = {"added": docs_added, "skipped": docs_skipped}
else:
    graph, docx_files = None, []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Document Analysis Chatbot")
    st.markdown("---")
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add one or more .docx files",
        type=["docx"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents", use_container_width=True):
        if not uploaded_files:
            st.info("Choose at least one .docx file to upload.")
        else:
            saved_files = save_uploaded_files(uploaded_files)
            st.session_state.upload_results = saved_files
            load_graph.clear()
            reset_chat_state()
            st.success(f"Indexed {len(saved_files)} uploaded document(s).")
            st.rerun()

    if st.session_state.upload_results:
        st.caption("Latest upload status")
        for upload_result in st.session_state.upload_results:
            st.markdown(
                f"- {upload_result['name']}: {upload_result['status']}"
            )

    uploaded_docx_files = list_uploaded_docx_files()
    if uploaded_docx_files:
        st.caption("Uploaded in UI session storage")
        for file_path in uploaded_docx_files:
            st.markdown(f"- `{os.path.basename(file_path)}`")

        if st.button("Remove uploaded documents", use_container_width=True):
            for file_path in uploaded_docx_files:
                os.remove(file_path)
            st.session_state.upload_results = []
            load_graph.clear()
            reset_chat_state()
            st.rerun()

    st.markdown("---")
    st.subheader("Loaded Documents")
    for f in docx_files:
        st.markdown(f"- `{os.path.basename(f)}`")

    if st.session_state.chroma_status.get("added") or st.session_state.chroma_status.get("skipped"):
        st.markdown("---")
        st.subheader("Vector Index Status")
        if st.session_state.chroma_status.get("added"):
            with st.expander("✅ New documents indexed"):
                for doc_name in st.session_state.chroma_status.get("added", []):
                    st.caption(f"+ {doc_name}")
        if st.session_state.chroma_status.get("skipped"):
            with st.expander("~ Already indexed"):
                for doc_name in st.session_state.chroma_status.get("skipped", []):
                    st.caption(f"~ {doc_name}")

    # Display ChromaDB statistics
    st.markdown("---")
    st.subheader("📊 Vector Index Statistics")
    stats = get_chroma_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indexed Documents", stats["unique_docs_count"])
    with col2:
        st.metric("Total Chunks", stats["total_chunks"])
    with col3:
        st.metric("Disk Usage", f"{stats['disk_usage_mb']} MB")

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset_chat_state()
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.header("💬 Ask questions about your Documents")

if not docx_files:
    st.info("Add transcript .docx files in the sidebar or place them in the transcripts folder to start chatting.")


def render_retrieved_context(context: str):
    chunks = [chunk.strip() for chunk in context.split("\n\n---\n\n") if chunk.strip()]
    if not chunks:
        return

    for index, chunk in enumerate(chunks, start=1):
        st.markdown(f"**Excerpt {index}**")
        st.markdown(chunk)
        if index < len(chunks):
            st.markdown("---")


def render_retrieved_context_chunks(context_chunks: list[dict[str, str]]):
    if not context_chunks:
        return

    for index, chunk in enumerate(context_chunks, start=1):
        source_name = chunk.get("source", "Unknown source")
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text:
            continue

        st.markdown(f"**Excerpt {index}**")
        st.caption(f"Source: {source_name}")
        st.markdown(chunk_text)
        if index < len(context_chunks):
            st.markdown("---")

# Render existing messages
for entry in st.session_state.chat_messages:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("sources"):
            st.caption(f"Sources: {', '.join(entry['sources'])}")
        if entry["role"] == "assistant" and entry.get("context"):
            with st.expander("📄 Retrieved Documents context", expanded=False):
                if entry.get("context_chunks"):
                    render_retrieved_context_chunks(entry["context_chunks"])
                else:
                    render_retrieved_context(entry["context"])


# Input box pinned at the bottom
user_input = st.chat_input(
    "Ask a question about the transcript…",
    disabled=not bool(graph),
)

if user_input and graph:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    # Run the LangGraph pipeline
    state = st.session_state.graph_state
    state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

    with st.spinner("Thinking…"):
        result = graph.invoke(state)

    # Extract latest AI answer
    last_ai = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg
            break

    answer = last_ai.content if last_ai else "No answer generated."
    retrieved_context = result.get("retrieved_context", "").strip()
    retrieved_sources = result.get("retrieved_sources", [])
    retrieved_context_chunks = result.get("retrieved_context_chunks", [])

    # Show assistant message with optional context expander
    with st.chat_message("assistant"):
        st.markdown(answer)
        if retrieved_sources:
            st.caption(f"Sources: {', '.join(retrieved_sources)}")
        if retrieved_context:
            with st.expander("📄 Retrieved Documents context", expanded=False):
                if retrieved_context_chunks:
                    render_retrieved_context_chunks(retrieved_context_chunks)
                else:
                    render_retrieved_context(retrieved_context)

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_sources,
        "context": retrieved_context,
        "context_chunks": retrieved_context_chunks,
    })

    # Persist updated graph state
    st.session_state.graph_state = {
        "messages": result["messages"],
        "summary": result.get("summary", state["summary"]),
        "retrieved_context": "",
        "retrieved_sources": [],
        "retrieved_context_chunks": [],
        "has_context": False,
    }
