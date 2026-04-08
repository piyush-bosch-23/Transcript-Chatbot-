import os
import hashlib

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from app.embedding_client import get_embedding, get_embeddings
from app.config import (
    RETRIEVER_TOP_K,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_PREFIX,
)


class BoschEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)


def _document_content_hash(doc_content: str) -> str:
    """Generate a short hash from document content for deduplication."""
    return hashlib.sha256(doc_content.encode("utf-8")).hexdigest()[:16]


def _build_collection_name() -> str:
    """Use stable collection name across all documents."""
    return CHROMA_COLLECTION_PREFIX.replace(" ", "_").lower()


def _get_existing_doc_hashes(vectorstore) -> set[str]:
    """Retrieve set of existing document IDs from ChromaDB collection."""
    try:
        all_docs = vectorstore.get(include=[])
        doc_ids = all_docs.get("ids", [])
        return set(doc_ids)
    except Exception:
        return set()


def build_retriever(split_docs):
    """Build retriever with deduplication: only add new documents to ChromaDB."""
    embeddings = BoschEmbeddings()
    collection_name = _build_collection_name()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    existing_hashes = _get_existing_doc_hashes(vectorstore)
    docs_to_add = []

    for doc in split_docs:
        source = doc.metadata.get("source", "")
        content_hash = _document_content_hash(doc.page_content)
        doc_id = f"{os.path.basename(source)}_{content_hash}"

        if doc_id not in existing_hashes:
            doc.metadata["content_hash"] = content_hash
            docs_to_add.append(doc)

    if docs_to_add:
        vectorstore.add_documents(
            docs_to_add,
            ids=[f"{os.path.basename(doc.metadata.get('source', ''))}_{doc.metadata.get('content_hash')}"
                  for doc in docs_to_add]
        )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )


def build_retriever_with_logging(split_docs) -> tuple:
    """Build retriever and return (retriever, added_docs_list, skipped_docs_list) for logging.
    
    Note: Returns unique document filenames (deduped across chunks).
    """
    embeddings = BoschEmbeddings()
    collection_name = _build_collection_name()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    existing_hashes = _get_existing_doc_hashes(vectorstore)
    docs_to_add = []
    docs_added_set = set()
    docs_skipped_set = set()

    for doc in split_docs:
        source = doc.metadata.get("source", "")
        content_hash = _document_content_hash(doc.page_content)
        doc_id = f"{os.path.basename(source)}_{content_hash}"
        filename = os.path.basename(source)

        if doc_id in existing_hashes:
            docs_skipped_set.add(filename)
        else:
            doc.metadata["content_hash"] = content_hash
            docs_to_add.append(doc)
            docs_added_set.add(filename)

    if docs_to_add:
        vectorstore.add_documents(
            docs_to_add,
            ids=[f"{os.path.basename(doc.metadata.get('source', ''))}_{doc.metadata.get('content_hash')}"
                  for doc in docs_to_add]
        )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )

    # Convert sets to sorted lists for consistent ordering
    docs_added = sorted(list(docs_added_set))
    docs_skipped = sorted(list(docs_skipped_set))

    return retriever, docs_added, docs_skipped


def _get_source_name(doc) -> str:
    source_path = doc.metadata.get("source", "")
    if not source_path:
        return "Unknown source"
    return os.path.basename(source_path)


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.split())


def _is_low_value_chunk(text: str) -> bool:
    normalized_text = _normalize_chunk_text(text)
    if not normalized_text:
        return True

    word_count = len(normalized_text.split())
    sentence_markers = sum(normalized_text.count(marker) for marker in ".?!")

    # Skip short title-like chunks that often rank highly but provide little usable context.
    return word_count < 18 and sentence_markers == 0


def get_relevant_context(retriever, query: str) -> tuple[str, bool, list[str], list[dict[str, str]]]:
    docs = retriever.invoke(query)

    filtered_docs = []
    fallback_docs = []
    source_names = []

    for doc in docs:
        text = _normalize_chunk_text(doc.page_content)
        if not text or len(text) <= 40:
            continue

        source_name = _get_source_name(doc)
        fallback_docs.append((text, source_name))

        if _is_low_value_chunk(text):
            continue

        filtered_docs.append((text, source_name))

    chosen_docs = filtered_docs or fallback_docs

    useful_docs = []
    context_chunks = []
    seen_chunks = set()
    for text, source_name in chosen_docs:
        if text in seen_chunks:
            continue

        useful_docs.append(text)
        context_chunks.append({"source": source_name, "text": text})
        seen_chunks.add(text)

        if source_name not in source_names:
            source_names.append(source_name)

        if len(useful_docs) >= RETRIEVER_TOP_K:
            break

    context = "\n\n---\n\n".join(useful_docs[:RETRIEVER_TOP_K])
    has_context = bool(useful_docs)

    return context, has_context, source_names[:RETRIEVER_TOP_K], context_chunks[:RETRIEVER_TOP_K]


def get_chroma_stats() -> dict:
    """Get ChromaDB statistics including document count and disk usage."""
    stats = {
        "total_chunks": 0,
        "unique_docs": set(),
        "disk_usage_mb": 0.0,
    }
    
    try:
        embeddings = BoschEmbeddings()
        collection_name = _build_collection_name()
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        # Get all documents in the collection
        all_data = vectorstore.get(include=[])
        chunk_ids = all_data.get("ids", [])
        stats["total_chunks"] = len(chunk_ids)
        
        # Extract unique document names from chunk IDs (format: "filename_hash")
        for chunk_id in chunk_ids:
            parts = chunk_id.rsplit("_", 1)  # Split on last underscore (hash separator)
            if parts:
                doc_name = parts[0]
                stats["unique_docs"].add(doc_name)
        
        # Calculate disk usage of .chroma directory
        if os.path.isdir(CHROMA_PERSIST_DIR):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(CHROMA_PERSIST_DIR):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            stats["disk_usage_mb"] = round(total_size / (1024 * 1024), 2)
    
    except Exception as e:
        print(f"Error getting ChromaDB stats: {e}")
    
    # Convert set to count for final output
    stats["unique_docs_count"] = len(stats["unique_docs"])
    del stats["unique_docs"]  # Don't return the set itself
    
    return stats