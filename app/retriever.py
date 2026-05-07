import os
import hashlib
from collections import defaultdict

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


def _document_hash_from_chunks(chunks: list) -> str:
    """Hash the full document content reconstructed from ordered chunks."""
    full_text = "\n".join(doc.page_content for doc in chunks)
    return hashlib.sha256(full_text.encode("utf-8")).hexdigest()


def _build_collection_name() -> str:
    """Use stable collection name across all documents."""
    return CHROMA_COLLECTION_PREFIX.replace(" ", "_").lower()


def _get_existing_index_state(vectorstore) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    """Return existing IDs and indexes by document-hash and source path."""
    try:
        all_docs = vectorstore.get(include=["metadatas"])
        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])

        by_doc_hash = defaultdict(list)
        by_source = defaultdict(list)

        for doc_id, metadata in zip(ids, metadatas):
            if not metadata:
                continue

            doc_hash = metadata.get("doc_hash")
            source = metadata.get("source", "")

            if doc_hash:
                by_doc_hash[doc_hash].append(doc_id)
            if source:
                by_source[source].append(doc_id)

        return set(ids), dict(by_doc_hash), dict(by_source)
    except Exception:
        return set(), {}, {}


def _group_by_source(split_docs) -> dict[str, list]:
    grouped = defaultdict(list)
    for doc in split_docs:
        source = doc.metadata.get("source", "")
        grouped[source].append(doc)
    return dict(grouped)


def _prepare_docs_for_upsert(vectorstore, split_docs, with_logging: bool = False):
    existing_ids, by_doc_hash, by_source = _get_existing_index_state(vectorstore)
    grouped_docs = _group_by_source(split_docs)

    docs_to_add = []
    ids_to_add = []
    ids_to_delete = []
    docs_added_set = set()
    docs_replaced_set = set()
    docs_skipped_set = set()

    for source, docs in grouped_docs.items():
        if not docs:
            continue

        filename = os.path.basename(source) if source else "Unknown source"
        doc_hash = _document_hash_from_chunks(docs)

        # Exact content already indexed (even under another file name/path).
        if doc_hash in by_doc_hash:
            if with_logging:
                docs_skipped_set.add(filename)
            continue

        # Same source path exists with different content; replace stale vectors.
        existing_source_ids = by_source.get(source, [])
        if existing_source_ids:
            ids_to_delete.extend(existing_source_ids)
            for existing_id in existing_source_ids:
                existing_ids.discard(existing_id)
            if with_logging:
                docs_replaced_set.add(filename)

        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        source_added = False

        for index, doc in enumerate(docs):
            content_hash = _document_content_hash(doc.page_content)
            doc.metadata["content_hash"] = content_hash
            doc.metadata["doc_hash"] = doc_hash

            doc_id = f"{source_hash}_{doc_hash[:16]}_{index}_{content_hash[:8]}"
            if doc_id in existing_ids:
                continue

            docs_to_add.append(doc)
            ids_to_add.append(doc_id)
            existing_ids.add(doc_id)
            source_added = True

        if with_logging and source_added:
            docs_added_set.add(filename)

    return (
        docs_to_add,
        ids_to_add,
        ids_to_delete,
        docs_added_set,
        docs_replaced_set,
        docs_skipped_set,
    )


def build_retriever(split_docs):
    """Build retriever with deduplication: only add new documents to ChromaDB."""
    embeddings = BoschEmbeddings()
    collection_name = _build_collection_name()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    docs_to_add, ids_to_add, ids_to_delete, _, _, _ = _prepare_docs_for_upsert(
        vectorstore,
        split_docs,
    )

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    if docs_to_add:
        vectorstore.add_documents(docs_to_add, ids=ids_to_add)

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )


def build_retriever_with_logging(split_docs) -> tuple:
    """Build retriever and return (retriever, added_docs, replaced_docs, skipped_docs).
    
    Note: Returns unique document filenames (deduped across chunks).
    """
    embeddings = BoschEmbeddings()
    collection_name = _build_collection_name()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    docs_to_add, ids_to_add, ids_to_delete, docs_added_set, docs_replaced_set, docs_skipped_set = _prepare_docs_for_upsert(
        vectorstore,
        split_docs,
        with_logging=True,
    )

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    if docs_to_add:
        vectorstore.add_documents(docs_to_add, ids=ids_to_add)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )

    # Convert sets to sorted lists for consistent ordering
    docs_added = sorted(list(docs_added_set))
    docs_replaced = sorted(list(docs_replaced_set))
    docs_skipped = sorted(list(docs_skipped_set))

    return retriever, docs_added, docs_replaced, docs_skipped


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
        all_data = vectorstore.get(include=["metadatas"])
        chunk_ids = all_data.get("ids", [])
        metadatas = all_data.get("metadatas", [])
        stats["total_chunks"] = len(chunk_ids)
        
        # Extract unique document names from metadata source paths.
        for metadata in metadatas:
            if not metadata:
                continue

            source = metadata.get("source", "")
            if source:
                stats["unique_docs"].add(os.path.basename(source))
        
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