from __future__ import annotations

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TRANSCRIPTS_FOLDER
from app.utils import find_docx_files


def load_and_split_transcript(docx_files: list[str] | None = None):
    docx_files = docx_files or find_docx_files(TRANSCRIPTS_FOLDER)
    documents = []

    for docx_file in docx_files:
        loader = Docx2txtLoader(docx_file)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = splitter.split_documents(documents)
    return docx_files, split_docs