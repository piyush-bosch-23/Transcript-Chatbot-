from __future__ import annotations

import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, TRANSCRIPTS_FOLDER
from app.utils import find_transcript_files


def load_and_split_transcript(document_files: list[str] | None = None):
    document_files = document_files or find_transcript_files(TRANSCRIPTS_FOLDER)
    documents = []

    for file_path in document_files:
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            continue

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = splitter.split_documents(documents)
    return document_files, split_docs