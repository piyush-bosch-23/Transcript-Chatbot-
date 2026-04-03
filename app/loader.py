from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils import find_docx_file


def load_and_split_transcript():
    docx_file = find_docx_file()

    loader = Docx2txtLoader(docx_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    split_docs = splitter.split_documents(documents)
    return docx_file, split_docs