from langchain_community.document_loaders import JSONLoader, UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, Docx2txtLoader
from langchain_core.documents import Document

from typing import List, Tuple, Any, Optional
from chroma_db import ChromaDB
from pathlib import Path

PERSIST_DIRECTORY = "chroma_storage"
MY_COLLECTION = "my_books"

def get_file_extention(file_path: Path) -> str:
    return (file_path.suffix).strip(".")

def process_selected_file(files:list[Path]):
    document_details = list(map(lambda x: (x, get_file_extention(x)), files))
    return document_details

def load_documents(document_name: Path, document_type: str) -> List[Document]:
    match document_type:
        case "docx":

            loader = UnstructuredWordDocumentLoader(file_path=document_name)
            return loader.load()
        case "doc":
            loader = UnstructuredWordDocumentLoader(file_path=document_name)
            return loader.load()
        case "pdf":
            loader = PyPDFLoader(file_path=document_name)
            return loader.load()
        case "json":
            loader = JSONLoader(file_path= document_name, jq_schema=".[]", text_content=False)
            return loader.load()
        case _:
            print(f"Unsupported document {document_type} found in the list..")

def ingest(files:list[Path]):
    vec = ChromaDB(PERSIST_DIRECTORY, MY_COLLECTION)
    vector_store = vec.connect_vector_store()
    for (document_name, document_type) in process_selected_file(files):
        try:
            documents = load_documents(document_name, document_type)
            vec.add_documents(vector_store, documents)
        except Exception as e:
            error_message = f"Error {e} while processing the file {document_name}"
            print(error_message)
