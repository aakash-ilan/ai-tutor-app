from langchain_community.vectorstores import Chroma
from langchain_ollama import embeddings
from typing import List, Tuple, Any, Optional
from langchain_core.documents import Document
from langchain.embeddings import OllamaEmbeddings

class ChromaDB:
    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.connection = None
        self.embedding_function = OllamaEmbeddings(
                                                    model='nomic-embed-text',
                                                    base_url="http://10.42.0.93:11434"  # Remote Ollama server
                                                  )

    def connect_vector_store(self) -> Chroma:
        return Chroma(
            collection_name= self.collection_name,
            embedding_function= self.embedding_function,
            persist_directory= self.db_path,
            collection_metadata= {"hnsw:space": "cosine"},
        )

    def add_documents(self, vector_store: Chroma, documents:List[Document]):
        vector_store.add_documents(documents = documents)
        vector_store.persist()


