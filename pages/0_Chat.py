import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

PERSIST_DIRECTORY = "chroma_storage"
MY_COLLECTION = "my_books"
MY_MODEL = "mistral"

model = OllamaLLM(
    base_url="http://10.42.0.93:11434",
    model="mistral"
)

embedding_function = OllamaEmbeddings(
    model='nomic-embed-text',
    base_url="http://10.42.0.93:11434"  # Remote Ollama server
)

vectorstore = Chroma(
    collection_name=MY_COLLECTION,
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory=PERSIST_DIRECTORY,
)

retriever = vectorstore.as_retriever()
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model
    | StrOutputParser()
)

prompt = st.chat_input("Ask something")
if prompt:
    st.write(f"🧑 : {prompt}")
    st.write(f"🤖 : {after_rag_chain.invoke(prompt)}")