import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="BB Offline RAG Counsellor", layout="wide")
st.title("🧠 BB Offline RAG Counsellor Bot")

DATA_FOLDER = "BB_Data"

# ---------------------------
# Embedding Model (Stable)
# ---------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(texts):
    model = load_embedding_model()
    return model.encode(texts, show_progress_bar=False)

# ---------------------------
# Load Documents
# ---------------------------
def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

    return documents

# ---------------------------
# Chunking
# ---------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

# ---------------------------
# Vector Store
# ---------------------------
@st.cache_resource
def build_vectorstore(chunks):
    texts = [c.page_content for c in chunks]
    vectors = embed_texts(texts)

    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=None
    )

# ---------------------------
# Counsellor Prompt
# ---------------------------
def counsellor_prompt(context, question):
    return f"""
You are a professional counsellor.
Answer politely, clearly, and professionally.
Use simple English or Hinglish if helpful.
Be accurate and grounded in the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# ---------------------------
# Main Logic
# ---------------------------
if not os.path.exists(DATA_FOLDER):
    st.error("❌ BB_Data folder not found")
    st.stop()

with st.spinner("📄 Loading documents..."):
    docs = load_documents(DATA_FOLDER)

if not docs:
    st.warning("No PDF or DOCX files found in BB_Data")
    st.stop()

with st.spinner("✂️ Splitting documents..."):
    chunks = split_documents(docs)

with st.spinner("📦 Building vector database (first run takes time)..."):
    vectorstore = build_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------------------------
# Chat UI
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask your question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            docs = retriever.get_relevant_documents(user_question)
            context = "\n\n".join([d.page_content for d in docs])

            answer = counsellor_prompt(context, user_question)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
