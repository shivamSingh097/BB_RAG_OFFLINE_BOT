import os
import streamlit as st

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from sentence_transformers import SentenceTransformer


# ===============================
# CONFIG
# ===============================
DATA_FOLDER = "BB_Data"
FAISS_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="BB Counsellor Bot", page_icon="🧠", layout="centered")


# ===============================
# LOAD DOCUMENTS
# ===============================
def load_documents(folder_path: str):
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


# ===============================
# SPLIT DOCUMENTS
# ===============================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ===============================
# EMBEDDINGS (cached safely)
# ===============================
@st.cache_resource
def load_embeddings():
    return SentenceTransformer(EMBEDDING_MODEL)


# ===============================
# VECTORSTORE (IMPORTANT FIX HERE)
# ===============================
@st.cache_resource
def build_vectorstore(_chunks):
    embeddings = load_embeddings()

    texts = [doc.page_content for doc in _chunks]
    metadatas = [doc.metadata for doc in _chunks]

    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )


# ===============================
# COUNSELLOR STYLE ANSWER
# ===============================
def counsellor_answer(query, docs):
    context = "\n\n".join([d.page_content for d in docs])

    return f"""
I understand your concern, and it's good that you're seeking clarity.

Based on the information available, here is a thoughtful and balanced perspective:

{context}

From a counsellor’s point of view, my suggestion would be to reflect calmly on this, consider your personal situation, and take a step-by-step approach rather than rushing into decisions.

If you'd like, you can ask follow-up questions or share more details so I can guide you better.
"""


# ===============================
# UI
# ===============================
st.title("🧠 BB Counsellor AI (Offline RAG)")
st.caption("Answers based strictly on your uploaded documents")

if not os.path.exists(DATA_FOLDER):
    st.error("❌ BB_Data folder not found")
    st.stop()

with st.spinner("📄 Loading documents..."):
    docs = load_documents(DATA_FOLDER)

if not docs:
    st.warning("No PDF or DOCX files found in BB_Data")
    st.stop()

with st.spinner("✂️ Preparing knowledge base..."):
    chunks = split_documents(docs)

with st.spinner("📦 Building vector database (first run may take time)..."):
    vectorstore = build_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

query = st.text_area("Ask your question", height=120)

if st.button("Get Counsellor Advice") and query.strip():
    with st.spinner("🤔 Thinking..."):
        relevant_docs = retriever.get_relevant_documents(query)
        answer = counsellor_answer(query, relevant_docs)

    st.markdown("### 🧠 Counsellor Response")
    st.write(answer)

    with st.expander("📚 Source Excerpts"):
        for i, d in enumerate(relevant_docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(d.page_content[:600] + "...")
