import os
import streamlit as st

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

st.set_page_config(
    page_title="BB Counsellor AI",
    page_icon="🧠",
    layout="centered"
)


# ===============================
# EMBEDDINGS (SAFE TO CACHE)
# ===============================
@st.cache_resource
def load_embeddings():
    return SentenceTransformer(EMBEDDING_MODEL)


# ===============================
# LOAD DOCUMENTS (NO CACHE)
# ===============================
def load_documents(folder):
    documents = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())

        elif file.lower().endswith(".docx"):
            documents.extend(Docx2txtLoader(path).load())

    return documents


# ===============================
# SPLIT DOCUMENTS (NO CACHE)
# ===============================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ===============================
# VECTORSTORE (DISK BASED – NO HASHING)
# ===============================
def get_vectorstore(chunks):
    embeddings = load_embeddings()

    if os.path.exists(FAISS_DIR):
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectors = embeddings.encode(texts, show_progress_bar=False)

    vectorstore = FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(FAISS_DIR)
    return vectorstore


# ===============================
# COUNSELLOR ANSWER STYLE
# ===============================
def counsellor_answer(query, docs):
    context = "\n\n".join(d.page_content for d in docs)

    return f"""
I understand your concern, and it's good that you're seeking clarity.

Based on the information available in your documents, here is a thoughtful perspective:

{context}

From a counsellor’s point of view, I would suggest reflecting calmly on this information and applying it step by step to your situation.

If you'd like, feel free to ask a follow-up question.
"""


# ===============================
# UI
# ===============================
st.title("🧠 BB Counsellor AI (Offline RAG)")
st.caption("Answers strictly from your documents")

if not os.path.exists(DATA_FOLDER):
    st.error("❌ BB_Data folder not found")
    st.stop()

with st.spinner("📄 Loading documents..."):
    docs = load_documents(DATA_FOLDER)

if not docs:
    st.warning("No PDF or DOCX files found in BB_Data")
    st.stop()

with st.spinner("✂️ Processing knowledge base..."):
    chunks = split_documents(docs)

with st.spinner("📦 Preparing vector database..."):
    vectorstore = get_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

query = st.text_area("Ask your question", height=120)

if st.button("Get Counsellor Advice") and query.strip():
    with st.spinner("🤔 Thinking..."):
        results = retriever.get_relevant_documents(query)
        answer = counsellor_answer(query, results)

    st.markdown("### 🧠 Counsellor Response")
    st.write(answer)

    with st.expander("📚 Source Context"):
        for i, d in enumerate(results, 1):
            st.markdown(f"**Source {i}**")
            st.write(d.page_content[:600] + "...")
