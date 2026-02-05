# ======================================================
# STREAMLIT-ONLY LOCAL RAG CHATBOT (FOLDER-BASED INGEST)
# Reads ALL PDF & Word files from BB_Data folder
# Acts like a professional counsellor in answers
# ======================================================

# ------------------------------------------------------
# PROJECT STRUCTURE
# ------------------------------------------------------
# project/
# ├── app.py
# ├── requirements.txt
# └── BB_Data/
#     ├── file1.pdf
#     ├── file2.docx
#     └── ...

# ------------------------------------------------------
# app.py
# ------------------------------------------------------

import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATA_FOLDER = "BB_Data"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 200

st.set_page_config(page_title="BB Counsellor RAG Bot", layout="wide")

st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}
.chat-bot {
    background-color: #F3F4F6;
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 BB Professional Counsellor Bot")
st.caption("Reads all documents from BB_Data | English + Hinglish | Fully Offline")

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ------------------------------------------------------
# LOAD MODELS (CACHED)
# ------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.25
    )

embeddings = load_embeddings()
llm = load_llm()

# ------------------------------------------------------
# DOCUMENT INGESTION FROM FOLDER
# ------------------------------------------------------
def load_documents_from_folder(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())
        elif file.lower().endswith(".docx"):
            documents.extend(Docx2txtLoader(path).load())
    return documents

@st.cache_resource
def build_vector_db():
    docs = load_documents_from_folder(DATA_FOLDER)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# ------------------------------------------------------
# RAG QUESTION ANSWERING (COUNSELLOR STYLE)
# ------------------------------------------------------
def ask_rag(question):
    docs = st.session_state.vector_db.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a senior professional counsellor and advisor.
Your tone should be calm, empathetic, clear, and professional.

Rules:
- Use ONLY the information from the context below
- Reframe the answer in a structured and professional way
- Be accurate and factual
- Do NOT add assumptions
- If the answer is not available, clearly say so

Context:
{context}

User Question:
{question}

Professional Counsellor Answer:
"""

    response = llm(prompt)[0]["generated_text"]
    return response.split("Professional Counsellor Answer:")[-1].strip()

# ------------------------------------------------------
# SIDEBAR ACTION
# ------------------------------------------------------
st.sidebar.header("📂 Knowledge Base")

if st.sidebar.button("🔄 Load / Refresh Documents"):
    with st.spinner("Reading documents and building knowledge base..."):
        st.session_state.vector_db = build_vector_db()
        st.session_state.chat = []
    st.sidebar.success("Documents loaded successfully")

# ------------------------------------------------------
# CHAT UI
# ------------------------------------------------------
query = st.chat_input("Ask your question...")

if query and st.session_state.vector_db:
    st.session_state.chat.append(("user", query))
    answer = ask_rag(query)
    st.session_state.chat.append(("bot", answer))

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-user'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'><b>Counsellor:</b> {msg}</div>", unsafe_allow_html=True)

if not st.session_state.vector_db:
    st.info("⬅ Click 'Load / Refresh Documents' to initialize knowledge base")

# ------------------------------------------------------
# END OF app.py
# ------------------------------------------------------

# ------------------------------------------------------
# requirements.txt  (For Streamlit Hosting)
# ------------------------------------------------------
# streamlit
# langchain
# faiss-cpu
# sentence-transformers
# transformers
# torch
# accelerate
# pypdf
# python-docx
