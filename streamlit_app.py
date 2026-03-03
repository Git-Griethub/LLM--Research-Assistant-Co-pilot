import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Faculty Research Co-Pilot",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING (Institutional Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
h1 {
    color: #003366;
}
.sidebar .sidebar-content {
    background-color: #e6ecf2;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------
st.title("🎓 Faculty Research Co-Pilot")
st.markdown("AI-Powered Literature Review, Research Gap & Proposal Assistant")

st.markdown("---")


# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Literature Review",
     "Research Gap Analysis",
     "Proposal Draft",
     "Paper Summary"]
)

k_value = st.sidebar.slider("Number of Context Chunks", 2, 8, 4)


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "📂 Upload Research Papers (PDF)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    st.info("Processing uploaded documents...")

    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    st.success("✅ Documents processed successfully!")


    # -----------------------------
    # LOAD FREE LLM
    # -----------------------------
    @st.cache_resource
    def load_model():
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        return generator

    generator = load_model()


    # -----------------------------
    # USER QUERY INPUT
    # -----------------------------
    user_query = st.text_area("🔎 Enter Your Research Query")

    if st.button("Generate Response"):

        retrieved_docs = vectorstore.similarity_search(user_query, k=k_value)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        if mode == "Literature Review":
            instruction = """
Generate a structured literature review including:
1. Introduction
2. Key Studies
3. Comparative Analysis
4. Research Gaps
5. Conclusion
"""

        elif mode == "Research Gap Analysis":
            instruction = """
Identify:
• Methodological gaps
• Theoretical gaps
• Dataset limitations
• Future research opportunities
"""

        elif mode == "Proposal Draft":
            instruction = """
Draft a research proposal including:
1. Title
2. Abstract
3. Problem Statement
4. Objectives
5. Methodology
6. Expected Outcomes
7. Timeline
"""

        else:
            instruction = "Provide a structured academic summary."


        prompt = f"""
You are a senior academic researcher.

Context:
{context}

Task:
{instruction}

User Question:
{user_query}
"""

        with st.spinner("Generating response..."):

            response = generator(
                prompt,
                max_new_tokens=500,
                temperature=0.3
            )

        st.markdown("### 📘 AI Response")
        st.write(response[0]["generated_text"])


st.markdown("---")
st.markdown("© 2026 Faculty Research Co-Pilot | Powered by LLM")
