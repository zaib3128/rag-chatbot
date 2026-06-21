import os
import gc
import stat
import uuid
import shutil
import tempfile

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
    )
    return embeddings, llm


embeddings, llm = load_models()

st.title("PDF RAG Chatbot")
st.caption("Upload any PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # If user uploaded a DIFFERENT file, clear everything from previous session
    if st.session_state.get("pdf_name") != uploaded_file.name:
        for key in ["db", "pdf_name"]:
            if key in st.session_state:
                del st.session_state[key]
        gc.collect()
        st.rerun()   # rerun immediately so old answer box disappears

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process PDF"):
        with st.spinner("Reading, chunking, and embedding..."):

            # Release old ChromaDB from memory completely
            if "db" in st.session_state:
                del st.session_state.db
                gc.collect()

            # Delete persisted folder if it exists (Windows fix included)
            if os.path.exists("./chroma_db"):
                def force_delete(func, path, _):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree("./chroma_db", onerror=force_delete)

            # Save uploaded file to disk temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            # Load → chunk
            pages = PyPDFLoader(pdf_path).load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            ).split_documents(pages)

            # THIS IS THE FIX: unique collection name every upload
            # Without this, Chroma reuses "langchain" collection and
            # accumulates chunks from all previous PDFs
            collection_name = f"pdf_{uuid.uuid4().hex[:8]}"

            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=collection_name,  # ← fresh collection each time
            )

            st.session_state.db = db
            st.session_state.pdf_name = uploaded_file.name

        st.success(f"Done! {len(chunks)} chunks stored. Ask your questions below.")


if "db" in st.session_state:
    st.divider()
    st.subheader(f"Asking about: {st.session_state.pdf_name}")

    question = st.text_input("Your question")

    if question:
        with st.spinner("Searching and thinking..."):
            docs = st.session_state.db.similarity_search(question, k=3)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""Answer ONLY using the context below.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}

Answer:"""

            answer = llm.invoke(prompt).content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Source chunks used to answer"):
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Chunk {i+1} — page {page}**")
                st.text(doc.page_content[:300] + "...")
