import os
import gc
import stat
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()


@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Groq is free and 10x faster than Ollama
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    return embeddings, llm


embeddings, llm = load_models()

st.title("PDF RAG Chatbot")
st.caption("Upload any PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Process PDF"):
        with st.spinner("Reading, chunking, and embedding..."):

            # Windows fix: release ChromaDB file lock before deleting
            if "db" in st.session_state:
                del st.session_state.db

            if os.path.exists("./chroma_db"):
                import stat
                import gc
                gc.collect()  # force Python to release file handles

                # onerror handler forces Windows to release read-only locks
                def force_delete(func, path, _):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                shutil.rmtree("./chroma_db", onerror=force_delete)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            pages = PyPDFLoader(pdf_path).load()

            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            ).split_documents(pages)

            db = Chroma.from_documents(
                chunks,
                embeddings,
                persist_directory="./chroma_db",
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

            answer = llm.invoke(prompt).content  # .content extracts text from Groq response

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Source chunks used to answer"):
            for i, doc in enumerate(docs):
                page = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Chunk {i+1} — page {page}**")
                st.text(doc.page_content[:300] + "...")