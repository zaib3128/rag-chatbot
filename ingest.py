import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_pdf(pdf_path):

    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"Pages: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    print(f"Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("Vector DB created successfully!")


if __name__ == "__main__":
    ingest_pdf("data/sample.pdf")