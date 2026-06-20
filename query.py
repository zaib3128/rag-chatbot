from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

def ask(question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question: {question}

Answer clearly:
"""

    llm = Ollama(model="mistral")

    return llm.invoke(prompt)


if __name__ == "__main__":

    while True:
        q = input("\nAsk: ")

        if q.lower() == "exit":
            break

        print("\n", ask(q))