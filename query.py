import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
VECTORSTORE_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


# ---------------------------------------------------
# Embedding model
# ---------------------------------------------------
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ---------------------------------------------------
# Load vectorstore
# ---------------------------------------------------
def load_vectorstore():
    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


# ---------------------------------------------------
# Retriever
# ---------------------------------------------------
def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


# ---------------------------------------------------
# LLM
# ---------------------------------------------------
def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ---------------------------------------------------
# Format context
# ---------------------------------------------------
def format_context(docs):
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "Unknown Source")
        page = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"""
[Source {i}]
File: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------
# Prompt
# ---------------------------------------------------
def build_prompt(question, context):
    return f"""
You are a professional academic assistant chatbot for college documents.

Your job is to answer student questions ONLY from the provided document context.

Rules:
1. Use only the provided context.
2. If the answer is not available in the context, say:
   "I could not find the answer in the provided documents."
3. Do not make up facts.
4. Keep the answer clear, concise, and student-friendly.
5. If possible, summarize the answer in 2-5 lines.

-------------------------
CONTEXT:
{context}
-------------------------

QUESTION:
{question}

ANSWER:
"""


# ---------------------------------------------------
# Main RAG function
# ---------------------------------------------------
def ask_question(question, retriever, llm):
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)
    prompt = build_prompt(question, context)
    response = llm.invoke(prompt)

    return response.content, retrieved_docs


# ---------------------------------------------------
# CLI mode for testing
# ---------------------------------------------------
if __name__ == "__main__":
    print("Loading vectorstore and LLM...")
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    print("College RAG Assistant is ready!")
    print("Type your question below.\n")

    while True:
        question = input("Ask a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer, docs = ask_question(question, retriever, llm)

        print("\n================ ANSWER ================\n")
        print(answer)

        print("\n============= SOURCES USED =============\n")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_file", "Unknown Source")
            page = doc.metadata.get("page", "N/A")

            print(f"Source {i}: {source} | Page: {page}")
            print(doc.page_content[:300])
            print("-" * 60)
