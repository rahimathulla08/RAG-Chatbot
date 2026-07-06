import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
DATA_FOLDER = "data"
VECTORSTORE_PATH = "vectorstore/faiss_index"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------
# Step 1: Load all PDFs
# ---------------------------------------------------
def load_documents(data_folder: str):
    documents = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_folder, file_name)
            print(f"Loading: {file_path}")

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Add source name into metadata
            for doc in docs:
                doc.metadata["source_file"] = file_name

            documents.extend(docs)

    return documents


# ---------------------------------------------------
# Step 2: Split documents into chunks
# ---------------------------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ---------------------------------------------------
# Step 3: Create embedding model
# ---------------------------------------------------
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings


# ---------------------------------------------------
# Step 4: Build FAISS vector store
# ---------------------------------------------------
def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------------------------------------------------
# Step 5: Main ingestion pipeline
# ---------------------------------------------------
def main():
    print("Step 1: Loading PDF documents...")
    documents = load_documents(DATA_FOLDER)
    print(f"Loaded {len(documents)} pages/documents")

    print("\nStep 2: Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nStep 3: Loading embedding model...")
    embeddings = get_embedding_model()

    print("\nStep 4: Creating FAISS vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)

    print(f"\nStep 5: Saving vector store to {VECTORSTORE_PATH} ...")
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("\nIngestion complete.")
    print("Your vector database is ready.")


if __name__ == "__main__":
    main()
