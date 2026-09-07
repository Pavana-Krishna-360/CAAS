import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "McGrawHill - Machine Learning -Tom Mitchell.pdf"

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "machine_learning_book"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf():

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    print("Loading PDF...")

    loader = PyPDFLoader(PDF_PATH)

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages."
    )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Embedding model loaded.")

    return embeddings


# ============================================================
# STORE IN CHROMADB
# ============================================================

def store_in_chromadb(
    chunks,
    embeddings
):

    print("Creating/updating ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )

    print(
        "Documents successfully stored "
        "in ChromaDB."
    )

    print(
        f"Total chunks in database: "
        f"{vector_store._collection.count()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NAIVE RAG - DOCUMENT INGESTION")
    print("=" * 60)

    # 1. Load PDF
    documents = load_pdf()

    # 2. Split into chunks
    chunks = split_documents(
        documents
    )

    # 3. Create embeddings
    embeddings = create_embeddings()

    # 4. Store in ChromaDB
    store_in_chromadb(
        chunks,
        embeddings
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()