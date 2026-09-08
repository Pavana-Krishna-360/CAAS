# import os

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PDF_PATH = "McGrawHill - Machine Learning -Tom Mitchell.pdf"

# CHROMA_DIR = "./chroma_db"

# COLLECTION_NAME = "machine_learning_book"

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# # Chunking parameters
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200


# # ============================================================
# # LOAD PDF
# # ============================================================

# def load_pdf():

#     if not os.path.exists(PDF_PATH):
#         raise FileNotFoundError(
#             f"PDF file not found: {PDF_PATH}"
#         )

#     print("Loading PDF...")

#     loader = PyPDFLoader(PDF_PATH)

#     documents = loader.load()

#     print(
#         f"Loaded {len(documents)} pages."
#     )

#     return documents


# # ============================================================
# # SPLIT DOCUMENTS
# # ============================================================

# def split_documents(documents):

#     print("Splitting documents into chunks...")

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=[
#             "\n\n",
#             "\n",
#             ". ",
#             " ",
#             ""
#         ]
#     )

#     chunks = text_splitter.split_documents(
#         documents
#     )

#     print(
#         f"Created {len(chunks)} chunks."
#     )

#     return chunks


# # ============================================================
# # CREATE EMBEDDINGS
# # ============================================================

# def create_embeddings():

#     print(
#         f"Loading embedding model: "
#         f"{EMBEDDING_MODEL}"
#     )

#     embeddings = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL,
#         model_kwargs={
#             "device": "cpu"
#         },
#         encode_kwargs={
#             "normalize_embeddings": True
#         }
#     )

#     print("Embedding model loaded.")

#     return embeddings


# # ============================================================
# # STORE IN CHROMADB
# # ============================================================

# def store_in_chromadb(
#     chunks,
#     embeddings
# ):

#     print("Creating/updating ChromaDB...")

#     vector_store = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         collection_name=COLLECTION_NAME,
#         persist_directory=CHROMA_DIR
#     )

#     print(
#         "Documents successfully stored "
#         "in ChromaDB."
#     )

#     print(
#         f"Total chunks in database: "
#         f"{vector_store._collection.count()}"
#     )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 60)
#     print("NAIVE RAG - DOCUMENT INGESTION")
#     print("=" * 60)

#     # 1. Load PDF
#     documents = load_pdf()

#     # 2. Split into chunks
#     chunks = split_documents(
#         documents
#     )

#     # 3. Create embeddings
#     embeddings = create_embeddings()

#     # 4. Store in ChromaDB
#     store_in_chromadb(
#         chunks,
#         embeddings
#     )

#     print("\n" + "=" * 60)
#     print("INGESTION COMPLETED SUCCESSFULLY")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()

# ingest.py

import os
import random

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


# ============================================================
# LB-CaaS CONFIGURATION
# ============================================================

# Static user budget
BUDGET = 100.0

# Static random chunk pricing
PRICE_MIN = 1.0
PRICE_MAX = 5.0

# Fixed seed ensures prices remain static
RANDOM_SEED = 42


# ============================================================
# CHUNKING PARAMETERS
# ============================================================

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
# ASSIGN STATIC RANDOM PRICES
# ============================================================

def assign_chunk_prices(chunks):

    print(
        "Assigning static random prices to chunks..."
    )

    random_generator = random.Random(
        RANDOM_SEED
    )

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        # Random price between ₹1 and ₹5
        # rounded to 2 decimal places
        price = round(
            random_generator.uniform(
                PRICE_MIN,
                PRICE_MAX
            ),
            2
        )

        # Store price in Chroma metadata
        chunk.metadata["price"] = price

        # Store chunk ID for identification
        chunk.metadata["chunk_id"] = index

    print(
        f"Prices assigned to {len(chunks)} chunks."
    )

    print(
        f"Price range: ₹{PRICE_MIN:.2f} - ₹{PRICE_MAX:.2f}"
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
    print("LB-CaaS - DOCUMENT INGESTION")
    print("=" * 60)

    print(
        f"\nAllocated Budget: ₹{BUDGET:.2f}"
    )

    print(
        f"Chunk Price Range: "
        f"₹{PRICE_MIN:.2f} - ₹{PRICE_MAX:.2f}"
    )

    # 1. Load PDF
    documents = load_pdf()

    # 2. Split documents
    chunks = split_documents(
        documents
    )

    # 3. Assign static prices
    chunks = assign_chunk_prices(
        chunks
    )

    # 4. Create embeddings
    embeddings = create_embeddings()

    # 5. Store in ChromaDB
    store_in_chromadb(
        chunks,
        embeddings
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
