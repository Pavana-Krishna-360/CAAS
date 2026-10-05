import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import db


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CORPUS_PDFS = [
    "McGrawHill - Machine Learning -Tom Mitchell.pdf",
    "Deep Learning.pdf",
    "artificial intelligence.pdf"
]

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "machine_learning_book"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BUDGET = 100.0
PRICE_MIN = 1.0
PRICE_MAX = 5.0

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# EMBEDDINGS & VECTOR STORE HELPERS
# ============================================================

def create_embeddings():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embedding model loaded.")
    return embeddings


def get_vector_store(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


# ============================================================
# CHUNKING & EXTRACTION
# ============================================================

def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    print(f"Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}.")
    return documents


def split_documents(documents, document_id, pdf_path):
    print(f"Splitting document '{document_id}' into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = db.generate_chunk_id(document_id, index)
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["document_id"] = document_id
        chunk.metadata["chunk_index"] = index
        chunk.metadata["source"] = os.path.basename(pdf_path)
        chunk.metadata["price"] = 1.0
        chunk.metadata["royalty"] = 1.0

    print(f"Created {len(chunks)} chunks for {document_id}.")
    return chunks


# ============================================================
# INCREMENTAL DOCUMENT INGESTION
# ============================================================

def ingest_pdf(pdf_path, vector_store=None, embeddings=None, force_reset=False):
    """
    Ingests a single PDF non-destructively:
    - If document is already ingested (status == 'EXISTING') and not force_reset:
      Preserves all existing chunks, historical usage, and frequencies.
    - If force_reset:
      Deletes old SQLite records and Chroma vectors for this document only,
      re-reads PDF, re-chunks, re-embeds, and resets frequency = 0.
    """
    if not os.path.exists(pdf_path):
        print(f"[WARNING] Skipping missing PDF file: {pdf_path}")
        return False, vector_store, embeddings

    doc_id = db.get_canonical_document_id(pdf_path)
    file_hash = db.compute_file_hash(pdf_path)
    status = db.check_document_status(doc_id, pdf_path)

    if not force_reset and status == "EXISTING":
        doc_record = db.get_document(doc_id)
        total_chunks = doc_record.get("total_chunks", 0) if doc_record else 0
        print(f"\n[PRESERVED] Document '{doc_id}' is already ingested and up-to-date.")
        print(f"            Preserving all {total_chunks} existing chunks, historical frequency, and prices.")
        return False, vector_store, embeddings

    # Initialize embedding & vector store on demand
    if embeddings is None:
        embeddings = create_embeddings()
    if vector_store is None:
        vector_store = get_vector_store(embeddings)

    if force_reset:
        print(f"\n[RESETTING] Explicit reset requested for document '{doc_id}'...")
        # 1. Delete Chroma vectors for this document by chunk_id
        old_chunks = db.get_chunks_by_document(doc_id)
        if old_chunks:
            old_ids = [c["chunk_id"] for c in old_chunks]
            try:
                vector_store._collection.delete(ids=old_ids)
                print(f"Deleted {len(old_ids)} existing Chroma vectors for '{doc_id}'.")
            except Exception as e:
                print(f"Note on Chroma vector cleanup: {e}")

        # 2. Delete SQLite records for this document
        deleted_count = db.delete_chunks_by_document(doc_id)
        db.delete_document(doc_id)
        print(f"Deleted {deleted_count} SQLite chunk records and document entry for '{doc_id}'.")

    print(f"\n[INGESTING] Processing document '{doc_id}' (Status: {'RESET' if force_reset else status})...")

    # 1. Load & chunk
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents, doc_id, pdf_path)

    # 2. Store in ChromaDB with deterministic IDs
    print(f"Storing {len(chunks)} chunks in ChromaDB for '{doc_id}'...")
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    # 3. Register chunks in SQLite (resets frequency = 0 on fresh/reset ingestion)
    if force_reset:
        # Explicit reset resets frequency to 0
        conn = db.get_db_connection()
        cursor = conn.cursor()
        for index, chunk in enumerate(chunks, start=1):
            cid = chunk.metadata["chunk_id"]
            text_hash = db.compute_text_hash(chunk.page_content)
            cursor.execute("""
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, text_hash, chunk_text, source, page,
                    raw_uniqueness, normalized_uniqueness,
                    raw_information_density, normalized_information_density,
                    frequency, log_frequency, normalized_frequency,
                    royalty, price, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 1.0, 1.0, CURRENT_TIMESTAMP)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    chunk_index = excluded.chunk_index,
                    text_hash = excluded.text_hash,
                    chunk_text = excluded.chunk_text,
                    source = excluded.source,
                    page = excluded.page,
                    frequency = 0,
                    log_frequency = 0.0,
                    normalized_frequency = 0.0,
                    price = 1.0,
                    updated_at = CURRENT_TIMESTAMP
            """, (cid, doc_id, index, text_hash, chunk.page_content, os.path.basename(pdf_path), int(chunk.metadata.get("page", 0))))
        conn.commit()
        conn.close()
    else:
        db.register_chunks(chunks, document_id=doc_id, default_royalty=1.0)

    # 4. Register document record in SQLite
    db.register_document(doc_id, os.path.basename(pdf_path), file_hash, len(chunks))

    print(f"[SUCCESS] Document '{doc_id}' successfully ingested ({len(chunks)} chunks).")
    return True, vector_store, embeddings


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LB-CaaS / DCAAS Incremental Document Ingestion & Reset")
    parser.add_argument(
        "--pdfs",
        nargs="*",
        default=DEFAULT_CORPUS_PDFS,
        help="List of PDF file paths to ingest (defaults to the 3-book corpus)"
    )
    parser.add_argument(
        "--reset",
        nargs="*",
        default=None,
        help="List of document IDs or PDF file names to explicitly reset and re-ingest"
    )
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help="Explicitly reset all corpus documents (development/testing only)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DCAAS - DOCUMENT INGESTION & RESET")
    print("=" * 60)

    db.init_db()

    # Determine reset target document IDs
    reset_doc_ids = set()
    if args.full_reset:
        print("[NOTICE] Full reset mode enabled. All corpus documents will be reset.")
        for pdf_path in args.pdfs:
            reset_doc_ids.add(db.get_canonical_document_id(pdf_path))
    elif args.reset:
        for item in args.reset:
            canonical_id = db.get_canonical_document_id(item)
            reset_doc_ids.add(canonical_id)
        print(f"[NOTICE] Selective reset requested for documents: {list(reset_doc_ids)}")

    embeddings = None
    vector_store = None
    any_new_ingestion = False

    for pdf_path in args.pdfs:
        doc_id = db.get_canonical_document_id(pdf_path)
        should_reset = doc_id in reset_doc_ids

        newly_ingested, vector_store, embeddings = ingest_pdf(
            pdf_path,
            vector_store=vector_store,
            embeddings=embeddings,
            force_reset=should_reset
        )
        if newly_ingested:
            any_new_ingestion = True

    # If any document was ingested or reset, or if chunks lack information density, compute metrics
    conn = db.get_db_connection()
    missing_nid = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE raw_information_density IS NULL OR raw_information_density = 0.0"
    ).fetchone()[0]
    conn.close()

    if any_new_ingestion or missing_nid > 0:
        if any_new_ingestion:
            print("\nRecalculating corpus uniqueness and pricing normalizations...")
            raw_uniqueness_map = db.compute_raw_uniqueness(
                k=5,
                chroma_dir=CHROMA_DIR,
                collection_name=COLLECTION_NAME
            )
            db.update_raw_uniqueness_in_db(raw_uniqueness_map)
            db.normalize_uniqueness_in_db()

        if missing_nid > 0 or any_new_ingestion:
            print(f"\nCalculating SLM information density (Qwen2.5-0.5B token surprisal) for {missing_nid} uncomputed chunks...")
            density_map = db.compute_all_information_densities(only_uncomputed=True)
            if density_map:
                db.update_raw_information_density_in_db(density_map)
            db.normalize_information_density_in_db()

        db.normalize_frequency_in_db()
        db.update_all_prices_in_db()
        print("Global metrics and dynamic prices updated across all corpus chunks.")
    else:
        print("\nAll documents and metrics are up-to-date. No re-indexing or metric recalculation required.")

    # Print summary of all documents currently registered
    all_docs = db.get_all_documents()
    print("\n" + "=" * 60)
    print("CURRENT REGISTERED CORPUS IN SQLITE")
    print("=" * 60)
    for doc in all_docs:
        print(f"- {doc['document_id']}: {doc['file_name']} ({doc['total_chunks']} chunks)")
    print("=" * 60)
    print("INGESTION WORKFLOW FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
