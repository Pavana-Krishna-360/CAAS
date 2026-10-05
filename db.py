import sqlite3
import os
import math
import hashlib

DB_PATH = os.path.join(os.path.dirname(__file__), "caas.db")


def get_db_connection(db_path=DB_PATH):
    """
    Returns a connection to the SQLite database with row_factory set to sqlite3.Row.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DB_PATH):
    """
    Initializes the SQLite database with documents and chunks tables,
    performing non-destructive schema migrations if existing tables lack new columns.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_hash TEXT,
            total_chunks INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 2. Chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_index INTEGER,
            text_hash TEXT,
            chunk_text TEXT,
            source TEXT,
            page INTEGER,
            raw_uniqueness REAL DEFAULT 0.0,
            normalized_uniqueness REAL DEFAULT 0.0,
            raw_information_density REAL DEFAULT 0.0,
            normalized_information_density REAL DEFAULT 0.0,
            frequency INTEGER DEFAULT 0,
            log_frequency REAL DEFAULT 0.0,
            normalized_frequency REAL DEFAULT 0.0,
            royalty REAL DEFAULT 1.0,
            price REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
    """)

    # 3. Schema migration: Add missing columns if upgrading an existing chunks table
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(chunks)").fetchall()]
    migration_columns = {
        "chunk_index": "INTEGER DEFAULT 0",
        "text_hash": "TEXT",
        "raw_information_density": "REAL DEFAULT 0.0",
        "normalized_information_density": "REAL DEFAULT 0.0"
    }

    for col_name, col_type in migration_columns.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()
    print(f"Database initialized and verified at: {db_path}")


# ============================================================
# DOCUMENT CRUD OPERATIONS
# ============================================================

def register_document(
    document_id: str,
    file_name: str,
    file_hash: str = None,
    total_chunks: int = 0,
    db_path: str = DB_PATH
) -> dict:
    """
    Registers or updates document metadata in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (
            document_id, file_name, file_hash, total_chunks, updated_at
        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(document_id) DO UPDATE SET
            file_name = excluded.file_name,
            file_hash = excluded.file_hash,
            total_chunks = excluded.total_chunks,
            updated_at = CURRENT_TIMESTAMP
    """, (str(document_id), str(file_name), str(file_hash) if file_hash else None, int(total_chunks)))

    conn.commit()

    row = cursor.execute("SELECT * FROM documents WHERE document_id = ?", (str(document_id),)).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_document(document_id: str, db_path: str = DB_PATH) -> dict:
    """
    Retrieves a document record from SQLite by document_id.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM documents WHERE document_id = ?", (str(document_id),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_documents(db_path: str = DB_PATH) -> list:
    """
    Retrieves all registered documents from SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM documents ORDER BY created_at ASC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_document(document_id: str, db_path: str = DB_PATH) -> bool:
    """
    Deletes a document and its associated chunks from SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (str(document_id),))
    cursor.execute("DELETE FROM documents WHERE document_id = ?", (str(document_id),))
    conn.commit()
    conn.close()
    return True


def compute_file_hash(file_path: str) -> str:
    """
    Computes SHA-256 hash of a file for deterministic versioning and change detection.
    """
    if not os.path.exists(file_path):
        return None
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_canonical_document_id(file_name_or_path: str) -> str:
    """
    Returns a deterministic, canonical document ID from a file name or path.
    Preserves 'machine_learning_book' for the historical textbook.
    """
    base_name = os.path.basename(file_name_or_path).lower()
    if "machine learning" in base_name or "mitchell" in base_name:
        return "machine_learning_book"
    elif "deep learning" in base_name:
        return "deep_learning"
    elif "artificial intelligence" in base_name:
        return "artificial_intelligence"
    else:
        # Fallback: sanitized slug
        slug = os.path.splitext(os.path.basename(file_name_or_path))[0]
        return "".join(c if c.isalnum() else "_" for c in slug.lower()).strip("_")


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Generates a stable, deterministic chunk ID.
    Preserves exact historical 'chunk_{index}' format for machine_learning_book,
    and '{document_id}_chunk_{index}' for new documents to prevent cross-book collisions.
    """
    if document_id == "machine_learning_book":
        return f"chunk_{chunk_index}"
    return f"{document_id}_chunk_{chunk_index}"


def check_document_status(document_id: str, file_path: str, db_path: str = DB_PATH) -> str:
    """
    Determines document state:
    - 'NEW': Document does not exist in SQLite or has 0 chunks.
    - 'MODIFIED': Document exists but file content hash has changed.
    - 'EXISTING': Document exists with matching content hash and chunks.
    """
    doc = get_document(document_id, db_path)
    if not doc or doc.get("total_chunks", 0) == 0:
        return "NEW"

    current_hash = compute_file_hash(file_path)
    if current_hash and doc.get("file_hash") and current_hash != doc["file_hash"]:
        return "MODIFIED"

    return "EXISTING"


# ============================================================
# CHUNK CRUD OPERATIONS
# ============================================================

def compute_text_hash(text: str) -> str:
    """
    Returns SHA-256 hash of chunk text for content integrity.
    """
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def register_chunk(
    chunk_id: str,
    document_id: str,
    chunk_text: str,
    source: str,
    page: int,
    chunk_index: int = 0,
    royalty: float = 1.0,
    db_path: str = DB_PATH
):
    """
    Registers or updates a single chunk in SQLite.
    Initial state: frequency = 0, log_frequency = 0.0, normalized_frequency = 0.0,
    royalty = 1.0, price = 1.0.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    text_hash = compute_text_hash(chunk_text)

    cursor.execute("""
        INSERT INTO chunks (
            chunk_id, document_id, chunk_index, text_hash, chunk_text, source, page,
            raw_uniqueness, normalized_uniqueness,
            raw_information_density, normalized_information_density,
            frequency, log_frequency, normalized_frequency,
            royalty, price, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chunk_id) DO UPDATE SET
            document_id=excluded.document_id,
            chunk_index=excluded.chunk_index,
            text_hash=excluded.text_hash,
            chunk_text=excluded.chunk_text,
            source=excluded.source,
            page=excluded.page,
            updated_at=CURRENT_TIMESTAMP
    """, (str(chunk_id), str(document_id), int(chunk_index), text_hash, chunk_text, source, page, royalty, royalty))

    conn.commit()
    conn.close()


def register_chunks(
    chunks,
    document_id: str = "machine_learning_book",
    default_royalty: float = 1.0,
    db_path: str = DB_PATH
):
    """
    Registers a list of LangChain Document chunks into SQLite.
    Preserves existing frequency/usage state if already registered.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.metadata.get("chunk_id", f"chunk_{index}")
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["royalty"] = default_royalty
        chunk.metadata["chunk_index"] = index

        source = str(chunk.metadata.get("source", "Unknown"))
        page = int(chunk.metadata.get("page", 0))
        chunk_text = chunk.page_content
        text_hash = compute_text_hash(chunk_text)

        cursor.execute("""
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text_hash, chunk_text, source, page,
                raw_uniqueness, normalized_uniqueness,
                raw_information_density, normalized_information_density,
                frequency, log_frequency, normalized_frequency,
                royalty, price, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                text_hash=excluded.text_hash,
                chunk_text=excluded.chunk_text,
                source=excluded.source,
                page=excluded.page,
                updated_at=CURRENT_TIMESTAMP
        """, (str(chunk_id), str(document_id), index, text_hash, chunk_text, source, page, default_royalty, default_royalty))

    conn.commit()
    conn.close()
    print(f"Registered {len(chunks)} chunks for document '{document_id}' in SQLite ({db_path})")


def get_chunks_by_document(document_id: str, db_path: str = DB_PATH) -> list:
    """
    Retrieves all chunks belonging to a document from SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC
    """, (str(document_id),)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_chunks_by_document(document_id: str, db_path: str = DB_PATH) -> int:
    """
    Deletes all chunks belonging to a document from SQLite.
    Returns the count of deleted chunks.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (str(document_id),))
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def sync_chunks_from_chroma(
    chroma_dir: str = "./chroma_db",
    collection_name: str = "machine_learning_book",
    document_id: str = "machine_learning_book",
    default_royalty: float = 1.0,
    db_path: str = DB_PATH
):
    """
    Synchronizes chunks existing in ChromaDB directly into SQLite,
    ensuring initial frequency = 0, royalty = 1.0, and price = 1.0.
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    total_count = collection.count()

    print(f"Reading {total_count} chunks from ChromaDB collection '{collection_name}'...")
    data = collection.get(include=["documents", "metadatas"])

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    ids = data["ids"]
    docs = data["documents"]
    metas = data["metadatas"]

    for i in range(len(ids)):
        meta = metas[i] if metas and metas[i] else {}
        # Use chunk_id from metadata if present, else fallback to chunk_{i+1}
        raw_cid = meta.get("chunk_id")
        if raw_cid is not None:
            chunk_id = f"chunk_{raw_cid}" if not str(raw_cid).startswith("chunk_") else str(raw_cid)
        else:
            chunk_id = f"chunk_{i+1}"

        source = str(meta.get("source", "Unknown"))
        page = int(meta.get("page", 0))
        chunk_text = docs[i] if docs and docs[i] else ""

        cursor.execute("""
            INSERT INTO chunks (
                chunk_id, document_id, chunk_text, source, page,
                raw_uniqueness, normalized_uniqueness,
                frequency, log_frequency, normalized_frequency,
                royalty, price, updated_at
            ) VALUES (?, ?, ?, ?, ?, 0.0, 0.0, 0, 0.0, 0.0, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_text=excluded.chunk_text,
                source=excluded.source,
                page=excluded.page,
                updated_at=CURRENT_TIMESTAMP
        """, (chunk_id, document_id, chunk_text, source, page, default_royalty, default_royalty))

    conn.commit()
    conn.close()
    print(f"Successfully synchronized {len(ids)} chunks from ChromaDB into SQLite.")


def compute_raw_uniqueness(
    k: int = 5,
    chroma_dir: str = "./chroma_db",
    collection_name: str = "machine_learning_book"
):
    """
    Computes raw uniqueness for each chunk in ChromaDB using K-nearest neighbors cosine similarity.
    1. Finds K nearest neighbors for each chunk using embedding cosine similarity (excluding self).
    2. Calculates average cosine similarity to those K neighbors.
    3. Calculates raw_uniqueness = 1.0 - average_neighbor_similarity.
    """
    import chromadb
    import numpy as np

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    data = collection.get(include=["embeddings", "metadatas"])

    if not data["ids"] or data["embeddings"] is None or len(data["embeddings"]) == 0:
        return {}

    metas = data["metadatas"]
    ids = data["ids"]
    embeddings = np.array(data["embeddings"], dtype=np.float32)
    n_chunks = len(ids)

    # Normalize embeddings to ensure exact cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.clip(norms, 1e-12, None)

    # Cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    # Exclude self-similarity by setting diagonal to -infinity
    np.fill_diagonal(similarity_matrix, -np.inf)

    # Effective K cannot exceed available other chunks
    effective_k = min(k, max(1, n_chunks - 1))

    # Get top K similarities for each chunk
    sorted_similarities = np.sort(similarity_matrix, axis=1)
    top_k_similarities = sorted_similarities[:, -effective_k:]

    # Average neighbor similarity
    avg_neighbor_similarity = np.mean(top_k_similarities, axis=1)

    # raw_uniqueness = 1 - average_neighbor_similarity
    raw_uniqueness_scores = 1.0 - avg_neighbor_similarity

    # Map chunk_id to raw_uniqueness
    results = {}
    for i in range(n_chunks):
        meta = metas[i] if metas and metas[i] else {}
        raw_cid = meta.get("chunk_id")
        if raw_cid is not None:
            cid_str = str(raw_cid)
            if cid_str.isdigit():
                chunk_id = f"chunk_{cid_str}"
            else:
                chunk_id = cid_str
        else:
            chunk_id = str(ids[i])
        results[chunk_id] = float(raw_uniqueness_scores[i])

    return results


def update_raw_uniqueness_in_db(
    uniqueness_map: dict,
    db_path: str = DB_PATH
):
    """
    Updates the raw_uniqueness column in SQLite for each chunk in uniqueness_map.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        UPDATE chunks
        SET raw_uniqueness = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, [(score, chunk_id) for chunk_id, score in uniqueness_map.items()])

    conn.commit()
    conn.close()
    print(f"Updated raw_uniqueness for {len(uniqueness_map)} chunks in SQLite.")


def normalize_uniqueness_in_db(db_path: str = DB_PATH):
    """
    Normalizes raw_uniqueness using Min-Max normalization:
        normalized_uniqueness = (raw_uniqueness - min_raw) / (max_raw - min_raw)
    If max_raw == min_raw, sets normalized_uniqueness = 0.0 for all chunks.
    Values are constrained to [0.0, 1.0].
    Updates normalized_uniqueness in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows = cursor.execute("SELECT chunk_id, raw_uniqueness FROM chunks").fetchall()
    if not rows:
        conn.close()
        return

    raw_values = [float(row["raw_uniqueness"]) for row in rows]
    min_raw = min(raw_values)
    max_raw = max(raw_values)

    updates = []
    if max_raw == min_raw or abs(max_raw - min_raw) < 1e-12:
        for row in rows:
            updates.append((0.0, row["chunk_id"]))
    else:
        denom = max_raw - min_raw
        for row in rows:
            norm_val = (float(row["raw_uniqueness"]) - min_raw) / denom
            norm_val = max(0.0, min(1.0, float(norm_val)))
            updates.append((norm_val, row["chunk_id"]))

    cursor.executemany("""
        UPDATE chunks
        SET normalized_uniqueness = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print(f"Normalized uniqueness updated for {len(updates)} chunks (min_raw={min_raw:.4f}, max_raw={max_raw:.4f}).")


# ============================================================
# INFORMATION DENSITY (STAGE 6 - SLM TOKEN SURPRISAL)
# ============================================================

def compute_chunk_surprisal(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_length: int = 48
) -> float:
    """
    Computes average token surprisal (Information Density) for a chunk:
        I(t) = -log2 P(t | previous tokens)
    Returns:
        Average token surprisal over content tokens in bits/token.
        Returns 0.0 for empty or single-token chunks.
    """
    if not text or not text.strip():
        return 0.0

    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    input_ids = inputs["input_ids"].to(device)

    # Need at least 2 tokens to predict next token
    if input_ids.shape[1] < 2:
        return 0.0

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape [1, seq_len, vocab_size]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_nats = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )

        # Convert nats to bits: -log2 P = (-ln P) / ln(2)
        token_surprisals = token_nats / math.log(2.0)
        avg_surprisal = token_surprisals.mean().item()

    return float(round(avg_surprisal, 4))


def compute_batch_surprisals(
    texts: list,
    model,
    tokenizer,
    device: str = "cpu",
    max_length: int = 48
) -> list:
    """
    Computes average token surprisal (Information Density) for a batch of chunks in parallel:
        I(t) = -log2 P(t | previous tokens)
    Returns:
        List of average token surprisal values in bits/token.
    """
    if not texts:
        return []

    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    valid_indices = []
    valid_texts = []
    scores = [0.0] * len(texts)

    for idx, t in enumerate(texts):
        if t and t.strip():
            valid_indices.append(idx)
            valid_texts.append(t)

    if not valid_texts:
        return scores

    enc = tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if input_ids.shape[1] < 2:
        return scores

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_nats = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        ).reshape(shift_labels.shape)

        token_surprisals = token_nats / math.log(2.0)
        masked_surprisals = (token_surprisals * shift_mask).sum(dim=1)
        valid_counts = shift_mask.sum(dim=1).clamp(min=1)
        avg_surprisals = (masked_surprisals / valid_counts).tolist()

        for orig_idx, s in zip(valid_indices, avg_surprisals):
            scores[orig_idx] = float(round(s, 4))

    return scores


def compute_all_information_densities(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    only_uncomputed: bool = False,
    db_path: str = DB_PATH,
    max_length: int = 48,
    batch_size: int = 16
) -> dict:
    """
    Loads Qwen/Qwen2.5-0.5B once and calculates raw_information_density
    for chunks in SQLite using batched inference.
    Returns:
        dict mapping chunk_id to raw_information_density
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device == "cpu":
        torch.set_num_threads(8)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if only_uncomputed:
        rows = cursor.execute("SELECT chunk_id, chunk_text FROM chunks WHERE raw_information_density IS NULL OR raw_information_density = 0.0").fetchall()
    else:
        rows = cursor.execute("SELECT chunk_id, chunk_text FROM chunks").fetchall()

    if not rows:
        conn.close()
        return {}

    print(f"Loading SLM '{model_name}' for Information Density calculation on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("SLM loaded successfully.", flush=True)

    results = {}
    total = len(rows)
    print(f"Calculating token surprisal for {total} chunks (batch_size={batch_size}, max_length={max_length})...", flush=True)

    batch_updates = []
    for start_idx in range(0, total, batch_size):
        batch_rows = rows[start_idx:start_idx + batch_size]
        batch_cids = [r["chunk_id"] for r in batch_rows]
        batch_texts = [r["chunk_text"] or "" for r in batch_rows]

        batch_scores = compute_batch_surprisals(batch_texts, model, tokenizer, device=device, max_length=max_length)

        for cid, score in zip(batch_cids, batch_scores):
            results[cid] = score
            batch_updates.append((score, cid))

        if len(batch_updates) >= 50 or (start_idx + batch_size) >= total:
            cursor.executemany(
                "UPDATE chunks SET raw_information_density = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
                batch_updates
            )
            conn.commit()
            processed_count = min(start_idx + batch_size, total)
            print(f"  Processed {processed_count}/{total} chunks (Last: {batch_cids[-1]} -> {batch_scores[-1]:.4f} bits/token)", flush=True)
            batch_updates = []

    conn.close()
    return results


def update_raw_information_density_in_db(
    density_map: dict,
    db_path: str = DB_PATH
):
    """
    Updates raw_information_density column in SQLite for each chunk in density_map.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.executemany("""
        UPDATE chunks
        SET raw_information_density = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, [(score, chunk_id) for chunk_id, score in density_map.items()])
    conn.commit()
    conn.close()
    print(f"Updated raw_information_density for {len(density_map)} chunks in SQLite.")


def normalize_information_density_in_db(db_path: str = DB_PATH):
    """
    Normalizes raw_information_density using Min-Max normalization:
        normalized_information_density = (raw - min) / (max - min)
    Handles zero variance safely (sets 0.0 for all if max == min).
    Clamped to [0.0, 1.0].
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT chunk_id, raw_information_density FROM chunks").fetchall()
    if not rows:
        conn.close()
        return

    scores = [float(row["raw_information_density"]) for row in rows]
    min_val = min(scores)
    max_val = max(scores)

    updates = []
    if max_val == min_val or abs(max_val - min_val) < 1e-12:
        for row in rows:
            updates.append((0.0, row["chunk_id"]))
    else:
        denom = max_val - min_val
        for row in rows:
            norm_val = (float(row["raw_information_density"]) - min_val) / denom
            norm_val = max(0.0, min(1.0, float(norm_val)))
            updates.append((norm_val, row["chunk_id"]))

    cursor.executemany("""
        UPDATE chunks
        SET normalized_information_density = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print(f"Normalized information density updated for {len(updates)} chunks (min={min_val:.4f}, max={max_val:.4f}).")



def get_chunk(chunk_id: str, db_path: str = DB_PATH):
    """
    Retrieves metadata and pricing information for a single chunk from SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    # Normalize ID to handle '1' vs 'chunk_1'
    cid_str = str(chunk_id)
    if not cid_str.startswith("chunk_") and cid_str.isdigit():
        formatted_id = f"chunk_{cid_str}"
    else:
        formatted_id = cid_str

    row = cursor.execute("""
        SELECT * FROM chunks WHERE chunk_id = ? OR chunk_id = ?
    """, (cid_str, formatted_id)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_chunks_by_ids(chunk_ids: list, db_path: str = DB_PATH):
    """
    Retrieves metadata for a list of chunk IDs from SQLite, returning a dictionary keyed by chunk_id.
    """
    if not chunk_ids:
        return {}
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = cursor.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", [str(cid) for cid in chunk_ids]).fetchall()
    conn.close()
    return {row["chunk_id"]: dict(row) for row in rows}


def get_all_chunks(db_path: str = DB_PATH):
    """
    Retrieves all chunk records from SQLite as a list of dicts.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM chunks").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def increment_chunk_frequency(chunk_id: str, db_path: str = DB_PATH) -> int:
    """
    Increments the usage frequency of a chunk by 1 in SQLite.
    Only called when UCOSA successfully selects and charges for the chunk.
    Returns the new frequency value.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cid_str = str(chunk_id)
    if not cid_str.startswith("chunk_") and cid_str.isdigit():
        formatted_id = f"chunk_{cid_str}"
    else:
        formatted_id = cid_str

    cursor.execute("""
        UPDATE chunks
        SET frequency = frequency + 1, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ? OR chunk_id = ?
    """, (cid_str, formatted_id))

    row = cursor.execute("""
        SELECT frequency FROM chunks WHERE chunk_id = ? OR chunk_id = ?
    """, (cid_str, formatted_id)).fetchone()

    conn.commit()
    conn.close()

    if row is None:
        raise ValueError(f"Chunk ID '{chunk_id}' not found in database.")
    return row["frequency"]


def calculate_log_frequency(frequency: int) -> float:
    """
    Calculates log frequency using natural logarithm:
        log_frequency = ln(1 + frequency)
    """
    return float(math.log(1.0 + max(0, frequency)))


def update_all_log_frequencies(db_path: str = DB_PATH):
    """
    Calculates and updates log_frequency for all chunks in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows = cursor.execute("SELECT chunk_id, frequency FROM chunks").fetchall()
    updates = [
        (calculate_log_frequency(row["frequency"]), row["chunk_id"])
        for row in rows
    ]

    cursor.executemany("""
        UPDATE chunks
        SET log_frequency = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print(f"Updated log_frequency for {len(updates)} chunks in SQLite.")


def normalize_frequency_in_db(db_path: str = DB_PATH):
    """
    Normalizes log_frequency using Min-Max normalization:
        normalized_frequency = (log_frequency - min_log) / (max_log - min_log)
    COLD-START RULE:
    When max_log == min_log (e.g. all frequency = 0), sets normalized_frequency = 0.0 for all chunks.
    No division by zero occurs.
    Values are constrained to [0.0, 1.0].
    Updates normalized_frequency in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows = cursor.execute("SELECT chunk_id, log_frequency FROM chunks").fetchall()
    if not rows:
        conn.close()
        return

    log_values = [float(row["log_frequency"]) for row in rows]
    min_log = min(log_values)
    max_log = max(log_values)

    updates = []
    if max_log == min_log or abs(max_log - min_log) < 1e-12:
        for row in rows:
            updates.append((0.0, row["chunk_id"]))
    else:
        denom = max_log - min_log
        for row in rows:
            norm_val = (float(row["log_frequency"]) - min_log) / denom
            norm_val = max(0.0, min(1.0, float(norm_val)))
            updates.append((norm_val, row["chunk_id"]))

    cursor.executemany("""
        UPDATE chunks
        SET normalized_frequency = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print(f"Normalized frequency updated for {len(updates)} chunks (min_log={min_log:.4f}, max_log={max_log:.4f}).")


# ============================================================
# DYNAMIC PRICING CONFIGURATION (STAGE 8)
# ============================================================

DEFAULT_BASE_PRICE = 1.0
DEFAULT_ROYALTY = 1.0
USE_INFORMATION_DENSITY = True


def calculate_price(
    normalized_uniqueness: float,
    normalized_frequency: float,
    normalized_information_density: float = 1.0,
    base_price: float = DEFAULT_BASE_PRICE,
    royalty: float = DEFAULT_ROYALTY,
    use_information_density: bool = USE_INFORMATION_DENSITY
) -> float:
    """
    Calculates dynamic chunk price / cost:
        Enhanced (Stage 8):
            Cost = (BasePrice * NormalizedFrequency * NormalizedInformationDensity * NormalizedUniqueness) + Royalty
            Cost = 1.0 + (NF * NID * NU)
        Original (Stage 18 ablation):
            Cost = (BasePrice * NormalizedFrequency * NormalizedUniqueness) + Royalty
            Cost = 1.0 + (NF * NU)
    At cold start (NF = 0):
        Cost = Royalty = 1.0
    Bounds: [1.0, 2.0]
    """
    nu = float(normalized_uniqueness)
    nf = float(normalized_frequency)

    if use_information_density:
        nid = float(normalized_information_density)
        dynamic_component = base_price * nf * nid * nu
    else:
        dynamic_component = base_price * nf * nu

    return float(round(dynamic_component + float(royalty), 4))


def update_all_prices_in_db(
    base_price: float = DEFAULT_BASE_PRICE,
    use_information_density: bool = USE_INFORMATION_DENSITY,
    db_path: str = DB_PATH
):
    """
    Calculates and updates dynamic prices for all chunks in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT chunk_id, normalized_uniqueness, normalized_frequency, normalized_information_density, royalty
        FROM chunks
    """).fetchall()

    updates = [
        (
            calculate_price(
                row["normalized_uniqueness"],
                row["normalized_frequency"],
                normalized_information_density=row["normalized_information_density"] if row["normalized_information_density"] is not None else 1.0,
                base_price=base_price,
                royalty=row["royalty"] if row["royalty"] is not None else DEFAULT_ROYALTY,
                use_information_density=use_information_density
            ),
            row["chunk_id"]
        )
        for row in rows
    ]

    cursor.executemany("""
        UPDATE chunks
        SET price = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()
    conn.close()
    print(f"Updated dynamic prices for {len(updates)} chunks in SQLite (use_nid={use_information_density}).")


def record_chunk_usage(
    chunk_id: str,
    base_price: float = DEFAULT_BASE_PRICE,
    use_information_density: bool = USE_INFORMATION_DENSITY,
    db_path: str = DB_PATH
) -> dict:
    """
    Executes the usage pipeline when UCOSA successfully selects a chunk:
    1. Increments frequency for the selected chunk.
    2. Recalculates log frequency for all chunks.
    3. Recalculates normalized frequency across all chunks (with cold-start safety).
    4. Recalculates dynamic prices according to the new normalizations (NF * NID * NU).
    5. Persists all updates atomically in SQLite.
    6. Returns the updated record for the selected chunk.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cid_str = str(chunk_id)
    if not cid_str.startswith("chunk_") and cid_str.isdigit():
        formatted_id = f"chunk_{cid_str}"
    else:
        formatted_id = cid_str

    # 1. Increment frequency for selected chunk
    cursor.execute("""
        UPDATE chunks
        SET frequency = frequency + 1, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ? OR chunk_id = ?
    """, (cid_str, formatted_id))

    # 2. Fetch all chunks to update normalization
    rows = cursor.execute("""
        SELECT chunk_id, frequency, normalized_uniqueness, normalized_information_density, royalty
        FROM chunks
    """).fetchall()

    if not rows:
        conn.close()
        raise ValueError("No chunks found in database.")

    # 3. Compute log frequencies
    log_data = []
    log_values = []
    for row in rows:
        lf = calculate_log_frequency(row["frequency"])
        norm_nid = float(row["normalized_information_density"]) if row["normalized_information_density"] is not None else 1.0
        log_data.append((
            row["chunk_id"],
            lf,
            float(row["normalized_uniqueness"]),
            norm_nid,
            float(row["royalty"] if row["royalty"] is not None else DEFAULT_ROYALTY)
        ))
        log_values.append(lf)

    min_log = min(log_values)
    max_log = max(log_values)

    # 4. Compute normalized frequencies and dynamic prices
    updates = []
    for cid, lf, norm_uniq, norm_nid, royalty in log_data:
        if max_log == min_log or abs(max_log - min_log) < 1e-12:
            norm_freq = 0.0
        else:
            norm_freq = max(0.0, min(1.0, (lf - min_log) / (max_log - min_log)))

        price = calculate_price(
            norm_uniq,
            norm_freq,
            normalized_information_density=norm_nid,
            base_price=base_price,
            royalty=royalty,
            use_information_density=use_information_density
        )
        updates.append((lf, norm_freq, price, cid))

    # 5. Persist updates
    cursor.executemany("""
        UPDATE chunks
        SET log_frequency = ?, normalized_frequency = ?, price = ?, updated_at = CURRENT_TIMESTAMP
        WHERE chunk_id = ?
    """, updates)

    conn.commit()

    # 6. Fetch and return updated record for selected chunk
    updated_row = cursor.execute("""
        SELECT * FROM chunks WHERE chunk_id = ? OR chunk_id = ?
    """, (cid_str, formatted_id)).fetchone()

    conn.close()
    return dict(updated_row) if updated_row else {}


if __name__ == "__main__":
    init_db()
    sync_chunks_from_chroma()
    raw_uniq = compute_raw_uniqueness(k=5)
    update_raw_uniqueness_in_db(raw_uniq)
    normalize_uniqueness_in_db()
    update_all_log_frequencies()
    normalize_frequency_in_db()
    update_all_prices_in_db()
