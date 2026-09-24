import sqlite3
import os
import math

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
    Initializes the SQLite database and creates the chunks table if it does not exist.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_text TEXT,
            source TEXT,
            page INTEGER,
            raw_uniqueness REAL DEFAULT 0.0,
            normalized_uniqueness REAL DEFAULT 0.0,
            frequency INTEGER DEFAULT 0,
            log_frequency REAL DEFAULT 0.0,
            normalized_frequency REAL DEFAULT 0.0,
            royalty REAL DEFAULT 1.0,
            price REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {db_path}")


def register_chunk(
    chunk_id: str,
    document_id: str,
    chunk_text: str,
    source: str,
    page: int,
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
    """, (str(chunk_id), document_id, chunk_text, source, page, royalty, royalty))

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
    Sets frequency = 0 and royalty = 1.0.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = f"chunk_{index}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["royalty"] = default_royalty
        chunk.metadata["frequency"] = 0
        chunk.metadata["price"] = default_royalty

        source = str(chunk.metadata.get("source", "Unknown"))
        page = int(chunk.metadata.get("page", 0))
        chunk_text = chunk.page_content

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
    print(f"Registered {len(chunks)} chunks into SQLite database at {db_path}")


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
            chunk_id = f"chunk_{raw_cid}" if not str(raw_cid).startswith("chunk_") else str(raw_cid)
        else:
            chunk_id = f"chunk_{i+1}"
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


def calculate_price(
    normalized_uniqueness: float,
    normalized_frequency: float,
    base_price: float = 1.0,
    royalty: float = 1.0
) -> float:
    """
    Calculates dynamic chunk price:
        Price_i = BasePrice * NormalizedUniqueness_i * NormalizedFrequency_i + Royalty_i
        Price_i = 1.0 + (NormalizedUniqueness_i * NormalizedFrequency_i)
    Initial price range: ₹1.00 to ₹2.00
    """
    dynamic_component = base_price * float(normalized_uniqueness) * float(normalized_frequency)
    return float(round(dynamic_component + float(royalty), 4))


def update_all_prices_in_db(base_price: float = 1.0, db_path: str = DB_PATH):
    """
    Calculates and updates dynamic prices for all chunks in SQLite.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT chunk_id, normalized_uniqueness, normalized_frequency, royalty
        FROM chunks
    """).fetchall()

    updates = [
        (
            calculate_price(
                row["normalized_uniqueness"],
                row["normalized_frequency"],
                base_price=base_price,
                royalty=row["royalty"] if row["royalty"] is not None else 1.0
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
    print(f"Updated dynamic prices for {len(updates)} chunks in SQLite.")


def record_chunk_usage(
    chunk_id: str,
    base_price: float = 1.0,
    db_path: str = DB_PATH
) -> dict:
    """
    Executes the usage pipeline when UCOSA successfully selects a chunk:
    1. Increments frequency for the selected chunk.
    2. Recalculates log frequency for all chunks.
    3. Recalculates normalized frequency across all chunks (with cold-start safety).
    4. Recalculates dynamic prices according to the new normalizations.
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
        SELECT chunk_id, frequency, normalized_uniqueness, royalty
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
        log_data.append((row["chunk_id"], lf, float(row["normalized_uniqueness"]), float(row["royalty"] if row["royalty"] is not None else 1.0)))
        log_values.append(lf)

    min_log = min(log_values)
    max_log = max(log_values)

    # 4. Compute normalized frequencies and dynamic prices
    updates = []
    for cid, lf, norm_uniq, royalty in log_data:
        if max_log == min_log or abs(max_log - min_log) < 1e-12:
            norm_freq = 0.0
        else:
            norm_freq = max(0.0, min(1.0, (lf - min_log) / (max_log - min_log)))

        price = calculate_price(norm_uniq, norm_freq, base_price=base_price, royalty=royalty)
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
