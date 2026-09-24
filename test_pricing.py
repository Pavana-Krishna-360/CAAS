import os
import sys
import math
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import db
import RAG
from langchain_core.documents import Document

TEST_DB_PATH = os.path.join(os.path.dirname(__file__), "test_caas.db")


def setup_test_db():
    """Creates a clean test database with controlled chunk samples."""
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    db.init_db(TEST_DB_PATH)

    # Insert 4 test chunks with varied uniqueness
    # chunk_A: high uniqueness (1.0)
    # chunk_B: medium uniqueness (0.5)
    # chunk_C: low uniqueness (0.0)
    # chunk_D: baseline chunk (0.75)
    test_chunks = [
        ("chunk_A", "doc1", "text A", "source1", 1, 0.8, 1.0, 0, 0.0, 0.0, 1.0, 1.0),
        ("chunk_B", "doc1", "text B", "source1", 2, 0.5, 0.5, 0, 0.0, 0.0, 1.0, 1.0),
        ("chunk_C", "doc1", "text C", "source1", 3, 0.2, 0.0, 0, 0.0, 0.0, 1.0, 1.0),
        ("chunk_D", "doc1", "text D", "source1", 4, 0.65, 0.75, 0, 0.0, 0.0, 1.0, 1.0),
    ]

    conn = db.get_db_connection(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO chunks (
            chunk_id, document_id, chunk_text, source, page,
            raw_uniqueness, normalized_uniqueness,
            frequency, log_frequency, normalized_frequency,
            royalty, price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, test_chunks)
    conn.commit()
    conn.close()


def teardown_test_db():
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass


def test_1_cold_start():
    print("\n--- TEST 1: All chunks have frequency 0 ---")
    setup_test_db()
    db.normalize_frequency_in_db(TEST_DB_PATH)
    db.update_all_prices_in_db(base_price=1.0, db_path=TEST_DB_PATH)

    conn = db.get_db_connection(TEST_DB_PATH)
    rows = conn.execute("SELECT chunk_id, frequency, normalized_frequency, price FROM chunks").fetchall()
    conn.close()

    for row in rows:
        assert row["frequency"] == 0, f"Chunk {row['chunk_id']} has non-zero frequency"
        assert row["normalized_frequency"] == 0.0, f"Chunk {row['chunk_id']} normalized_frequency is not 0"
        assert row["price"] == 1.0, f"Chunk {row['chunk_id']} price is not ₹1.00"

    print("✓ TEST 1 PASSED: All chunks have normalized_frequency = 0.0 and price = ₹1.00 under cold start.")


def test_2_repeated_usage():
    print("\n--- TEST 2: Use one chunk repeatedly ---")
    setup_test_db()

    # Repeatedly use chunk_A 5 times
    prev_freq = 0
    prev_log_freq = 0.0
    prev_norm_freq = 0.0
    prev_price = 1.0

    for i in range(1, 6):
        updated = db.record_chunk_usage("chunk_A", base_price=1.0, db_path=TEST_DB_PATH)
        assert updated["frequency"] == i, f"Expected freq {i}, got {updated['frequency']}"
        assert updated["frequency"] > prev_freq
        assert updated["log_frequency"] > prev_log_freq
        assert updated["normalized_frequency"] >= prev_norm_freq
        assert updated["price"] >= prev_price

        expected_log = math.log(1 + i)
        assert abs(updated["log_frequency"] - expected_log) < 1e-6

        prev_freq = updated["frequency"]
        prev_log_freq = updated["log_frequency"]
        prev_norm_freq = updated["normalized_frequency"]
        prev_price = updated["price"]

        print(f"  Usage {i}: freq={updated['frequency']}, log_freq={updated['log_frequency']:.4f}, norm_freq={updated['normalized_frequency']:.4f}, price=₹{updated['price']:.2f}")

    print("✓ TEST 2 PASSED: Frequency, log frequency, normalized frequency, and price monotonically increased.")


def test_3_uniqueness_comparison():
    print("\n--- TEST 3: Compare two chunks with different uniqueness ---")
    setup_test_db()

    # Give both chunk_A (norm_uniq=1.0) and chunk_B (norm_uniq=0.5) the same frequency = 3
    for _ in range(3):
        db.record_chunk_usage("chunk_A", base_price=1.0, db_path=TEST_DB_PATH)
        db.record_chunk_usage("chunk_B", base_price=1.0, db_path=TEST_DB_PATH)

    chunk_a = db.get_chunk("chunk_A", TEST_DB_PATH)
    chunk_b = db.get_chunk("chunk_B", TEST_DB_PATH)

    print(f"  Chunk A (Uniq={chunk_a['normalized_uniqueness']}, Freq={chunk_a['frequency']}) -> Price = ₹{chunk_a['price']:.4f}")
    print(f"  Chunk B (Uniq={chunk_b['normalized_uniqueness']}, Freq={chunk_b['frequency']}) -> Price = ₹{chunk_b['price']:.4f}")

    assert chunk_a["frequency"] == chunk_b["frequency"] == 3
    assert chunk_a["normalized_uniqueness"] > chunk_b["normalized_uniqueness"]
    assert chunk_a["price"] > chunk_b["price"]

    print("✓ TEST 3 PASSED: Higher uniqueness contributes a larger dynamic price when frequency is equal.")


def test_4_frequency_comparison():
    print("\n--- TEST 4: Compare two chunks with different frequency ---")
    setup_test_db()

    # Chunk A and Chunk B: compare price from frequency
    # Use chunk_A 5 times, chunk_B 1 time
    for _ in range(5):
        db.record_chunk_usage("chunk_A", base_price=1.0, db_path=TEST_DB_PATH)
    db.record_chunk_usage("chunk_B", base_price=1.0, db_path=TEST_DB_PATH)

    chunk_a = db.get_chunk("chunk_A", TEST_DB_PATH)
    chunk_b = db.get_chunk("chunk_B", TEST_DB_PATH)

    price_high_freq = db.calculate_price(normalized_uniqueness=0.8, normalized_frequency=chunk_a["normalized_frequency"])
    price_low_freq = db.calculate_price(normalized_uniqueness=0.8, normalized_frequency=chunk_b["normalized_frequency"])

    print(f"  High Frequency ({chunk_a['frequency']}, norm={chunk_a['normalized_frequency']:.4f}) -> Price at 0.8 uniq = ₹{price_high_freq:.4f}")
    print(f"  Low Frequency  ({chunk_b['frequency']}, norm={chunk_b['normalized_frequency']:.4f}) -> Price at 0.8 uniq = ₹{price_low_freq:.4f}")

    assert chunk_a["normalized_frequency"] > chunk_b["normalized_frequency"]
    assert price_high_freq > price_low_freq

    print("✓ TEST 4 PASSED: Higher frequency contributes a larger dynamic price when uniqueness is equal.")


def test_5_price_bounds():
    print("\n--- TEST 5: Verify minimum price = ₹1.00 and maximum price <= ₹2.00 ---")
    test_cases = [
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.8, 0.75),
    ]

    for u, f in test_cases:
        p = db.calculate_price(normalized_uniqueness=u, normalized_frequency=f, base_price=1.0, royalty=1.0)
        assert 1.0 <= p <= 2.0, f"Price {p} out of bounds [1.0, 2.0] for u={u}, f={f}"
        print(f"  Uniqueness={u:.2f}, Frequency={f:.2f} -> Price = ₹{p:.4f} (Valid: 1.0 <= {p:.4f} <= 2.0)")

    print("✓ TEST 5 PASSED: Minimum price is ₹1.00 and maximum price is ₹2.00 across all bounds.")


def test_6_retrieval_does_not_increase_frequency():
    print("\n--- TEST 6: Verify that merely retrieving a chunk does NOT increase frequency ---")
    setup_test_db()

    initial_chunk = db.get_chunk("chunk_A", TEST_DB_PATH)
    assert initial_chunk["frequency"] == 0

    # Simulate retrieval of chunk_A without UCOSA selection
    doc = Document(page_content="Simulated text", metadata={"chunk_id": "chunk_A"})
    retrieved_data = db.get_chunk(doc.metadata["chunk_id"], TEST_DB_PATH)
    doc.metadata.update(retrieved_data)

    after_retrieval = db.get_chunk("chunk_A", TEST_DB_PATH)
    assert after_retrieval["frequency"] == 0, f"Frequency increased to {after_retrieval['frequency']} on retrieval!"

    print(f"  Initial frequency: {initial_chunk['frequency']}, After retrieval: {after_retrieval['frequency']}")
    print("✓ TEST 6 PASSED: Retrieval alone does not increase chunk frequency.")


def test_7_ucosa_selection_increases_frequency():
    print("\n--- TEST 7: Verify that UCOSA selection DOES increase frequency ---")
    setup_test_db()

    # Retrieve 2 candidates
    doc_A = Document(page_content="Candidate A text", metadata={"chunk_id": "chunk_A"})
    doc_B = Document(page_content="Candidate B text", metadata={"chunk_id": "chunk_B"})

    doc_A.metadata.update(db.get_chunk("chunk_A", TEST_DB_PATH))
    doc_B.metadata.update(db.get_chunk("chunk_B", TEST_DB_PATH))

    # candidate A has distance 0.1 (relevance 0.9), candidate B has distance 0.5 (relevance 0.5)
    candidates = [(doc_A, 0.1), (doc_B, 0.5)]

    selected, min_p, L, U, valid_cands = RAG.ucosa_select_chunk(candidates, spent=0.0, budget=100.0)
    assert selected is not None
    assert selected["document"].metadata["chunk_id"] == "chunk_A"

    # Simulate charging and recording usage
    selected_cid = selected["document"].metadata["chunk_id"]
    db.record_chunk_usage(selected_cid, base_price=1.0, db_path=TEST_DB_PATH)

    updated_A = db.get_chunk("chunk_A", TEST_DB_PATH)
    updated_B = db.get_chunk("chunk_B", TEST_DB_PATH)

    assert updated_A["frequency"] == 1, "Selected chunk frequency was not incremented!"
    assert updated_B["frequency"] == 0, "Unselected chunk frequency was erroneously incremented!"

    print(f"  Selected chunk (chunk_A) frequency: {updated_A['frequency']}")
    print(f"  Unselected chunk (chunk_B) frequency: {updated_B['frequency']}")
    print("✓ TEST 7 PASSED: Only successfully selected and charged chunks increment frequency.")


def test_8_updated_price_used_in_subsequent_ucosa():
    print("\n--- TEST 8: Verify that updated price is used by the next UCOSA calculation ---")
    setup_test_db()

    # Query 1: Cold start price is ₹1.00
    doc_A1 = Document(page_content="Candidate A", metadata={"chunk_id": "chunk_A"})
    doc_A1.metadata.update(db.get_chunk("chunk_A", TEST_DB_PATH))
    dist = 0.2  # relevance = 0.8
    sel1, _, _, _, _ = RAG.ucosa_select_chunk([(doc_A1, dist)], spent=0.0, budget=100.0)
    price_query_1 = sel1["price"]
    ratio_query_1 = sel1["ratio"]

    print(f"  Query 1: Price = ₹{price_query_1:.4f}, Relevance = {sel1['relevance']:.4f}, R/P = {ratio_query_1:.4f}")

    # Charge and record usage
    db.record_chunk_usage("chunk_A", base_price=1.0, db_path=TEST_DB_PATH)

    # Query 2: Re-query chunk_A, now with updated dynamic price
    doc_A2 = Document(page_content="Candidate A", metadata={"chunk_id": "chunk_A"})
    doc_A2.metadata.update(db.get_chunk("chunk_A", TEST_DB_PATH))
    sel2, _, _, _, _ = RAG.ucosa_select_chunk([(doc_A2, dist)], spent=price_query_1, budget=100.0)
    price_query_2 = sel2["price"]
    ratio_query_2 = sel2["ratio"]

    print(f"  Query 2: Price = ₹{price_query_2:.4f}, Relevance = {sel2['relevance']:.4f}, R/P = {ratio_query_2:.4f}")

    assert price_query_2 > price_query_1, f"Price did not increase ({price_query_2} <= {price_query_1})"
    assert ratio_query_2 < ratio_query_1, f"R/P ratio did not decrease with higher price ({ratio_query_2} >= {ratio_query_1})"

    print("✓ TEST 8 PASSED: Subsequent UCOSA calculations use the updated dynamic price and recalculate R/P ratios accordingly.")

    teardown_test_db()


def main():
    print("=" * 60)
    print("RUNNING LB-CaaS DYNAMIC PRICING TEST SUITE (STAGE 13)")
    print("=" * 60)

    test_1_cold_start()
    test_2_repeated_usage()
    test_3_uniqueness_comparison()
    test_4_frequency_comparison()
    test_5_price_bounds()
    test_6_retrieval_does_not_increase_frequency()
    test_7_ucosa_selection_increases_frequency()
    test_8_updated_price_used_in_subsequent_ucosa()

    print("\n" + "=" * 60)
    print("ALL 8 TESTS COMPLETED AND PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
