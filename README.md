# DCAAS: Dynamic Cost-Aware Answer Synthesis (LB-CaaS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/vectorstore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/models-HuggingFace-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An economics-driven, budget-constrained **Retrieval-Augmented Generation (RAG)** system implementing **Limiting-Budget Context-as-a-Service (LB-CaaS)**. 

DCAAS treats knowledge retrieval as an economic transaction: chunks are dynamically priced based on query-independent intrinsic value (Information Density and Uniqueness) and access frequency. Chunks are selected under budget constraints via the online competitive knapsack algorithm **UCOSA (Utility-Cost Online Selection Algorithm)**, progressively expanded until empirical sufficiency, verified for factual entailment via **ModernBERT NLI**, and charged using an atomic **Transactional / Deferred Billing Model**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User([User Query]) --> Dec[Sub-Question Decomposition]
    Dec --> HR[Hybrid Retrieval: Dense MiniLM + Sparse BM25Okapi]
    HR --> EAL{Evidence Admissibility Layer\nCosine >= 0.35 or BM25 >= 12.0}
    
    EAL -- Inadmissible / Out-of-Scope --> OOS[Honest Non-Answer\nCharge: ₹0.00 | Balance Intact]
    EAL -- Admissible Candidates --> UCOSA[UCOSA Competitive Thresholding\nPsi(z) = (U/L)^z * (L/e)]
    
    UCOSA --> Pool[Ranked Qualified Evidence Pool]
    Pool --> PE[Progressive Evidence Expansion]
    
    subgraph Iterative Loop
        PE --> Gen[Answer Generation: Llama-3.2-1B-Instruct]
        Gen --> NLI[ModernBERT NLI Claim Verification]
        Gen --> Cov[Per-Subquestion Coverage Evaluator]
        NLI & Cov --> Guard{Faithfulness >= 0.65\nAND Coverage == COMPLETE?}
        Guard -- No & Budget Remaining --> PE
    end

    Guard -- Passed --> Commit[Transactional Commit\nCharge Total Evidence Cost\nUpdate SQLite Frequencies & Prices]
    Guard -- Exhausted / Insufficient --> Rollback[Transactional Rollback\nCharge: ₹0.00\nFrequencies & Prices Unchanged]
    
    Commit --> Output([Final Grounded Response])
    Rollback --> Output
    OOS --> Output
```

---

## 🚀 Key Features & Innovations

### 1. Intrinsic Knowledge Economics & Dynamic Pricing
Unlike standard RAG systems that treat all chunks uniformly, DCAAS models information value intrinsically:
- **Information Density ($\mathcal{I}$)**: Computed using token surprisal (negative log-likelihood) under `Qwen/Qwen2.5-0.5B`:
  $$\mathcal{I}(c) = - \frac{1}{|c|} \sum_{i=1}^{|c|} \log_2 P(t_i \mid t_{<i})$$
- **Uniqueness ($\mathcal{U}$)**: Average cosine distance to 5-nearest neighbors in corpus embedding space via `all-MiniLM-L6-v2`:
  $$\mathcal{U}(c) = \frac{1}{k} \sum_{j=1}^{k} \left(1 - \cos(\mathbf{e}_c, \mathbf{e}_{NN_j})\right)$$
- **Dynamic Pricing ($P$)**: Prices adapt dynamically with access frequency ($f$) in SQLite:
  $$P(c) = \text{base\_price} \times (1 + \alpha \cdot \hat{\mathcal{U}}(c)) \times (1 + \beta \cdot \hat{\mathcal{I}}(c)) \times (1 + \gamma \cdot \hat{f}(c))$$
  - *Cold-start guarantee*: Chunks with $f=0$ strictly cost base price ₹1.00.
  - *Retrieval non-increment invariance*: Frequency is never incremented during retrieval or provisional evaluation.

### 2. Multi-Subquery Hybrid Retrieval & Interleaving
- **Query Decomposition**: Decomposes complex and compound multi-topic queries into constituent subqueries.
- **Hybrid Fusion**: Combines dense semantic similarity with sparse keyword matching:
  $$\text{HybridScore} = \alpha \cdot \text{NormCosine} + (1 - \alpha) \cdot \text{NormBM25} \quad (\alpha = 0.5)$$
- **Subquery Balancing**: Interleaves candidate chunks across subqueries so higher-scoring topics do not crowd out lower-scoring ones.
- **Deduplication with Joint Association**: If a chunk supports multiple subqueries, it is mapped to multiple subquery IDs but **charged at most once**.

### 3. DCAAS Evidence Admissibility Layer
To prevent out-of-corpus queries (e.g., *"What was Apple's stock price in 1994?"*) from tricking the competitive budgeting algorithm:
- Candidates must pass an experimentally calibrated admissibility floor:
  $$\text{CosineSimilarity} \ge 0.35 \quad \lor \quad \text{BM25Okapi} \ge 12.0$$
- If zero candidates meet admissibility, retrieval immediately aborts, prevents false candidate qualification, returns an honest limitation notice, and charges ₹0.00.

### 4. Online Competitive Knapsack Allocation (UCOSA)
Implements the **Utility-Cost Online Selection Algorithm**:
- Budget state: $z = \frac{\text{Spent}}{\text{Budget}} \in [0, 1]$.
- Candidate utility: $\text{Utility}_j = \frac{\text{Relevance}_j}{\text{Price}_j}$.
- Dynamic competitive threshold:
  $$\Psi(z) = \left(\frac{U}{L}\right)^z \cdot \left(\frac{L}{e}\right)$$
  where $L = \min(\text{Utility})$, $U = \max(\text{Utility})$, and $e \approx 2.718$.
- Qualified candidate set:
  $$\mathcal{Q} = \{j \mid \text{Utility}_j \ge \Psi(z) \land \text{Price}_j \le \text{Budget} - \text{Spent}\}$$

### 5. Progressive Evidence Expansion
- Starts with the minimum sufficient evidence (1 chunk).
- Expands adaptively by prioritizing qualified chunks that address currently uncovered subquestions.
- Terminates early as soon as the answer is factually supported and complete, minimizing overall cost.

### 6. ModernBERT NLI Claim-Level Verification
- Evaluates premise-hypothesis entailment via `tasksource/ModernBERT-base-nli` (2048 token window).
- **Claim-Specific Premises**: Ranks active evidence chunks by lexical overlap per atomic claim rather than blindly truncating the global context.
- **Hypothesis Protection**: Uses `truncation="only_first"` to ensure the generated claim is never clipped.
- **Strict Verification Threshold**:
  $$\text{Faithfulness} \ge 0.65 \implies \text{SUPPORTED}$$
  $$\text{Faithfulness} < 0.65 \implies \text{NOT SUPPORTED (PARTIAL or INSUFFICIENT)}$$

### 7. Deferred Transactional Billing
- Progressive evidence expansion operates on **provisional staged usage**.
- **Final Commit Rule**:
  $$\text{Status} == \text{"SUPPORTED"} \land \text{Coverage} == \text{"COMPLETE"} \implies \text{Commit SQLite usage \& deduct balance}$$
- **Rollback Rule**:
  $$\text{Status} == \text{"INSUFFICIENT"} \lor \text{Coverage} \neq \text{"COMPLETE"} \implies \text{Rollback: Charge ₹0.00, balance \& frequencies unchanged}$$

---

## 📦 Required Models

All models run locally and are cached via Hugging Face:

| Task | Model | Context / Size |
|---|---|---|
| **Text Generation** | `meta-llama/Llama-3.2-1B-Instruct` | 1.23B params, 128k context |
| **Embeddings & Uniqueness** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim dense vectors |
| **Information Density** | `Qwen/Qwen2.5-0.5B` | 0.5B causal LM surprisal |
| **NLI Verification** | `tasksource/ModernBERT-base-nli` | 2048 sequence length |

---

## 📂 Repository Structure

```
d:\mean\caas\
│
├── RAG.py                   # Core DCAAS pipeline (Hybrid retrieval, UCOSA, progressive expansion,
│                            # NLI verification, generation, transactional billing, CLI)
├── ingest.py                # One-time offline indexing pipeline (PDF extraction, Qwen ID,
│                            # MiniLM Uniqueness, SQLite registration, ChromaDB vectorization)
├── db.py                    # SQLite database interface, schema management, and atomic dynamic pricing
├── requirements.txt         # Project dependencies
├── .env                     # Local environment configuration (HF_TOKEN)
├── caas.db                  # Persistent SQLite database storing chunk metadata, surprisal, and frequencies
├── chroma_db/               # Persistent ChromaDB vector store
│
├── Deep Learning.pdf        # Goodfellow, Bengio, Courville Deep Learning textbook
├── artificial intelligence.pdf # Russell & Norvig Artificial Intelligence: A Modern Approach
└── McGrawHill - Machine Learning -Tom Mitchell.pdf # Tom Mitchell Machine Learning textbook
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Pavana-Krishna-360/CAAS.git
cd CAAS
```

### 2. Set Up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
HF_TOKEN = your_huggingface_token_here
```

---

## ⚡ Execution Workflow

### Step 1: One-Time Corpus Ingestion (Offline Pipeline)
> **Note:** Run this once to parse the textbooks, compute Information Density and Uniqueness, populate `caas.db`, and build `chroma_db/`. You do not need to rerun this before every query.
```powershell
python ingest.py
```

### Step 2: Interactive RAG CLI
Start the interactive question-answering session:
```powershell
python RAG.py
```

Example session:
```
============================================================
LB-CaaS - MACHINE LEARNING RAG
============================================================

Total Budget: ₹100.00
Loading embedding model: all-MiniLM-L6-v2
Connected to ChromaDB.
Loading LLM: meta-llama/Llama-3.2-1B-Instruct
Minimum chunk cost: ₹1.00

Question: What is a well posed learning problem?

[Expansion Step 1] Added evidence chunk: chunk_18 (Cost: ₹1.04) | Active Evidence: 1
Generating answer with 1 evidence chunk(s)...
Verification result: SUPPORTED (Score: 0.7151) | Question Coverage: COMPLETE
Evidence is fully sufficient (SUPPORTED >= 0.65) and question coverage is complete! Stopping progressive expansion.

[BILLING COMMIT] Final answer ACCEPTED. Committed 1 chunks. Charged: ₹1.04

============================================================
DCAAS RESPONSE
============================================================
## Answer
Based on the retrieved context, a well-posed learning problem is defined as a computer program that improves its performance at some task through experience...

Faithfulness Status: SUPPORTED
Faithfulness Score: 0.7151
Question Coverage: COMPLETE
Evidence Chunks Used: 1
Cost Charged This Turn: ₹1.04
Remaining Balance: ₹98.96
============================================================
```

---

## 🧪 Comprehensive Evaluation Suite

To run the complete automated test matrix covering metric invariance, edge cases, multi-part queries, and transactional billing:

```powershell
python -c "
import RAG, run_full_evaluation
run_full_evaluation.run_all()
"
```

### Evaluation Benchmark Results

| Test Label | Query | Ret / Qual / Sel | Faithfulness | Coverage | Cost Charged | Remaining Balance | Billing Action |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TEST A** | *What is a well posed learning problem?* | 4 / 3 / 1 | **SUPPORTED** (0.7151) | **COMPLETE** | ₹1.04 | ₹98.96 | Committed |
| **TEST B** | *Explain the checkers problem.* | 4 / 4 / 1 | **SUPPORTED** (0.7715) | **COMPLETE** | ₹1.00 | ₹97.96 | Committed |
| **TEST C** | *Types of AI, backprop, & concept learning (Multi-query)* | 12 / 12 / 4 | **SUPPORTED** (0.7824) | **COMPLETE** | ₹4.15 | ₹93.80 | Committed |
| **TEST D** | *What are the issues of Machine Learning?* | 4 / 2 / 2 | **INSUFFICIENT** (0.3475) | **COMPLETE** | **₹0.00** | ₹93.80 | **Rolled Back** |
| **TEST E** | *What is convergence in Q-Learning?* | 4 / 4 / 4 | **SUPPORTED** (0.8014) | **COMPLETE** | ₹4.02 | ₹89.79 | Committed |
| **TEST F1** | *Repeated Query Turn 1: Well posed learning problem* | 4 / 3 / 1 | **SUPPORTED** (0.7151) | **COMPLETE** | ₹1.04 | ₹88.74 | Committed |
| **TEST F2** | *Repeated Query Turn 2: Dynamic price adaptation* | 4 / 3 / 1 | **SUPPORTED** (0.7151) | **COMPLETE** | ₹1.04 | ₹87.70 | Committed |
| **TEST G** | *Backpropagation vs Concept Learning* | 4 / 4 / 4 | **INSUFFICIENT** (0.2879) | **COMPLETE** | **₹0.00** | ₹87.70 | **Rolled Back** |
| **TEST H** | *Apple stock price in 1994 (Unsupported Out-of-Corpus)* | 4 / 0 / 0 | **INSUFFICIENT** (0.0000) | **INSUFFICIENT** | **₹0.00** | ₹87.70 | **Zero Charge** |

---

## 🛡️ Guarantees & Assertions

1. **No False Support**: A response with faithfulness score $< 0.65$ can **never** be labeled `SUPPORTED`.
2. **Honest Coverage**: Queries outside the corpus scope (like Apple stock price in 1994) are recognized as having zero admissible evidence and evaluated honestly as `INSUFFICIENT` coverage.
3. **No Double Charging**: If a single chunk resolves multiple subquestions in a decomposed query, it is charged **exactly once**.
4. **Zero-Waste Guarantee**: If the pipeline cannot substantiate an answer to $\ge 0.65$ faithfulness, the user is charged **₹0.00**, chunk access frequencies remain unchanged, and the budget is fully preserved.

---

## 📜 Citation & References

```bibtex
@article{dcaas2026,
  title={DCAAS: Dynamic Cost-Aware Answer Synthesis in Limiting-Budget Context-as-a-Service},
  author={Cheedella Rahul Sai Sudheer, Devulapalli Pavana Krishna and Gutta Jagan Mohan},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
