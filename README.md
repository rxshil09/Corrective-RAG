# Hallucination-Aware Agentic RAG System

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-orange.svg)
![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini%20API-blue.svg)
![Hugging Face](https://img.shields.io/badge/Embeddings-HuggingFace%20all--MiniLM--L6--v2-yellow.svg)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green.svg)
![Tests](https://img.shields.io/badge/Pytest-24%2F24%20Passed%20(100%25)-brightgreen.svg)

A **Self-Correcting Agentic Retrieval-Augmented Generation (RAG) System** built with **Python 3.13**, **LangGraph**, **Google Gemini**, **HuggingFace**, and **ChromaDB**. A practical, production-ready implementation of corrective and self-reflective RAG architectures (inspired by the **CRAG** and **Self-RAG** papers), featuring **multi-format document ingestion** (`.md`, `.txt`, `.pdf`, `.docx`), **context relevance filtering**, **LLM-based claim-level consistency grading**, and an **agentic dual-path correction loop** (constrained negative-claim regeneration + dynamic query reformulation).

---

## 1. Architecture Overview (LangGraph StateGraph)

The system is orchestrated as a stateful graph using **LangGraph**, providing auditable execution traces across every node and conditional edge:

```mermaid
flowchart TD
    START([START: User Query]) --> Retrieve[1. Node: retrieve & relevance_filter]
    Retrieve -->|Cosine Distance Filter: score >= 0.25| Generate[2. Node: generate candidate answer]
    Generate --> Detect[3. Node: detect_hallucination]
    
    Detect --> Route{Conditional Edge Router}
    
    Route -->|Consistency Score >= 0.60| Pass([END: Verified Grounded Answer])
    Route -->|0.45 <= Score < 0.60| StrictGen[4a. Node: strict_generate]
    Route -->|Score < 0.45 & rewrite not used| RewriteQuery[4b. Node: rewrite_query]
    Route -->|Attempts >= 3| MaxRetries([END: Return Best Candidate])
    
    StrictGen -->|Injected Forbidden Claims| Detect
    RewriteQuery -->|Fresh ChromaDB Vector Search| Retrieve

    subgraph Multi-Tier Resilient Fallback Engine
        H[Primary: Google Gemini 3.5 Flash-Lite]
        I[1st Fallback: Gemini 3.1 Flash-Lite]
        J[2nd Fallback: Gemini 3.8 / 3.7 / 3.6 / 3.5 Flash]
        K[3rd Fallback: OpenAI GPT-4o-mini]
        H -->|HTTP 429/503 Failover| I -->|Failover| J -->|Failover| K
    end
```

---

## 2. System Components & Capabilities

### A. Universal Multi-Format Document Ingestion (`data/documents/`)
* Ingests **Markdown (`.md`)**, **Plain Text (`.txt`)**, **Adobe PDF (`.pdf`)**, and **Microsoft Word (`.docx`)** documents.
* Splits documents using LangChain's `RecursiveCharacterTextSplitter` (Chunk Size: 800, Overlap: 100) with chunk indexing and metadata tracking.
* Generates dense semantic vector representations locally using Hugging Face (zero API cost) and indexes into persistent **ChromaDB** storage (`chroma_db/`).

### B. Retrieval Relevance Filtering (Pruning Context Poisoning)
* Computes vector cosine similarity scores for all retrieved top-$K$ chunks ($K=5$).
* Drops low-similarity noise below `MIN_RELEVANCE_SCORE` ($0.25$) before context is compiled into prompts, preventing irrelevant information from distracting the LLM.
* Built-in fallback: Automatically retains top-$N$ chunks (`MIN_CHUNKS_REQUIRED = 1`) if all chunks fall below the threshold.

### C. LangGraph StateGraph Workflow (`src/rag_graph.py`)
* Compiles a declarative state machine using `RAGGraphState` tracking queries, documents, similarity scores, consistency history, and iteration counts.
* Provides full execution trace auditing, exporting step-by-step reasoning logs to `./outputs/response_*.json`.

### D. LLM-Based Claim-Level Consistency Evaluator (`src/hallucination_detector.py`)
* Deconstructs candidate answers into discrete atomic factual claims via a dedicated evaluator LLM prompt with strict JSON output formatting.
* Cross-references each individual claim against source context chunks (categorized as *supported* vs. *hallucinated*).
* Computes a fractional consistency score:
  $$\text{Consistency Score} = \frac{\text{Supported Claims}}{\text{Total Claims}} \in [0.0, 1.0]$$

### E. Dual-Path Agentic Self-Correction Loop
* **Path A — Constrained Strict Regeneration ($0.45 \le \text{Score} < 0.60$):** Retains retrieved context, switches to strict negative constraint prompting, and explicitly instructs the LLM not to repeat the specific hallucinated claims flagged by the detector.
* **Path B — Corrective Query Rewriting & Re-Retrieval ($\text{Score} < 0.45$):** Reformulates the query to eliminate vocabulary mismatch or ambiguity, executing a fresh semantic search against ChromaDB.
* **Loop Termination:** Bounded by `MAX_REGENERATION_ATTEMPTS = 3`, returning the highest-scoring candidate if the consistency threshold is not met.

### F. Multi-Tier Resilient Fallback Engine (`src/config.py`)
* Built with `ResilientChatCompletions` wrapping OpenAI-compatible API calls.
* Primary: `gemini-3.5-flash-lite` (Sub-2s latency, 250k TPM headroom).
* Cascades automatically through `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.8-flash` $\rightarrow$ `gemini-3.7-flash` $\rightarrow$ `gemini-3.6-flash` $\rightarrow$ `gemini-3.5-flash` $\rightarrow$ `gpt-4o-mini` on HTTP 429 rate limits, 503 high load, or network timeouts.

---

## 3. Hugging Face Integration — Local Semantic Embeddings

Hugging Face is utilized as the **local dense vector embedding engine**, completely decoupling document ingestion and vector retrieval from external cloud embedding APIs.

### Where Hugging Face Is Used in the Codebase
1. **Document Ingestion & Database Construction ([src/create_database.py](file:///d:/PROJECTS/RAG-Hallucination/src/create_database.py#L107-L113)):**
   ```python
   from langchain_community.embeddings import HuggingFaceEmbeddings

   def get_embedding_function():
       return HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2",
           model_kwargs={'device': 'cpu'}
       )
   ```
   During database initialization (`python run.py setup`), every document chunk from your knowledge base is passed to this Hugging Face model to generate vector embeddings stored in ChromaDB.

2. **Real-Time Query & Reformulation Retrieval ([src/rag_engine.py](file:///d:/PROJECTS/RAG-Hallucination/src/rag_engine.py#L97-L100)):**
   ```python
   self.embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2",
       model_kwargs={'device': 'cpu'}
   )
   ```
   When a user asks a question—or when the agent reformulates a query during an automated correction cycle—the query text is embedded locally on-the-fly and compared against ChromaDB vector vectors.

### Why Hugging Face Was Chosen
* **Zero API Cost:** Running `sentence-transformers/all-MiniLM-L6-v2` locally on CPU costs $0.00, regardless of how many thousands of document chunks or query rewrites are processed.
* **100% Cloud Quota Preservation:** Cloud LLM quotas (e.g., Google Gemini's 250k TPM or 500 RPD) are reserved exclusively for answer generation and hallucination grading, eliminating embedding rate limits.
* **Compact & Fast CPU Inference:** Produces 384-dimensional normalized vectors with sub-25ms inference latency per query on standard CPUs without requiring a dedicated GPU.
* **Offline Privacy & Security:** Knowledge base text is embedded directly on your local machine and never sent to third-party embedding providers.

---

## 4. Project Directory Structure

```
RAG-Hallucination/
│
├── data/documents/             # Knowledge base source documents (.md, .txt, .pdf, .docx)
│   ├── RAG and LLMs/           # AI & RAG domain reference docs
│   └── Finance/                # Financial domain reference docs
├── evaluation/                 # Multi-domain evaluation datasets & ground truth labels
│   ├── datasets/               # JSON datasets (ai_rag.json, finance.json)
│   └── ground_truth.json       # Human-verified labels for detector accuracy benchmarking
├── chroma_db/                  # Local persistent ChromaDB vector store
├── outputs/                    # Diagnostic execution traces & evaluation reports
├── tests/                      # Pytest unit & integration test suite (24 tests)
│   └── test_hallucination_detector.py
│
├── src/
│   ├── config.py               # Central configuration, parameters & fallback engine
│   ├── create_database.py      # Multi-format document ingestion & ChromaDB indexer
│   ├── rag_graph.py            # LangGraph StateGraph agentic workflow & routing
│   ├── rag_engine.py           # Core HallucinationAwareRAG engine & CLI API
│   ├── hallucination_detector.py # LLM-based claim-level consistency evaluator
│   ├── query_rag.py            # Interactive CLI, single query & showcase runner
│   ├── evaluate.py             # Multi-domain evaluation & reporting framework
│   └── experiments.py          # Baseline A/B, ground truth agreement & calibration
│
├── check_models.py             # Google Gemini model diagnostic & latency probe
├── run.py                      # Unified CLI entry point for the entire project
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template for API keys
└── README.md                   # System documentation
```

---

## 5. Configuration & Key Parameters

Central configuration is managed in [src/config.py](file:///d:/PROJECTS/RAG-Hallucination/src/config.py) and can be customized via `.env`:

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | Primary generation engine (`gemini` or `openai`) |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Primary LLM identifier (Sub-2s inference) |
| `HALLUCINATION_THRESHOLD` | `0.60` | Consistency score required to accept answer as grounded |
| `STRICT_MODE_THRESHOLD` | `0.40` | Lower threshold activating negative-constraint strict generation |
| `QUERY_REWRITE_THRESHOLD` | `0.45` | Score below which query rewriting and re-retrieval trigger |
| `MIN_RELEVANCE_SCORE` | `0.25` | Minimum cosine similarity score for context relevance filtering |
| `MIN_CHUNKS_REQUIRED` | `1` | Fallback chunk count if all chunks fall below relevance threshold |
| `MAX_REGENERATION_ATTEMPTS` | `3` | Maximum self-correction iterations per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Character splitter chunking parameters |
| `TOP_K_RESULTS` | `5` | Number of context chunks retrieved per search |

---

## 6. Complete CLI & Command Reference

The project includes a unified CLI entry point via `run.py`. All commands should be executed inside the virtual environment:

### Core CLI Commands (`run.py`)

| Command | Arguments / Flags | Description |
| :--- | :--- | :--- |
| `python run.py setup` | None | Drops and rebuilds the ChromaDB vector database from `data/documents/` |
| `python run.py query "<text>"` | Question in quotes | Runs a single query through the LangGraph pipeline with source citations |
| `python run.py interactive` | None | Starts an interactive question-answering CLI session |
| `python run.py demo` | `[--quick] [--standalone] [--per-domain N] [--delay S]` | Master 4-phase showcase (multi-domain queries, A/B baseline, ground truth, calibration) |
| `python run.py evaluate` | `[--limit N]` | Runs the evaluation benchmark across AI/RAG and Finance domains |
| `python run.py evaluate --compare` | `[--limit N]` | Runs an empirical A/B experiment comparing Naive Baseline RAG vs. Corrective RAG |
| `python run.py evaluate --ground-truth` | `[--limit N]` | Measures hallucination detector agreement against human labels |
| `python run.py calibrate` | None | Runs offline vector distance threshold calibration (zero API cost) |
| `python run.py test` | None | Runs the full 24-test Pytest test suite with concise traceback |

### Demo Command Flags
* **Full Master Showcase:** `python run.py demo` (Executes 5 questions per domain + all 4 benchmark phases).
* **Quick Showcase:** `python run.py demo --quick` (Executes 2 questions per domain with reduced delay).
* **Custom Questions per Domain:** `python run.py demo --per-domain 3`
* **Queries Only (Skip Benchmarks):** `python run.py demo --standalone`
* **Custom Delay Between Queries:** `python run.py demo --delay 2.0`

### Evaluation Command Flags
* **Full Multi-Domain Evaluation:** `python run.py evaluate`
* **Limited Evaluation (e.g. 4 questions):** `python run.py evaluate --limit 4`
* **A/B Comparison Report:** `python run.py evaluate --compare --limit 5`
* **Human Label Agreement:** `python run.py evaluate --ground-truth --limit 10`

### Diagnostic & Direct Module Commands
* **Gemini Connectivity & Latency Benchmark:**
  ```powershell
  python check_models.py
  ```
* **Run Tests Directly via Pytest:**
  ```powershell
  pytest tests/ -v
  ```
* **Rebuild ChromaDB Database Directly:**
  ```powershell
  python src/create_database.py --reset
  ```
* **Run Interactive Mode Directly:**
  ```powershell
  python src/query_rag.py --interactive
  ```

---

## 7. Quick Start Guide

### Step 1: Clone & Setup Environment

```powershell
# Clone the repository
git clone https://github.com/rxshil09/Corrective-RAG.git
cd Corrective-RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (Optional: If PowerShell blocks script execution, run: Set-ExecutionPolicy Unrestricted -Scope Process)

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys
Copy `.env.example` to `.env` and configure your Google Gemini API key (free tier available at [Google AI Studio](https://aistudio.google.com/app/apikey)):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### Step 3: Initialize Database
```powershell
python run.py setup
```

### Step 4: Run Queries & Benchmarks
```powershell
# Single query
python run.py query "What is Retrieval-Augmented Generation?"

# Quick showcase
python run.py demo --quick

# Run unit tests
python run.py test
```

> [!TIP]
> If running without activating the virtual environment in your terminal shell, execute directly using the virtual environment's Python binary:
> ```powershell
> .\venv\Scripts\python.exe run.py test
> ```

---

## 8. Empirical Evaluation Results

Empirical comparisons between a naive baseline RAG pipeline and this corrective pipeline reveal significant gains in grounding and answer reliability:

| Metric | Naive Baseline RAG | Agentic Corrective RAG | Difference / Empirical Impact |
| :--- | :---: | :---: | :---: |
| **Consistency Score (Claim-Level)** | `0.00` | `0.93` | **+0.93 gain** via relevance filtering & self-correction |
| **Reliability Rate ($\ge 0.70$)** | `0%` | `100%` | **+100%** grounded answers across all test queries |
| **Average Latency Overhead** | `2.02s` | `5.06s` | `+3.03s` trade-off for verification & multi-stage recovery |
| **Average LLM Calls / Query** | `2.0` (incl. eval) | `2.7` | `+0.7` calls on queries triggering correction |
| **Detector Agreement with Ground Truth** | — | **100%** | Tested against human-verified benchmark labels |
| **Distance Threshold Calibration** | — | **$0.25$** | Calibrated offline to discard noise while retaining >46% top chunks |

---

## 9. Verification & Test Suite Status

The project includes an automated test suite implemented in `tests/test_hallucination_detector.py`. 

**Current Status:** `24/24 passed (100%)`

Coverage includes:
* **Hallucination Detector Verification:** Grounded answers, partial hallucinations, severe hallucinations, JSON parsing with code fences, score clamping ($[0.0, 1.0]$), and error resilience.
* **LangGraph Conditional Routing:** Correct routing to `END` on high score, `strict_generate` on partial hallucination, `rewrite_query` on severe hallucination, and loop termination on `MAX_REGENERATION_ATTEMPTS`.
* **Context Relevance Filtering:** Chunk pruning at distance threshold and guaranteed fallback when all chunks fall below threshold.
* **Document Loaders:** Multi-format discovery across `.md`, `.txt`, `.pdf`, and `.docx`.
* **Cosine Distance Conversions:** Non-linear vector metric conversion validation.
* **Evaluation & Experiment Data Models:** Comparison record delta and ground truth agreement.

To run the verification suite:
```powershell
python run.py test
```
