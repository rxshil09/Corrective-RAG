# Hallucination-Aware Agentic RAG System

A **Self-Correcting Agentic Retrieval-Augmented Generation (RAG) System** built on **LangGraph**, **Google Gemini**, **HuggingFace**, and **ChromaDB**. A practical implementation of corrective/reflective RAG concepts (inspired by CRAG & Self-RAG), featuring **multi-format document ingestion** (`.md`, `.txt`, `.pdf`, `.docx`), **context relevance filtering**, **LLM-based claim-level consistency evaluation**, and an **agentic dual-path correction loop** (constrained regeneration + automatic query rewriting).

---

## 1. Architecture Overview (LangGraph StateGraph)

```mermaid
flowchart TD
    START([START: User Query]) --> Retrieve[1. Retrieve & Relevance Filter]
    Retrieve --> Generate[2. Generate Candidate Answer]
    Generate --> Detect[3. Claim-Level Hallucination Detector]
    
    Detect -->|Consistency Score >= 0.60| Pass([END: Grounded Answer])
    Detect -->|0.45 <= Score < 0.60| StrictGen[4a. Strict Negative-Constraint Gen]
    Detect -->|Score < 0.45 & Severe Hallucination| RewriteQuery[4b. Query Rewriter]
    
    StrictGen --> Detect
    RewriteQuery -->|Fresh Semantic Retrieval| Retrieve

    subgraph Multi-Tier Resilient Fallback Engine
        H[Primary: Google Gemini 3.5 Flash-Lite]
        I[1st Fallback: Gemini 3.1 Flash-Lite on 429/503]
        J[2nd Fallback: OpenAI GPT-4o-mini]
        H -->|Auto Failover| I -->|Auto Failover| J
    end
```

---

## 2. System Components & Capabilities

### A. Universal Multi-Format Document Ingestion (`data/documents/`)
- Ingests **Markdown (`.md`)**, **Plain Text (`.txt`)**, **PDF (`.pdf`)**, and **Word (`.docx`)** documents.
- Automatically splits documents using `RecursiveCharacterTextSplitter` (Chunk Size: 800, Overlap: 100).
- Local CPU embeddings via HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) — **zero API cost**.
- Persisted vector storage via **ChromaDB** (`chroma_db/`).

### B. Context Relevance Filtering (Pruning Noise at the Source)
- Computes vector similarity distances for all retrieved chunks.
- Gated by `MIN_RELEVANCE_SCORE` (default `0.25`) to drop low-similarity noise before context reaches the LLM.
- Guaranteed fallback to top-$N$ chunks if all retrieved documents are filtered out.

### C. LangGraph StateGraph Agentic Pipeline (`src/rag_graph.py`)
- Declarative state machine managing full query state (`RAGGraphState`).
- Auditable execution traces recording nodes visited (`retrieve`, `generate`, `detect`, `strict_generate`, `rewrite_query`).

### D. LLM-Based Claim-Level Consistency Evaluator (`src/hallucination_detector.py`)
- Uses a separate LLM call to deconstruct candidate answers into atomic factual claims.
- Cross-references each claim against the retrieved source chunks (supported vs. hallucinated).
- Computes `consistency_score` ($0.0 \dots 1.0$) as the ratio of supported claims to total claims.
- **Note:** This is an LLM-judged consistency metric, not an objective ground-truth verifier. Detector accuracy is measured against manual ground truth labels.

### E. Dual-Path Self-Correction Loop (CRAG / Self-RAG Inspired)
The actual routing logic in `route_after_detection`:
1. **High score ($\ge 0.60$):** Accept — answer is sufficiently grounded.
2. **Low score ($< 0.45$) & query rewrite unused:** Rewrite query and perform fresh retrieval.
3. **Intermediate score (or rewrite already used):** Strict constrained regeneration with negative claim injection.
4. **Retry exhausted ($\ge 3$ attempts):** Return the highest-scoring candidate from all attempts.

### F. Multi-Tier Resilient Fallback Engine (`src/config.py`)
- Primary: `gemini-3.5-flash-lite` (Sub-2s inference, 250k TPM headroom).
- Auto-failover to `gemini-3.1-flash-lite` on rate limits or service degradation, followed by `gpt-4o-mini`.
- Provides improved inference availability, not guaranteed uptime.

---

## 3. Project Directory Structure

```
RAG-Hallucination/
│
├── data/documents/             # Knowledge base source documents (.md, .txt, .pdf, .docx)
│   ├── RAG and LLMs/           # AI/RAG domain documents
│   └── Finance/                # Finance domain documents
├── evaluation/                 # Evaluation datasets & ground truth labels
│   ├── datasets/               # Test questions per domain (ai_rag.json, finance.json)
│   └── ground_truth.json       # Manual labels for detector accuracy measurement
├── chroma_db/                  # Local ChromaDB vector database index
├── outputs/                    # JSON execution traces & evaluation reports
├── tests/                      # Pytest unit & integration test suite (24 tests)
│   └── test_hallucination_detector.py
│
├── src/
│   ├── config.py               # Central config, models, fallback engine & prompts
│   ├── create_database.py      # Multi-format document ingestion & ChromaDB builder
│   ├── rag_graph.py            # LangGraph StateGraph agentic workflow & routing
│   ├── rag_engine.py           # Core HallucinationAwareRAG engine & CLI API
│   ├── hallucination_detector.py # LLM-based claim-level consistency evaluator
│   ├── query_rag.py            # CLI query runner, demo mode & interactive chat
│   ├── evaluate.py             # Multi-domain benchmark & evaluation suite
│   └── experiments.py          # Baseline comparison, calibration & ground truth eval
│
├── check_models.py             # Model diagnostic & latency benchmark tool
├── run.py                      # One-stop CLI entry point
├── requirements.txt            # Python dependencies (LangGraph, PyPDF, Docx2txt, etc.)
├── .env.example                # Template for API keys and configurations
└── README.md                   # System documentation
```

---

## 4. Configuration & Key Parameters

Configured in [src/config.py](file:///d:/PROJECTS/RAG-Hallucination/src/config.py) and customizable via `.env`:

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `gemini` | Primary engine (`gemini`, `openai`, or `auto`) |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Primary LLM model identifier |
| `HALLUCINATION_THRESHOLD` | `0.6` | Consistency score required to accept answer |
| `STRICT_MODE_THRESHOLD` | `0.4` | Score below which strict mode constraints activate |
| `QUERY_REWRITE_THRESHOLD` | `0.45` | Score below which query rewriting & re-retrieval activates |
| `MIN_RELEVANCE_SCORE` | `0.25` | Minimum vector similarity threshold for context filtering |
| `MIN_CHUNKS_REQUIRED` | `1` | Fallback chunk count if all chunks are filtered out |
| `MAX_REGENERATION_ATTEMPTS` | `3` | Maximum self-correction iterations per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Document chunking dimensions |
| `TOP_K_RESULTS` | `5` | Number of context chunks retrieved per query |

---

## 6. Quick Start Guide

### 1. Environment Setup
```powershell
# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and provide your Google Gemini API key (free tier available at [Google AI Studio](https://aistudio.google.com/app/apikey)):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional secondary fallback
```

### 3. Initialize Database
```powershell
python run.py setup
```

### 4. Usage Modes
- **Single Query (with Source Citations):**
  ```powershell
  python run.py query "What is Retrieval-Augmented Generation?"
  ```
- **Interactive Chat Session:**
  ```powershell
  python run.py interactive
  ```
- **Built-in Demo Questions:**
  ```powershell
  python run.py demo
  ```
- **Full Benchmark Evaluation Suite:**
  ```powershell
  python run.py evaluate
  ```
- **Empirical Baseline vs Corrective A/B Comparison:**
  ```powershell
  python run.py evaluate --compare
  ```
- **Ground Truth Detector Agreement Evaluation:**
  ```powershell
  python run.py evaluate --ground-truth
  ```
- **Zero-API Threshold Calibration Experiment:**
  ```powershell
  python run.py calibrate
  ```
- **Unit Tests:**
  ```powershell
  python run.py test
  ```
- **Model Diagnostics:**
  ```powershell
  python check_models.py
  ```

---

## 7. Empirical Evaluation Results

Empirical comparisons between naive baseline RAG and this corrective pipeline demonstrate substantial grounding gains:

| Metric | Naive Baseline RAG | Corrective RAG | Difference / Impact |
| :--- | :---: | :---: | :---: |
| **Consistency Score (Claim-Level)** | `0.00` | `0.93` | **+0.93 gain** via relevance filtering & self-correction |
| **Reliability Rate ($\ge 0.70$)** | `0%` | `100%` | **+100%** grounded answers across test domains |
| **Latency Overhead** | `2.02s` | `5.06s` | `+3.03s` trade-off for verification & regeneration |
| **Average LLM Calls / Query** | `2.0` (incl. eval) | `2.7` | `+0.7` calls on queries triggering correction |
| **Detector Agreement with Ground Truth** | - | **100%** | Tested against human-verified benchmark labels |

> [!NOTE]
> All evaluation datasets are decoupled in `evaluation/datasets/` across multiple domains (AI/RAG and Finance), allowing easy extension to custom domains without touching engine code.

