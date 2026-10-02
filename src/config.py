"""
config.py — Central configuration for the Hallucination-Aware RAG system
"""

import sys
import os
import openai
from dotenv import load_dotenv

# Ensure UTF-8 output encoding across Windows consoles to prevent charmap UnicodeEncodeErrors
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# ── LLM Provider Configuration ──────────────────────────────────────────────
LLM_PROVIDER: str = "gemini"

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

LLM_MODEL: str = os.getenv("LLM_MODEL", "")

GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Verified active Gemini models in priority fallback order:
# 3.5 flash lite -> 3.1 flash lite -> 3.8 flash -> 3.7 flash -> 3.6 flash -> 3.5 flash
# Note: gemini-2.5-flash-lite and gemini-2.5-flash were tested directly against Google's API
# and returned HTTP 404 (officially discontinued for new users by Google).
GEMINI_MODELS_FALLBACK_ORDER = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]


class ResilientChatCompletions:
    """Wraps chat completions with automatic multi-tier fallback across Gemini models."""
    def __init__(self, provider_chain):
        self.chain = provider_chain

    def create(self, **kwargs):
        import openai
        from rich.console import Console
        console = Console(legacy_windows=False)

        last_exception = None
        for i, pinfo in enumerate(self.chain):
            client = pinfo["client"]
            model = pinfo["model"]

            call_kwargs = dict(kwargs)
            call_kwargs["model"] = model

            try:
                response = client.chat.completions.create(**call_kwargs)
                # Ensure choices and content exist
                if not response.choices or response.choices[0].message.content is None:
                    raise ValueError(f"Model {model} returned empty content.")
                return response
            except openai.BadRequestError as e:
                # If response_format is unsupported by this model, retry without response_format
                if "response_format" in call_kwargs:
                    try:
                        no_rf_kwargs = dict(call_kwargs)
                        del no_rf_kwargs["response_format"]
                        response = client.chat.completions.create(**no_rf_kwargs)
                        if response.choices and response.choices[0].message.content is not None:
                            return response
                    except Exception:
                        pass

                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] Gemini ({model}) error ({type(e).__name__}). "
                        f"Falling back to [bold cyan]Gemini ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e
            except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] Gemini ({model}) error ({type(e).__name__}). "
                        f"Falling back to [bold cyan]Gemini ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e
            except Exception as e:
                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] Gemini ({model}) failed ({str(e)[:50]}). "
                        f"Falling back to [bold cyan]Gemini ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e

        raise last_exception or RuntimeError("All Gemini models in fallback chain failed.")


class ResilientLLMClient:
    """Composite client providing .chat.completions.create with fallback."""
    def __init__(self, provider_chain):
        self.chat = type("Chat", (), {"completions": ResilientChatCompletions(provider_chain)})()
        self.provider_chain = provider_chain
        self.primary_provider = provider_chain[0]["provider"]
        self.primary_model = provider_chain[0]["model"]


def resolve_llm_chain():
    """
    Builds the priority-ordered Gemini model fallback chain:
    1. gemini-3.5-flash-lite (Primary: ultra-fast, high-quota, grounded)
    2. gemini-3.1-flash-lite
    3. gemini-3.8-flash
    4. gemini-3.7-flash
    5. gemini-3.6-flash
    6. gemini-3.5-flash
    """
    custom_model = LLM_MODEL.strip().strip("'").strip('"')
    gemini_key = GEMINI_API_KEY.strip().strip("'").strip('"')

    if not gemini_key or gemini_key.startswith("your_"):
        raise ValueError(
            "GEMINI_API_KEY not found or unconfigured in .env!\n"
            "Please configure GEMINI_API_KEY in your .env file."
        )

    gemini_client = openai.OpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL)

    # Determine ordered list of models
    models_to_use = list(GEMINI_MODELS_FALLBACK_ORDER)
    if custom_model:
        if custom_model in models_to_use:
            models_to_use.remove(custom_model)
        models_to_use.insert(0, custom_model)

    chain = [
        {
            "client": gemini_client,
            "model": m,
            "provider": "gemini",
        }
        for m in models_to_use
    ]

    return chain


def resolve_llm_config():
    """Returns primary provider configuration for info/diagnostics."""
    chain = resolve_llm_chain()
    primary = chain[0]
    return {
        "provider": primary["provider"],
        "model": primary["model"],
        "api_key": "***",
        "fallback_count": len(chain) - 1,
        "chain_summary": " -> ".join(p["model"] for p in chain)
    }


def get_llm_client():
    """
    Creates and returns (resilient_client, primary_model_name, primary_provider_name).
    Automatically falls back across Gemini models:
    gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemini-3.8-flash -> gemini-3.7-flash -> gemini-3.6-flash -> gemini-3.5-flash.
    """
    chain = resolve_llm_chain()
    client = ResilientLLMClient(chain)
    return client, client.primary_model, client.primary_provider

# ── Paths ───────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
OUTPUTS_PATH: str = "./outputs"

# ── Hallucination Detection & Routing ───────────────────────────────────────
HALLUCINATION_THRESHOLD: float = float(os.getenv("HALLUCINATION_THRESHOLD", "0.6"))
STRICT_MODE_THRESHOLD: float = float(os.getenv("STRICT_MODE_THRESHOLD", "0.4"))
QUERY_REWRITE_THRESHOLD: float = float(os.getenv("QUERY_REWRITE_THRESHOLD", "0.45"))
MAX_REGENERATION_ATTEMPTS: int = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "3"))

# ── Chunking & Retrieval Relevance ──────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))  # Min (1 - distance) similarity
MIN_CHUNKS_REQUIRED: int = int(os.getenv("MIN_CHUNKS_REQUIRED", "1"))


def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """Convert ChromaDB distance to similarity score.

    For cosine distance with normalized vectors (default ChromaDB metric):
        similarity ≈ 1 - distance

    This transformation is only valid for the cosine distance metric.
    The assumption is made explicit here rather than buried in retrieval logic.

    Args:
        distance: Raw distance value from ChromaDB (lower = more similar).
        metric: Distance metric used by the vector store. Currently only 'cosine' is supported.

    Returns:
        Similarity score clamped to [0.0, 1.0].
    """
    if metric != "cosine":
        raise ValueError(f"Unsupported distance metric '{metric}'. Only 'cosine' is currently supported.")
    return max(0.0, min(1.0, 1.0 - distance))

# ── Embedding model (local, no API key needed) ───────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── ChromaDB collection name ─────────────────────────────────────────────────
COLLECTION_NAME: str = "hallucination_aware_rag"

# ── Prompts ──────────────────────────────────────────────────────────────────

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the context provided below to answer the user's question. If the context is missing details, do your best to provide a comprehensive answer by combining the context with your own knowledge.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

STRICT_RAG_PROMPT_TEMPLATE = """
You are an extremely precise and careful assistant. A previous answer was flagged for potential hallucination.
You MUST now generate a NEW answer following these STRICT rules:

STRICT RULES:
1. ONLY use information explicitly stated in the context below.
2. Every factual claim MUST be directly traceable to the context.
3. If a piece of information is not in the context, do NOT include it.
4. Do NOT infer, extrapolate, or assume anything beyond what is written.
5. If the context is insufficient, explicitly state: "The provided context does not contain enough information to fully answer this question."
6. Start each key fact with [FROM CONTEXT] to show grounding.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Strictly Grounded Answer:"""

HALLUCINATION_CHECK_PROMPT_TEMPLATE = """
You are a hallucination detection expert. Your task is to evaluate whether an AI-generated answer is consistent with the provided context.

--- CONTEXT ---
{context}
--- END CONTEXT ---

--- GENERATED ANSWER ---
{answer}
--- END ANSWER ---

Analyze the answer for hallucinations. A hallucination is any claim in the answer that:
1. Contradicts information in the context
2. Adds specific facts not present in the context
3. Misrepresents what the context says

INSTRUCTIONS:
1. Extract all distinct claims made in the generated answer.
2. For each claim, determine if it is 'supported' by the context or 'hallucinated'.
3. Calculate a partial consistency_score as the ratio: (number of supported claims) / (total number of claims). For example, if there are 3 supported claims and 1 hallucinated claim, the score should be 0.75. Do NOT simply output 0.0 or 1.0 unless it is 100% hallucinated or 100% supported.

Respond ONLY in the following JSON format (no other text):
{{
  "consistency_score": <float between 0.0 and 1.0 based on the ratio>,
  "has_hallucination": <true or false>,
  "hallucinated_claims": [<list of specific claims that are hallucinated>],
  "supported_claims": [<list of claims that ARE supported by context>],
  "reasoning": "<brief explanation including the claim counts and how the score was calculated>"
}}
"""

QUERY_REWRITE_PROMPT_TEMPLATE = """
You are an expert query reformulation assistant in a self-correcting RAG pipeline.
The initial retrieval for the question below resulted in unsupported or hallucinated claims.

--- ORIGINAL QUESTION ---
{question}

--- FLAGGED ISSUES / CLAIMS ---
{hallucinated_claims}

TASK:
Rewrite the question into a clear, precise, and keyword-rich search query optimized for vector document retrieval.
- Include essential technical keywords and terminology.
- Strip ambiguous phrasing, conversational filler, or assumptions.
- Output ONLY the rewritten query text. Do NOT add quotes, preamble, or markdown formatting.

Rewritten Search Query:"""