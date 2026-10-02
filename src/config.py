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
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

LLM_MODEL: str = os.getenv("LLM_MODEL", "")

GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "gemini_primary": "gemini-3.5-flash-lite",
    "gemini_fallback": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
}


class ResilientChatCompletions:
    """Wraps chat completions with automatic multi-tier fallback (Gemini 3.5 -> Gemini 3.1 -> OpenAI)."""
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
            pname = pinfo["provider"]

            call_kwargs = dict(kwargs)
            call_kwargs["model"] = model

            try:
                response = client.chat.completions.create(**call_kwargs)
                return response
            except openai.BadRequestError as e:
                # If response_format is unsupported by this model/provider, retry without response_format
                if "response_format" in call_kwargs:
                    try:
                        no_rf_kwargs = dict(call_kwargs)
                        del no_rf_kwargs["response_format"]
                        response = client.chat.completions.create(**no_rf_kwargs)
                        return response
                    except Exception:
                        pass

                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] {pname.capitalize()} ({model}) error ({type(e).__name__}). "
                        f"Falling back to [bold cyan]{next_p['provider'].capitalize()} ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e
            except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] {pname.capitalize()} ({model}) error ({type(e).__name__}). "
                        f"Falling back to [bold cyan]{next_p['provider'].capitalize()} ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e
            except Exception as e:
                last_exception = e
                if i < len(self.chain) - 1:
                    next_p = self.chain[i + 1]
                    console.print(
                        f"\n[yellow][WARN] {pname.capitalize()} ({model}) failed ({str(e)[:50]}). "
                        f"Falling back to [bold cyan]{next_p['provider'].capitalize()} ({next_p['model']})[/bold cyan]...[/yellow]"
                    )
                    continue
                else:
                    raise e

        raise last_exception or RuntimeError("All LLM providers in fallback chain failed.")


class ResilientLLMClient:
    """Composite client providing .chat.completions.create with fallback."""
    def __init__(self, provider_chain):
        self.chat = type("Chat", (), {"completions": ResilientChatCompletions(provider_chain)})()
        self.provider_chain = provider_chain
        self.primary_provider = provider_chain[0]["provider"]
        self.primary_model = provider_chain[0]["model"]


def resolve_llm_chain():
    """
    Builds the priority-ordered provider fallback chain:
    1. Google Gemini 3.5 Flash-Lite (gemini-3.5-flash-lite)
    2. Google Gemini 3.1 Flash-Lite (gemini-3.1-flash-lite) [Fallback on 429/error]
    3. OpenAI (gpt-4o-mini) [Fallback]
    """
    provider_pref = LLM_PROVIDER.strip().strip("'").strip('"').lower()
    custom_model = LLM_MODEL.strip().strip("'").strip('"')

    gemini_key = GEMINI_API_KEY.strip().strip("'").strip('"')
    openai_key = OPENAI_API_KEY.strip().strip("'").strip('"')

    gemini_client = None
    if gemini_key and not gemini_key.startswith("your_"):
        gemini_client = openai.OpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL)

    openai_client = None
    if openai_key and not openai_key.startswith("your_"):
        openai_client = openai.OpenAI(api_key=openai_key)

    if not gemini_client and not openai_client:
        raise ValueError(
            "No valid API keys found in .env!\n"
            "Please configure GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
        )

    chain = []

    if provider_pref == "openai" and openai_client:
        # User explicitly preferred OpenAI
        chain.append({
            "client": openai_client,
            "model": custom_model or DEFAULT_MODELS["openai"],
            "provider": "openai",
        })
        if gemini_client:
            chain.append({
                "client": gemini_client,
                "model": DEFAULT_MODELS["gemini_primary"],
                "provider": "gemini",
            })
            chain.append({
                "client": gemini_client,
                "model": DEFAULT_MODELS["gemini_fallback"],
                "provider": "gemini",
            })
    else:
        # Default: Gemini 3.5 Flash-Lite -> Gemini 3.1 Flash-Lite -> OpenAI
        if gemini_client:
            primary_model = custom_model if (provider_pref in ("gemini", "auto") and custom_model) else DEFAULT_MODELS["gemini_primary"]
            chain.append({
                "client": gemini_client,
                "model": primary_model,
                "provider": "gemini",
            })

            # If primary isn't already the 3.1 fallback, add 3.1 flash lite as secondary fallback
            fallback_model = DEFAULT_MODELS["gemini_fallback"]
            if primary_model != fallback_model:
                chain.append({
                    "client": gemini_client,
                    "model": fallback_model,
                    "provider": "gemini",
                })

        if openai_client:
            chain.append({
                "client": openai_client,
                "model": DEFAULT_MODELS["openai"],
                "provider": "openai",
            })

    if not chain:
        raise ValueError("Could not assemble a valid LLM execution chain.")

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
        "chain_summary": " -> ".join(f"{p['provider']} ({p['model']})" for p in chain)
    }


def get_llm_client():
    """
    Creates and returns (resilient_client, primary_model_name, primary_provider_name).
    Automatically falls back across Gemini 3.5 Flash-Lite -> Gemini 3.1 Flash-Lite -> OpenAI.
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