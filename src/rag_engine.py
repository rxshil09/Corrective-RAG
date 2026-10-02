"""
rag_engine.py — The Hallucination-Aware RAG Engine (LangGraph Powered)

This is the heart of the system. It combines:
  1. Relevance-Filtered RAG Retrieval (prunes noisy chunks)
  2. LangGraph StateGraph Agentic Workflow
  3. Claim-Level Hallucination Detection Layer
  4. Dual-Path Self-Correction Loop:
     - Strict Negative-Constraint Regeneration (minor hallucinations)
     - Query Rewriting + Re-Retrieval (severe retrieval failures)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    get_llm_client,
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    TOP_K_RESULTS, MIN_RELEVANCE_SCORE, MIN_CHUNKS_REQUIRED,
    HALLUCINATION_THRESHOLD, STRICT_MODE_THRESHOLD, QUERY_REWRITE_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS,
    RAG_PROMPT_TEMPLATE, STRICT_RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT_TEMPLATE, distance_to_similarity
)
from hallucination_detector import HallucinationDetector, HallucinationReport
from rag_graph import RAGGraphBuilder, RAGGraphState

console = Console(legacy_windows=False)


@dataclass
class RAGResponse:
    """Complete response from the Hallucination-Aware RAG system."""
    
    query: str
    answer: str
    sources: List[str]
    consistency_score: float
    hallucination_report: Optional[HallucinationReport]
    regeneration_count: int
    used_strict_mode: bool
    used_query_rewrite: bool = False
    rewritten_query: Optional[str] = None
    retrieved_chunks: List[Document] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    all_attempts: List[str] = field(default_factory=list)
    execution_trace: List[str] = field(default_factory=list)
    llm_call_count: int = 0
    processing_time_s: float = 0.0
    
    @property
    def is_reliable(self) -> bool:
        return self.consistency_score >= HALLUCINATION_THRESHOLD
    
    @property
    def confidence_label(self) -> str:
        if self.consistency_score >= 0.85:
            return "HIGH"
        elif self.consistency_score >= HALLUCINATION_THRESHOLD:
            return "MEDIUM"
        elif self.consistency_score >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"


class HallucinationAwareRAG:
    """
    Complete Hallucination-Aware RAG system with LangGraph agentic self-correction.

    Usage:
        rag = HallucinationAwareRAG()
        response = rag.query("What is retrieval-augmented generation?")
        print(response.answer)
        print(f"Reliability: {response.consistency_score:.2f}")
    """

    def __init__(self):
        console.print("[dim]Initializing RAG engine...[/dim]")
        
        # Initialize LLM client (Gemini with multi-tier model fallback)
        self.client, self.model_name, self.provider = get_llm_client()
        console.print(f"[dim]Using LLM provider: [bold cyan]{self.provider}[/bold cyan] (Model: [bold green]{self.model_name}[/bold green])[/dim]")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Load vector database
        self.db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        
        # Initialize hallucination detector
        self.detector = HallucinationDetector()

        # Build and compile LangGraph StateGraph
        self.graph_builder = RAGGraphBuilder(self)
        self.graph = self.graph_builder.graph
        
        # Ready to serve requests
        console.print("[green][SUCCESS] RAG engine & LangGraph workflow ready[/green]")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, question: str, verbose: bool = True) -> RAGResponse:
        """
        Main entry point. Executes the LangGraph agentic workflow
        with relevance filtering, claim-level grading, and dynamic self-correction.
        """
        start_time = time.time()
        
        if verbose:
            console.print(Rule("[bold blue]Hallucination-Aware RAG Agentic Pipeline[/bold blue]"))
            console.print(f"[bold]Question:[/bold] {question}\n")

        initial_state: RAGGraphState = {
            "query": question,
            "search_query": question,
            "rewritten_query": None,
            "retrieved_chunks": [],
            "relevance_scores": [],
            "context_str": "",
            "sources": [],
            "answer": "",
            "consistency_score": 0.0,
            "hallucination_report": None,
            "attempt": 1,
            "regeneration_count": 0,
            "used_strict_mode": False,
            "used_query_rewrite": False,
            "all_attempts": [],
            "best_answer": "",
            "best_score": 0.0,
            "best_report": None,
            "execution_trace": [],
            "llm_call_count": 0,
            "verbose": verbose,
        }

        # Invoke the compiled LangGraph state graph
        final_state = self.graph.invoke(initial_state)

        elapsed = time.time() - start_time

        response = RAGResponse(
            query=question,
            answer=final_state.get("best_answer") or final_state.get("answer", ""),
            sources=final_state.get("sources", []),
            consistency_score=final_state.get("best_score", final_state.get("consistency_score", 0.0)),
            hallucination_report=final_state.get("best_report") or final_state.get("hallucination_report"),
            regeneration_count=final_state.get("regeneration_count", 0),
            used_strict_mode=final_state.get("used_strict_mode", False),
            used_query_rewrite=final_state.get("used_query_rewrite", False),
            rewritten_query=final_state.get("rewritten_query"),
            retrieved_chunks=final_state.get("retrieved_chunks", []),
            relevance_scores=final_state.get("relevance_scores", []),
            all_attempts=final_state.get("all_attempts", []),
            execution_trace=final_state.get("execution_trace", []),
            llm_call_count=final_state.get("llm_call_count", 0),
            processing_time_s=elapsed,
        )

        if verbose:
            self._print_final_response(response)
        
        return response

    def baseline_query(self, question: str, verbose: bool = False) -> RAGResponse:
        """
        Execute standard naive RAG baseline: Retrieve -> Generate.
        No relevance filtering, no hallucination detection, no self-correction.
        Used for empirical A/B comparison against the corrective pipeline.
        """
        start_time = time.time()
        
        if verbose:
            console.print(Rule("[bold yellow]Naive Baseline RAG Pipeline[/bold yellow]"))
            console.print(f"[bold]Question:[/bold] {question}\n")

        # Standard top-k retrieval without filtering
        results = self.db.similarity_search_with_score(question, k=TOP_K_RESULTS)
        chunks = [doc for doc, _ in results]
        scores = [distance_to_similarity(dist) for _, dist in results]
        context_str = self._format_context(chunks)
        sources = self._extract_sources(chunks)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=question
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[Error generating response: {e}]"

        elapsed = time.time() - start_time

        baseline_res = RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            consistency_score=0.0,
            hallucination_report=None,
            regeneration_count=0,
            used_strict_mode=False,
            used_query_rewrite=False,
            rewritten_query=None,
            retrieved_chunks=chunks,
            relevance_scores=scores,
            all_attempts=[answer],
            execution_trace=["baseline_retrieve", "baseline_generate"],
            llm_call_count=1,
            processing_time_s=elapsed,
        )

        if verbose:
            self._print_final_response(baseline_res)

        return baseline_res

    # ─────────────────────────────────────────────────────────────────────────
    # Helper & Standalone Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, verbose: bool = True) -> Tuple[List[Document], List[float]]:
        """Retrieve relevant chunks with relevance score filtering."""
        if verbose:
            console.print(f"[bold cyan][Docs] Retrieving Context (top {TOP_K_RESULTS} chunks)...[/bold cyan]")
        
        results = self.db.similarity_search_with_score(query, k=TOP_K_RESULTS)
        
        chunks = []
        scores = []
        filtered_chunks = []

        for i, (doc, dist) in enumerate(results):
            sim_score = distance_to_similarity(dist)
            scores.append(sim_score)
            chunks.append(doc)
            src = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))

            is_relevant = sim_score >= MIN_RELEVANCE_SCORE
            status_tag = "ACCEPTED" if is_relevant else "FILTERED (low relevance)"
            status_color = "green" if is_relevant else "dim red"

            if verbose:
                console.print(f"   [{status_color}][{i+1}] {src} — sim: {sim_score:.3f} [{status_tag}][/{status_color}]")

            if is_relevant:
                filtered_chunks.append(doc)

        if len(filtered_chunks) < MIN_CHUNKS_REQUIRED:
            filtered_chunks = chunks[:max(MIN_CHUNKS_REQUIRED, 1)]

        return filtered_chunks, scores

    def _format_context(self, chunks: List[Document]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(chunks):
            src = chunk.metadata.get("source_name", chunk.metadata.get("source", "unknown"))
            parts.append(f"[Source {i+1}: {src}]\n{chunk.page_content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: List[Document]) -> List[str]:
        """Extract unique source names from chunks."""
        seen = set()
        sources = []
        for chunk in chunks:
            src = chunk.metadata.get("source_name", chunk.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def _rewrite_query(self, query: str, hallucinated_claims: Optional[List[str]] = None) -> str:
        """Reformulate user query for higher semantic retrieval recall."""
        claims_str = "\n".join(f"- {c}" for c in hallucinated_claims) if hallucinated_claims else "General retrieval mismatch / low confidence."
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(question=query, hallucinated_claims=claims_str)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        except Exception:
            return query

    def _generate(
        self,
        question: str,
        context: str,
        strict: bool = False,
        hallucinated_claims: Optional[List[str]] = None
    ) -> str:
        """Generate an answer using Gemini with the appropriate prompt template."""
        if strict:
            prompt = STRICT_RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
            if hallucinated_claims:
                claims_str = "\n".join(f"  - {c}" for c in hallucinated_claims)
                prompt += f"\n\nWARNING: The following claims from the previous answer were flagged as HALLUCINATIONS. Do NOT include them:\n{claims_str}"
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2 if not strict else 0.05,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating response: {e}]"

    def _print_final_response(self, response: RAGResponse):
        """Print the final formatted response."""
        color = response.hallucination_report.verdict_color if response.hallucination_report else "white"
        verdict = response.hallucination_report.verdict if response.hallucination_report else "UNKNOWN"
        
        console.print(Rule("[bold green]Final Response[/bold green]"))
        console.print(Panel(
            response.answer,
            title=f"[bold green]Final Answer[/bold green]",
            border_style="green"
        ))
        
        # Source citations panel
        if response.sources:
            citation_lines = [f"  {i}. {src}" for i, src in enumerate(response.sources, 1)]
            console.print(Panel(
                "\n".join(citation_lines),
                title="[bold cyan]📄 Sources[/bold cyan]",
                border_style="cyan"
            ))
        
        # Metadata panel
        meta_lines = [
            f"[{color}]Hallucination Check: {verdict}[/{color}]",
            f"Consistency Score:  [bold]{response.consistency_score:.2f}[/bold] / 1.00",
            f"Confidence Level:   [bold]{response.confidence_label}[/bold]",
            f"Regenerations:      {response.regeneration_count}",
            f"Strict Mode Used:   {'Yes' if response.used_strict_mode else 'No'}",
            f"Query Rewrite Used: {'Yes (' + str(response.rewritten_query) + ')' if response.used_query_rewrite else 'No'}",
            f"LLM Calls:          {response.llm_call_count}",
            f"Sources Used:       {', '.join(response.sources) if response.sources else 'None'}",
            f"Workflow Trace:     {' -> '.join(response.execution_trace) if response.execution_trace else 'Direct'}",
            f"Processing Time:    {response.processing_time_s:.2f}s",
        ]
        console.print(Panel("\n".join(meta_lines), title="[Info] Response Metadata", border_style="dim"))