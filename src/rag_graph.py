"""
rag_graph.py — LangGraph StateGraph Architecture for Hallucination-Aware RAG

Implements the multi-stage, stateful agentic workflow:
  1. [retrieve] with Relevance Filtering
  2. [generate] with Standard Prompting
  3. [detect_hallucination] with Claim-Level Consistency Evaluation
  4. Conditional Routing:
     - Score >= HALLUCINATION_THRESHOLD -> END (Pass)
     - Score < QUERY_REWRITE_THRESHOLD -> [rewrite_query] -> [retrieve] -> [generate]
     - Else -> [strict_generate] -> [detect_hallucination]
  5. Attempt bounds (MAX_REGENERATION_ATTEMPTS) ensure guaranteed termination.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, TypedDict, Dict, Any

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TOP_K_RESULTS, MIN_RELEVANCE_SCORE, MIN_CHUNKS_REQUIRED,
    HALLUCINATION_THRESHOLD, STRICT_MODE_THRESHOLD, QUERY_REWRITE_THRESHOLD,
    MAX_REGENERATION_ATTEMPTS, RAG_PROMPT_TEMPLATE, STRICT_RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT_TEMPLATE, distance_to_similarity
)
from hallucination_detector import HallucinationDetector, HallucinationReport

console = Console(legacy_windows=False)


class RAGGraphState(TypedDict):
    """Full execution state passed through the LangGraph StateGraph."""
    query: str
    search_query: str
    rewritten_query: Optional[str]
    retrieved_chunks: List[Document]
    relevance_scores: List[float]
    context_str: str
    sources: List[str]
    answer: str
    consistency_score: float
    hallucination_report: Optional[HallucinationReport]
    attempt: int
    regeneration_count: int
    used_strict_mode: bool
    used_query_rewrite: bool
    all_attempts: List[str]
    best_answer: str
    best_score: float
    best_report: Optional[HallucinationReport]
    execution_trace: List[str]
    llm_call_count: int
    verbose: bool


class RAGGraphBuilder:
    """Constructs and compiles the LangGraph StateGraph for the RAG engine."""

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.detector = rag_engine.detector
        self.db = rag_engine.db
        self.client = rag_engine.client
        self.model_name = rag_engine.model_name
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGGraphState)

        # Add Nodes
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("detect_hallucination", self.node_detect_hallucination)
        workflow.add_node("strict_generate", self.node_strict_generate)
        workflow.add_node("rewrite_query", self.node_rewrite_query)

        # Add Edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "detect_hallucination")

        # Conditional Routing from detection
        workflow.add_conditional_edges(
            "detect_hallucination",
            self.route_after_detection,
            {
                "pass": END,
                "strict_generate": "strict_generate",
                "rewrite_query": "rewrite_query",
                "end_max_attempts": END,
            }
        )

        # Loop edges
        workflow.add_edge("strict_generate", "detect_hallucination")
        workflow.add_edge("rewrite_query", "retrieve")

        return workflow.compile()

    # ── Node Definitions ─────────────────────────────────────────────────────

    def node_retrieve(self, state: RAGGraphState) -> Dict[str, Any]:
        """Retrieve chunks from ChromaDB with relevance score filtering."""
        query_to_search = state.get("search_query") or state["query"]
        verbose = state.get("verbose", True)
        trace = list(state.get("execution_trace", []))
        trace.append(f"retrieve('{query_to_search[:30]}...')")

        if verbose:
            console.print(f"\n[bold cyan][Docs] Node 'retrieve': Querying ChromaDB (Top {TOP_K_RESULTS})...[/bold cyan]")
            if query_to_search != state["query"]:
                console.print(f"   [dim]Using rewritten query:[/dim] [cyan]{query_to_search}[/cyan]")

        raw_results = self.db.similarity_search_with_score(query_to_search, k=TOP_K_RESULTS)

        chunks = []
        scores = []
        filtered_chunks = []

        for i, (doc, dist) in enumerate(raw_results):
            # Convert distance to similarity using the documented transformation
            sim_score = distance_to_similarity(dist)
            src = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
            scores.append(sim_score)
            chunks.append(doc)

            is_relevant = sim_score >= MIN_RELEVANCE_SCORE
            status_color = "green" if is_relevant else "dim red"
            status_tag = "ACCEPTED" if is_relevant else "FILTERED (low relevance)"

            if verbose:
                console.print(f"   [{status_color}][{i+1}] {src} — sim: {sim_score:.3f} [{status_tag}][/{status_color}]")

            if is_relevant:
                filtered_chunks.append(doc)

        # Fallback if too few chunks pass the relevance filter
        if len(filtered_chunks) < MIN_CHUNKS_REQUIRED:
            if verbose:
                console.print(f"   [yellow][WARN] Only {len(filtered_chunks)} chunk(s) passed threshold {MIN_RELEVANCE_SCORE:.2f}. Falling back to top {max(MIN_CHUNKS_REQUIRED, 1)} chunks.[/yellow]")
            filtered_chunks = chunks[:max(MIN_CHUNKS_REQUIRED, 1)]

        # Format context
        parts = []
        for idx, chunk in enumerate(filtered_chunks):
            src = chunk.metadata.get("source_name", chunk.metadata.get("source", "unknown"))
            parts.append(f"[Source {idx+1}: {src}]\n{chunk.page_content}")
        context_str = "\n\n---\n\n".join(parts)

        # Extract source names
        seen = set()
        sources = []
        for chunk in filtered_chunks:
            src = chunk.metadata.get("source_name", chunk.metadata.get("source", "unknown"))
            if src not in seen:
                seen.add(src)
                sources.append(src)

        return {
            "retrieved_chunks": filtered_chunks,
            "relevance_scores": scores,
            "context_str": context_str,
            "sources": sources,
            "execution_trace": trace,
        }

    def node_generate(self, state: RAGGraphState) -> Dict[str, Any]:
        """Generate answer from retrieved context using standard prompting."""
        verbose = state.get("verbose", True)
        trace = list(state.get("execution_trace", []))
        trace.append("generate")

        if verbose:
            console.print(f"\n[bold cyan][AI] Node 'generate': Drafting Answer...[/bold cyan]")

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=state["context_str"],
            question=state["query"]
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

        if verbose:
            console.print(Panel(answer, title="[bold]Initial Answer[/bold]", border_style="blue"))

        all_attempts = list(state.get("all_attempts", []))
        all_attempts.append(answer)

        return {
            "answer": answer,
            "all_attempts": all_attempts,
            "best_answer": state.get("best_answer") or answer,
            "best_score": state.get("best_score", 0.0),
            "execution_trace": trace,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    def node_detect_hallucination(self, state: RAGGraphState) -> Dict[str, Any]:
        """Inspect current candidate answer against context for hallucinations."""
        attempt = state.get("attempt", 1)
        verbose = state.get("verbose", True)
        trace = list(state.get("execution_trace", []))
        trace.append(f"detect(attempt={attempt})")

        report = self.detector.detect(
            answer=state["answer"],
            context=state["context_str"],
            attempt=attempt
        )

        best_score = state.get("best_score", 0.0)
        best_answer = state.get("best_answer", state["answer"])
        best_report = state.get("best_report", report)

        if state.get("best_report") is None or report.consistency_score >= best_score:
            best_score = report.consistency_score
            best_answer = state["answer"]
            best_report = report

        return {
            "consistency_score": report.consistency_score,
            "hallucination_report": report,
            "best_score": best_score,
            "best_answer": best_answer,
            "best_report": best_report,
            "execution_trace": trace,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    def node_strict_generate(self, state: RAGGraphState) -> Dict[str, Any]:
        """Regenerate with strict constraints and negative claim injection."""
        attempt = state.get("attempt", 1) + 1
        regen_count = state.get("regeneration_count", 0) + 1
        verbose = state.get("verbose", True)
        report = state.get("hallucination_report")
        hallucinated_claims = report.hallucinated_claims if report else []

        trace = list(state.get("execution_trace", []))
        trace.append(f"strict_generate(attempt={attempt})")

        if verbose:
            console.print(f"\n[yellow][Retry] Node 'strict_generate': Regenerating with strict negative constraints (attempt {attempt}/{MAX_REGENERATION_ATTEMPTS})[/yellow]")

        prompt = STRICT_RAG_PROMPT_TEMPLATE.format(
            context=state["context_str"],
            question=state["query"]
        )
        if hallucinated_claims:
            claims_str = "\n".join(f"  - {c}" for c in hallucinated_claims)
            prompt += f"\n\nWARNING: The following claims from the previous answer were flagged as HALLUCINATIONS. Do NOT include them:\n{claims_str}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[Error regenerating response: {e}]"

        if verbose:
            console.print(Panel(
                answer,
                title=f"[bold]Regenerated Answer (attempt {attempt})[/bold]",
                border_style="yellow"
            ))

        all_attempts = list(state.get("all_attempts", []))
        all_attempts.append(answer)

        return {
            "answer": answer,
            "attempt": attempt,
            "regeneration_count": regen_count,
            "used_strict_mode": True,
            "all_attempts": all_attempts,
            "execution_trace": trace,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    def node_rewrite_query(self, state: RAGGraphState) -> Dict[str, Any]:
        """Rewrite user query for better semantic vector retrieval."""
        attempt = state.get("attempt", 1) + 1
        regen_count = state.get("regeneration_count", 0) + 1
        verbose = state.get("verbose", True)
        report = state.get("hallucination_report")
        hallucinated_claims = report.hallucinated_claims if report else []

        trace = list(state.get("execution_trace", []))
        trace.append("rewrite_query")

        claims_str = "\n".join(f"- {c}" for c in hallucinated_claims) if hallucinated_claims else "General retrieval mismatch / low confidence."
        prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(
            question=state["query"],
            hallucinated_claims=claims_str
        )

        if verbose:
            console.print(f"\n[bold magenta][Rewrite] Node 'rewrite_query': Reformulating query due to low consistency ({state.get('consistency_score', 0):.2f} < {QUERY_REWRITE_THRESHOLD:.2f})...[/bold magenta]")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            rewritten = response.choices[0].message.content.strip().replace('"', '').replace("'", "")
        except Exception as e:
            rewritten = state["query"]

        if verbose:
            console.print(f"   [green]➜ Rewritten Query:[/green] [bold cyan]\"{rewritten}\"[/bold cyan]")

        return {
            "search_query": rewritten,
            "rewritten_query": rewritten,
            "attempt": attempt,
            "regeneration_count": regen_count,
            "used_query_rewrite": True,
            "execution_trace": trace,
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    # ── Conditional Router ───────────────────────────────────────────────────

    def route_after_detection(self, state: RAGGraphState) -> str:
        """Determines the next edge after hallucination detection."""
        report = state.get("hallucination_report")
        score = state.get("consistency_score", 0.0)
        attempt = state.get("attempt", 1)
        used_rewrite = state.get("used_query_rewrite", False)
        verbose = state.get("verbose", True)

        # 1. If grounded and verified -> PASS
        if (report and not report.has_hallucination) or score >= HALLUCINATION_THRESHOLD:
            if verbose:
                console.print(f"\n[green bold][PASS] Answer verified (score: {score:.2f} >= {HALLUCINATION_THRESHOLD:.2f}) -> Graph Completed[/green bold]")
            return "pass"

        # 2. If max attempts reached -> END
        if attempt >= MAX_REGENERATION_ATTEMPTS:
            if verbose:
                console.print(f"\n[yellow][WARN] Max attempts ({MAX_REGENERATION_ATTEMPTS}) reached -> Returning best candidate (score: {state.get('best_score', 0):.2f})[/yellow]")
            return "end_max_attempts"

        # 3. If score is severely low (< QUERY_REWRITE_THRESHOLD) and query rewrite hasn't been used yet
        if score < QUERY_REWRITE_THRESHOLD and not used_rewrite:
            return "rewrite_query"

        # 4. Otherwise use strict constrained generation
        return "strict_generate"
