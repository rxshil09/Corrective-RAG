"""
evaluate.py — Evaluation Module for Hallucination-Aware RAG

Measures system performance on:
  - Hallucination Detection Accuracy
  - Self-Correction Effectiveness
  - Answer Quality (Faithfulness, Relevance)
  - Comparison: RAG vs. Hallucination-Aware RAG

Usage:
    python evaluate.py
    python evaluate.py --questions custom_questions.json
"""

import sys
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import HallucinationAwareRAG, RAGResponse
from config import OUTPUTS_PATH, HALLUCINATION_THRESHOLD

console = Console(legacy_windows=False)


# ── Evaluation Dataset Loader ────────────────────────────────────────────────

EVALUATION_DATASETS_DIR = Path(__file__).parent.parent / "evaluation" / "datasets"


def load_evaluation_datasets(datasets_dir: Path = None) -> List[Dict]:
    """Load all evaluation question datasets from the evaluation/datasets/ directory.
    
    Discovers and merges all .json files in the datasets directory,
    supporting multiple domains (AI/RAG, Finance, etc.).
    
    Returns:
        List of question dicts with unified schema.
    """
    datasets_dir = datasets_dir or EVALUATION_DATASETS_DIR
    all_questions = []
    
    if not datasets_dir.exists():
        console.print(f"[yellow][WARN] Datasets directory not found: {datasets_dir}[/yellow]")
        return all_questions
    
    json_files = sorted(datasets_dir.glob("*.json"))
    if not json_files:
        console.print(f"[yellow][WARN] No dataset files found in {datasets_dir}[/yellow]")
        return all_questions
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                questions = json.load(f)
            domain = json_file.stem  # e.g. 'ai_rag', 'finance'
            for q in questions:
                q["domain"] = domain  # Tag each question with its source domain
            all_questions.extend(questions)
            console.print(f"  [dim]Loaded {len(questions)} questions from {json_file.name} (domain: {domain})[/dim]")
        except Exception as e:
            console.print(f"[red][ERROR] Failed to load {json_file.name}: {e}[/red]")
    
    return all_questions


@dataclass
class EvaluationResult:
    question: str
    category: str
    domain: str
    question_type: str
    answer: str
    consistency_score: float
    confidence_level: str
    regeneration_count: int
    llm_call_count: int
    used_strict_mode: bool
    used_query_rewrite: bool
    rewritten_query: Optional[str]
    sources: List[str]
    processing_time: float
    is_reliable: bool
    keyword_coverage: float
    is_trick: bool = False


class RAGEvaluator:
    """Evaluates the Hallucination-Aware RAG system across multiple dimensions."""

    def __init__(self, rag: HallucinationAwareRAG):
        self.rag = rag
        self.results: List[EvaluationResult] = []

    def run_evaluation(self, questions: List[Dict] = None) -> List[EvaluationResult]:
        """Run evaluation on all test questions.
        
        If no questions are provided, loads from evaluation/datasets/ directory.
        """
        if questions is None:
            questions = load_evaluation_datasets()
        
        if not questions:
            console.print("[red]No evaluation questions found. Check evaluation/datasets/ directory.[/red]")
            return []
        console.print(Panel.fit(
            f"[bold white]RAG System Evaluation[/bold white]\n"
            f"[dim]Testing {len(questions)} questions across {len(set(q['category'] for q in questions))} categories[/dim]",
            border_style="bright_blue"
        ))

        self.results = []
        for i, q_data in enumerate(questions, 1):
            console.print(f"\n[bold magenta]Test {i}/{len(questions)}[/bold magenta] | Category: {q_data['category']}")
            console.print(f"[dim]Question: {q_data['question']}[/dim]")
            
            try:
                response = self.rag.query(q_data['question'], verbose=False)
                
                # Calculate keyword coverage
                answer_lower = response.answer.lower()
                expected_kw = q_data.get("expected_keywords", [])
                if expected_kw:
                    covered = sum(1 for kw in expected_kw if kw.lower() in answer_lower)
                    kw_coverage = covered / len(expected_kw)
                else:
                    kw_coverage = 1.0
                
                result = EvaluationResult(
                    question=q_data['question'],
                    category=q_data.get('category', 'General'),
                    domain=q_data.get('domain', 'unknown'),
                    question_type=q_data.get('type', 'direct'),
                    answer=response.answer,
                    consistency_score=response.consistency_score,
                    confidence_level=response.confidence_label,
                    regeneration_count=response.regeneration_count,
                    llm_call_count=response.llm_call_count,
                    used_strict_mode=response.used_strict_mode,
                    used_query_rewrite=response.used_query_rewrite,
                    rewritten_query=response.rewritten_query,
                    sources=response.sources,
                    processing_time=response.processing_time_s,
                    is_reliable=response.is_reliable,
                    keyword_coverage=kw_coverage,
                    is_trick=q_data.get("is_trick", False),
                )
                
                self.results.append(result)
                
                # Quick status
                status = "✅" if result.is_reliable else "⚠️"
                console.print(
                    f"  {status} Score: {result.consistency_score:.2f} | "
                    f"Keywords: {kw_coverage*100:.0f}% | "
                    f"Regen: {result.regeneration_count} | "
                    f"Time: {result.processing_time:.1f}s"
                )
                
            except Exception as e:
                console.print(f"  [red]ERROR: {e}[/red]")

        return self.results

    def print_report(self):
        """Print a comprehensive evaluation report."""
        if not self.results:
            console.print("[red]No results to report[/red]")
            return

        console.print(Rule("[bold blue]Evaluation Report[/bold blue]"))

        # ── Per-question results table ───────────────────────────────────────
        table = Table(
            title="Question-by-Question Results",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        table.add_column("Domain", style="dim cyan", max_width=10)
        table.add_column("Category", style="cyan", max_width=18)
        table.add_column("Question", max_width=30)
        table.add_column("Score", justify="center")
        table.add_column("KW%", justify="center")
        table.add_column("Regen", justify="center")
        table.add_column("LLMs", justify="center")
        table.add_column("Time(s)", justify="center")
        table.add_column("Status", justify="center")

        for r in self.results:
            score_color = "green" if r.is_reliable else ("yellow" if r.consistency_score >= 0.4 else "red")
            q_short = r.question[:28] + "..." if len(r.question) > 30 else r.question
            table.add_row(
                r.domain,
                r.category,
                q_short,
                f"[{score_color}]{r.consistency_score:.2f}[/{score_color}]",
                f"{r.keyword_coverage*100:.0f}%",
                str(r.regeneration_count),
                str(r.llm_call_count),
                f"{r.processing_time:.1f}",
                "✅" if r.is_reliable else "⚠️"
            )
        
        console.print(table)

        # ── Aggregate metrics ────────────────────────────────────────────────
        total = len(self.results)
        reliable = sum(1 for r in self.results if r.is_reliable)
        avg_score = sum(r.consistency_score for r in self.results) / total
        avg_kw = sum(r.keyword_coverage for r in self.results) / total
        total_regens = sum(r.regeneration_count for r in self.results)
        total_llm_calls = sum(r.llm_call_count for r in self.results)
        avg_llm_calls = total_llm_calls / total
        strict_used = sum(1 for r in self.results if r.used_strict_mode)
        avg_time = sum(r.processing_time for r in self.results) / total
        
        # Trick question handling
        trick_results = [r for r in self.results if r.is_trick]
        trick_handled = sum(
            1 for r in trick_results
            if any(kw in r.answer.lower() for kw in ["not", "no information", "cannot", "don't"])
        )

        query_rewrites = sum(1 for r in self.results if r.used_query_rewrite)

        summary = Table(title="📊 Aggregate Metrics", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold")
        summary.add_column("Notes")

        summary.add_row("Total Questions", str(total), "")
        summary.add_row(
            "Reliable Responses",
            f"[green]{reliable}/{total} ({reliable/total*100:.0f}%)[/green]",
            f"Score ≥ {HALLUCINATION_THRESHOLD}"
        )
        summary.add_row("Avg Consistency Score", f"{avg_score:.3f}", "Higher = more grounded")
        summary.add_row("Avg Keyword Coverage", f"{avg_kw*100:.1f}%", "Expected terms found")
        summary.add_row("Total Regenerations", str(total_regens), "Self-corrections performed")
        summary.add_row("Total LLM Calls", str(total_llm_calls), f"Avg {avg_llm_calls:.1f} per query")
        summary.add_row("Strict Mode Activations", str(strict_used), "Severe hallucination triggers")
        summary.add_row("Query Rewrites Triggered", str(query_rewrites), "Severe retrieval failure recoveries")
        summary.add_row("Avg Processing Time", f"{avg_time:.2f}s", "Per query")
        if trick_results:
            summary.add_row(
                "Trick Questions Handled",
                f"{trick_handled}/{len(trick_results)}",
                "Correctly refused to hallucinate"
            )

        console.print(summary)

    def save_report(self) -> str:
        """Save the evaluation report to a JSON file."""
        os.makedirs(OUTPUTS_PATH, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(OUTPUTS_PATH, f"evaluation_report_{timestamp}.json")
        
        total = len(self.results)
        total_llm_calls = sum(r.llm_call_count for r in self.results)
        data = {
            "timestamp": timestamp,
            "summary": {
                "total_questions": total,
                "reliable_responses": sum(1 for r in self.results if r.is_reliable),
                "avg_consistency_score": sum(r.consistency_score for r in self.results) / total if total else 0,
                "avg_keyword_coverage": sum(r.keyword_coverage for r in self.results) / total if total else 0,
                "total_regenerations": sum(r.regeneration_count for r in self.results),
                "total_query_rewrites": sum(1 for r in self.results if r.used_query_rewrite),
                "total_llm_calls": total_llm_calls,
                "avg_llm_calls_per_query": total_llm_calls / total if total else 0,
            },
            "results": [
                {
                    "question": r.question,
                    "category": r.category,
                    "domain": r.domain,
                    "question_type": r.question_type,
                    "answer": r.answer,
                    "consistency_score": r.consistency_score,
                    "confidence_level": r.confidence_level,
                    "keyword_coverage": r.keyword_coverage,
                    "regeneration_count": r.regeneration_count,
                    "llm_call_count": r.llm_call_count,
                    "used_strict_mode": r.used_strict_mode,
                    "used_query_rewrite": r.used_query_rewrite,
                    "rewritten_query": r.rewritten_query,
                    "sources": r.sources,
                    "processing_time_s": r.processing_time,
                    "is_reliable": r.is_reliable,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[dim] Evaluation report saved to: {filepath}[/dim]")
        return filepath


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the Hallucination-Aware RAG system")
    parser.add_argument("--questions", help="Path to custom questions JSON file")
    parser.add_argument("--compare", action="store_true", help="Run empirical comparison against naive baseline RAG")
    parser.add_argument("--ground-truth", action="store_true", help="Run ground truth agreement evaluation for detector")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between queries to respect rate limits")
    args = parser.parse_args()

    # Load custom questions if provided
    questions = None
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)

    # Initialize RAG engine
    rag = HallucinationAwareRAG()

    if args.compare:
        from experiments import BaselineExperiment
        exp = BaselineExperiment(rag)
        exp.run(questions=questions, max_questions=args.limit, delay_s=args.delay)
        exp.print_comparison_report()
        exp.save_report()
    elif args.ground_truth:
        from experiments import GroundTruthEvaluator
        evaluator = GroundTruthEvaluator(rag)
        evaluator.run(delay_s=args.delay)
        evaluator.print_report()
        evaluator.save_report()
    else:
        evaluator = RAGEvaluator(rag)
        if questions is None:
            questions = load_evaluation_datasets()
        if args.limit:
            questions = questions[:args.limit]
        evaluator.run_evaluation(questions)
        evaluator.print_report()
        evaluator.save_report()


if __name__ == "__main__":
    main()