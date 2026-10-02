"""
run.py — One-Stop Entry Point for Hallucination-Aware RAG

Handles the full pipeline:
  1. Install dependencies (optional)
  2. Create vector database
  3. Run query / interactive / demo / evaluation mode

Usage:
    python run.py setup              # Create database
    python run.py query "Question?"  # Single query
    python run.py interactive        # Interactive chat
    python run.py demo               # Run demo questions
    python run.py evaluate           # Run evaluation suite
    python run.py test               # Run unit tests
"""

import sys
import os
import subprocess
from pathlib import Path

# Fix Windows console encoding for Unicode/rich output
if sys.platform.startswith("win"):
    try:
        if sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console(legacy_windows=False)

# Add src to path
SRC_PATH = str(Path(__file__).parent / "src")
sys.path.insert(0, SRC_PATH)


def banner():
    console.print(Panel(
        "[bold white]🧠 Hallucination-Aware Retrieval-Augmented Generation[/bold white]\n"
        "[cyan]with Agentic Self-Correction Loops[/cyan]\n\n"
        "[dim]Practical implementation of corrective RAG concepts\n"
        "using claim-level consistency grading and multi-stage recovery[/dim]",
        border_style="bright_blue",
        padding=(1, 4)
    ))


def check_env():
    """Check that .env file exists with valid GEMINI_API_KEY."""
    env_file = Path(".env")
    if not env_file.exists():
        console.print("[yellow]⚠️  .env file not found. Copying from .env.example...[/yellow]")
        import shutil
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
        console.print("[red]❗ Please edit .env and configure your GEMINI_API_KEY.[/red]")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    from config import resolve_llm_config
    try:
        cfg = resolve_llm_config()
        provider = cfg["provider"]
        model = cfg["model"]
        chain_summary = cfg.get("chain_summary", f"{provider} ({model})")
        
        console.print(f"[green]✓ Primary Provider:[/green] [bold cyan]{provider.capitalize()}[/bold cyan] [dim](Model: {model})[/dim]")
        if cfg.get("fallback_count", 0) > 0:
            console.print(f"[dim]  Active Gemini Fallback Chain: [cyan]{chain_summary}[/cyan][/dim]")
        return True
    except Exception as e:
        console.print(f"[red]❗ Configuration error: {e}[/red]")
        return False


def cmd_setup():
    """Create the vector database."""
    console.print(Rule("[bold blue]Setup: Creating Vector Database[/bold blue]"))
    
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, SRC_PATH)
    
    from create_database import main as create_db
    create_db(reset=True)


def cmd_query(question: str):
    """Run a single query."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_single_query
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    run_single_query(rag, question)


def cmd_interactive():
    """Run interactive mode."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_interactive
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    run_interactive(rag)


def cmd_demo(args_list: list = None):
    """Run comprehensive demonstration showcase (Phase 1 to Phase 4)."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from query_rag import run_demo
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    args_list = args_list or []
    
    per_domain = 5
    run_all_phases = True
    delay_s = 1.5
    
    if "--quick" in args_list:
        per_domain = 2
        delay_s = 1.0
    if "--standalone" in args_list:
        run_all_phases = False
        
    for i, a in enumerate(args_list):
        if a == "--per-domain" and i + 1 < len(args_list):
            try:
                per_domain = int(args_list[i + 1])
            except ValueError:
                pass
        elif a == "--delay" and i + 1 < len(args_list):
            try:
                delay_s = float(args_list[i + 1])
            except ValueError:
                pass
    
    run_demo(rag, per_domain=per_domain, delay_s=delay_s, run_all_phases=run_all_phases)


def cmd_evaluate(args_list: list = None):
    """Run evaluation suite with optional --compare or --ground-truth."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from evaluate import RAGEvaluator, load_evaluation_datasets
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    args_list = args_list or []
    
    if "--compare" in args_list:
        from experiments import BaselineExperiment
        limit = None
        for i, a in enumerate(args_list):
            if a == "--limit" and i + 1 < len(args_list):
                try:
                    limit = int(args_list[i + 1])
                except ValueError:
                    pass
        exp = BaselineExperiment(rag)
        exp.run(max_questions=limit)
        exp.print_comparison_report()
        exp.save_report()
    elif "--ground-truth" in args_list:
        from experiments import GroundTruthEvaluator
        limit = None
        for i, a in enumerate(args_list):
            if a == "--limit" and i + 1 < len(args_list):
                try:
                    limit = int(args_list[i + 1])
                except ValueError:
                    pass
        evaluator = GroundTruthEvaluator(rag)
        evaluator.run(max_items=limit, delay_s=2.0)
        evaluator.print_report()
        evaluator.save_report()
    else:
        evaluator = RAGEvaluator(rag)
        questions = load_evaluation_datasets()
        limit = None
        for i, a in enumerate(args_list):
            if a == "--limit" and i + 1 < len(args_list):
                try:
                    limit = int(args_list[i + 1])
                except ValueError:
                    pass
        if limit:
            questions = questions[:limit]
        evaluator.run_evaluation(questions)
        evaluator.print_report()
        evaluator.save_report()


def cmd_calibrate():
    """Run threshold calibration experiment (offline simulation)."""
    if not check_env():
        return
    
    os.chdir(Path(__file__).parent)
    from experiments import ThresholdCalibrator
    from rag_engine import HallucinationAwareRAG
    
    rag = HallucinationAwareRAG()
    calibrator = ThresholdCalibrator(rag)
    calibrator.run_all()


def cmd_test():
    """Run unit tests."""
    console.print(Rule("[bold blue]Running Unit Tests[/bold blue]"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    return result.returncode


def print_help():
    console.print("""
[bold]Usage:[/bold]
  python run.py [command] [options]

[bold]Commands:[/bold]
  [cyan]setup[/cyan]                     Create the ChromaDB vector database from documents
  [cyan]query[/cyan] "Question"          Run a single query with source citations
  [cyan]interactive[/cyan]               Start interactive question-answering session
  [cyan]demo[/cyan]                      Run master showcase: 5 questions/domain + baseline A/B + ground truth + calibration
  [cyan]demo --quick[/cyan]              Run quick showcase (2 questions/domain + all 4 phases)
  [cyan]demo --per-domain N[/cyan]       Run showcase with custom N questions per domain (default: 5)
  [cyan]demo --standalone[/cyan]         Run only multi-domain demo questions without benchmark phases
  [cyan]evaluate[/cyan]                  Run evaluation suite on all domains
  [cyan]evaluate --compare[/cyan]        Run empirical A/B comparison (Baseline vs Corrective)
  [cyan]evaluate --ground-truth[/cyan]   Run detector agreement against human labels
  [cyan]calibrate[/cyan]                 Run zero-cost threshold calibration experiment
  [cyan]test[/cyan]                      Run unit tests

[bold]Examples:[/bold]
  python run.py setup
  python run.py query "What is hallucination in AI?"
  python run.py demo
  python run.py demo --quick
  python run.py evaluate --compare
  python run.py calibrate
  python run.py test
""")


def main():
    banner()
    
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1].lower()
    extra_args = sys.argv[2:]
    
    os.chdir(Path(__file__).parent)
    
    if cmd == "setup":
        cmd_setup()
    elif cmd == "query":
        if not extra_args:
            console.print("[red]Usage: python run.py query 'Your question here'[/red]")
        else:
            cmd_query(" ".join(extra_args))
    elif cmd == "interactive":
        cmd_interactive()
    elif cmd == "demo":
        cmd_demo(extra_args)
    elif cmd == "evaluate":
        cmd_evaluate(extra_args)
    elif cmd == "calibrate":
        cmd_calibrate()
    elif cmd == "test":
        sys.exit(cmd_test())
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        print_help()


if __name__ == "__main__":
    main()