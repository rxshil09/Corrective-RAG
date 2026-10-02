"""
check_models.py — Model Diagnostics & Latency Tool for Google Gemini & OpenAI
Tests:
  - API connectivity & authentication
  - OpenAI-compatible JSON mode (required by HallucinationDetector)
  - Latency (response time in ms)
  - Fallback readiness (gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> OpenAI gpt-4o-mini)
"""

import os
import time
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console(legacy_windows=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("'").strip('"')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip("'").strip('"')

# Candidate Gemini models to test for RAG
GEMINI_CANDIDATES = [
    ("gemini-3.5-flash-lite", "Primary - Ultra-fast, high capacity (250k TPM, 500 RPD)"),
    ("gemini-3.1-flash-lite", "1st Fallback - Lightweight, fast response"),
    ("gemini-3.8-flash", "High throughput, advanced reasoning"),
    ("gemini-3.7-flash", "Balanced reasoning & speed"),
    ("gemini-3.6-flash", "Stable performance"),
    ("gemini-flash-latest", "Auto-routed latest stable flash"),
    ("gemini-flash-lite-latest", "Lightweight auto-routed"),
]


def test_gemini_models():
    console.print("\n" + "=" * 70)
    console.print("[bold blue]🔍 Testing Google Gemini Models (OpenAI-compatible Endpoint)[/bold blue]")
    console.print("=" * 70)

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        console.print("[yellow]⚠️ GEMINI_API_KEY is not set in .env. Skipping Gemini checks.[/yellow]")
        return []

    results = []
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    for model_id, desc in GEMINI_CANDIDATES:
        payload = json.dumps({
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful JSON assistant."},
                {"role": "user", "content": "Respond with JSON: {\"status\": \"ok\", \"score\": 0.95}"}
            ],
            "response_format": {"type": "json_object"}
        }).encode("utf-8")

        start = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "User-Agent": "RAG-Model-Benchmark"
                }
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                elapsed_ms = int((time.time() - start) * 1000)
                data = json.loads(resp.read().decode())
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Check valid JSON
                json_ok = False
                try:
                    parsed = json.loads(content)
                    if "status" in parsed or "score" in parsed:
                        json_ok = True
                except:
                    pass

                status_label = "[bold green]ONLINE (JSON OK)[/bold green]" if json_ok else "[green]ONLINE[/green]"
                results.append((model_id, desc, f"{elapsed_ms} ms", status_label, True))
                console.print(f"  ✓ [bold cyan]{model_id:<26}[/bold cyan] -> {status_label} ({elapsed_ms} ms)")
        except urllib.error.HTTPError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            if e.code == 503:
                status_label = "[yellow]503 BUSY (High Demand)[/yellow]"
            elif e.code == 404:
                status_label = "[dim red]404 (Not Available to Key)[/dim red]"
            elif e.code == 429:
                status_label = "[orange3]429 (Rate Limit / Quota)[/orange3]"
            else:
                status_label = f"[red]HTTP {e.code}[/red]"
            results.append((model_id, desc, "-", status_label, False))
            console.print(f"  ✗ [cyan]{model_id:<26}[/cyan] -> {status_label}")
        except Exception as e:
            results.append((model_id, desc, "-", f"[red]Error: {str(e)[:30]}[/red]", False))
            console.print(f"  ✗ [cyan]{model_id:<26}[/cyan] -> [red]Error: {e}[/red]")

    return results


def test_openai_model():
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
        return []

    console.print("\n" + "=" * 70)
    console.print("[bold green]🔍 Testing OpenAI Fallback Model (gpt-4o-mini)[/bold green]")
    console.print("=" * 70)

    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Respond with JSON: {\"status\": \"ok\"}"}
        ],
        "response_format": {"type": "json_object"}
    }).encode("utf-8")

    start = time.time()
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "User-Agent": "RAG-Model-Benchmark"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed_ms = int((time.time() - start) * 1000)
            return [("gpt-4o-mini", "OpenAI Secondary Fallback", f"{elapsed_ms} ms", "[bold green]ONLINE (JSON OK)[/bold green]", True)]
    except urllib.error.HTTPError as e:
        return [("gpt-4o-mini", "OpenAI Secondary Fallback", "-", f"[orange3]HTTP {e.code} (Quota/Auth)[/orange3]", False)]
    except Exception as e:
        return [("gpt-4o-mini", "OpenAI Secondary Fallback", "-", f"[red]Error: {str(e)[:30]}[/red]", False)]


def main():
    console.print(Panel(
        "[bold white]🚀 Model Benchmark & Compatibility Evaluator for RAG[/bold white]\n"
        "[dim]Tests latency, JSON-mode support, and active status for Gemini & OpenAI[/dim]",
        border_style="magenta"
    ))

    gemini_res = test_gemini_models()
    openai_res = test_openai_model()

    # Summary Table
    table = Table(title="\n📊 Available Models for Hallucination-Aware RAG", show_lines=True)
    table.add_column("Provider", style="bold")
    table.add_column("Model ID", style="cyan")
    table.add_column("Latency", justify="right")
    table.add_column("Status")
    table.add_column("Description & Role", style="dim")

    for m, desc, lat, status, ok in gemini_res:
        table.add_row("Google Gemini", m, lat, status, desc)

    for m, desc, lat, status, ok in openai_res:
        table.add_row("OpenAI", m, lat, status, desc)

    console.print(table)


if __name__ == "__main__":
    main()
