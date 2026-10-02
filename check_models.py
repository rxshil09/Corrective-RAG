"""
check_models.py — Model Diagnostics & Latency Tool for Google Gemini
Tests:
  - API connectivity & authentication
  - OpenAI-compatible JSON mode (required by HallucinationDetector)
  - Latency (response time in ms)
  - Multi-tier Gemini fallback readiness
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Ensure UTF-8 output encoding across Windows consoles
if sys.platform.startswith("win"):
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

load_dotenv()
console = Console(legacy_windows=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip("'").strip('"')

# Candidate Gemini models to test for RAG fallback chain
GEMINI_CANDIDATES = [
    ("gemini-3.5-flash-lite", "Primary - Ultra-fast, high capacity (250k TPM, 500 RPD)"),
    ("gemini-3.1-flash-lite", "1st Fallback - Lightweight, fast response"),
    ("gemini-3.8-flash", "2nd Fallback - High throughput, advanced reasoning"),
    ("gemini-3.7-flash", "3rd Fallback - Balanced reasoning & speed"),
    ("gemini-3.6-flash", "4th Fallback - Stable performance"),
    ("gemini-3.5-flash", "5th Fallback - High capability generation"),
    ("gemini-2.5-flash-lite", "Legacy candidate (Checked against API)"),
    ("gemini-2.5-flash", "Legacy candidate (Checked against API)"),
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
            with urllib.request.urlopen(req, timeout=25) as resp:
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
                status_label = "[dim red]404 (Discontinued by Google)[/dim red]"
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


def main():
    console.print(Panel(
        "[bold white]🚀 Google Gemini Model Diagnostics & Latency Tool[/bold white]\n"
        "[dim]Tests latency, JSON-mode support, and active status for Gemini models[/dim]",
        border_style="magenta"
    ))

    gemini_res = test_gemini_models()

    # Summary Table
    table = Table(title="\n📊 Google Gemini Fallback Readiness for RAG", show_lines=True)
    table.add_column("Provider", style="bold")
    table.add_column("Model ID", style="cyan")
    table.add_column("Latency", justify="right")
    table.add_column("Status")
    table.add_column("Role / Diagnosis", style="dim")

    for m, desc, lat, status, ok in gemini_res:
        table.add_row("Google Gemini", m, lat, status, desc)

    console.print(table)


if __name__ == "__main__":
    main()
