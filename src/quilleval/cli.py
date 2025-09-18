from __future__ import annotations
import json, typer
from typing import Optional, List, Dict
from rich.console import Console
from rich.table import Table
import pandas as pd

from .utils import load_any
from .evaluator import evaluate as eval_records, summarize_by_tag
from .registry import Registry

app = typer.Typer(add_completion=False)
console = Console()


def parse_weights(weights: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for w in weights:
        k, v = w.split("=", 1)
        out[k] = float(v)
    return out


@app.command("evaluate")
def evaluate(
    data: str = typer.Option(..., help="Path to JSONL/JSON/CSV dataset"),
    out: Optional[str] = typer.Option(None, help="CSV to save row-wise results"),
    weights: List[str] = typer.Option(
        ["bleu=0.2", "rougeL=0.3", "semantic=0.5"], help="Metric weights (repeat flag)"
    ),
    sbert_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model name (e.g., all-mpnet-base-v2)"
    ),
    registry: Optional[str] = typer.Option(None, help="SQLite path to log this run (e.g., quilleval.db)"),
    notes: Optional[str] = typer.Option(None, help="Optional run notes"),
):
    # Trigger SBERT model load via plugin (kept lazy so CLI starts fast)
    from .semantic import SBERTCosine  # noqa: F401

    ws = parse_weights(weights)
    recs = load_any(data)
    df = eval_records(recs, ws)

    # -------- Pretty table --------
    cols = [c for c in ["id", "model", "score", "bleu", "rougeL", "semantic", "tags"] if c in df.columns]
    table = Table(title="QuillEval — Top Results")
    for c in cols:
        table.add_column(c)

    for _, row in df.head(10).iterrows():
        cells = []
        for c in cols:
            val = row[c]
            cells.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        table.add_row(*cells)

    console.print(table)
    # -------- End pretty table --------

    tag_sum = summarize_by_tag(df)
    if tag_sum is not None and len(tag_sum) > 0:
        console.print("\n[bold]Per-tag summary (means with bootstrap CIs):[/bold]")
        console.print(tag_sum)

    if out:
        df.to_csv(out, index=False)
        console.print(f"[green]Saved results to {out}[/green]")

    if registry:
        reg = Registry(registry)
        run = reg.create_run({"data": data, "weights": ws, "sbert_model": sbert_model}, notes=notes)
        for _, row in df.iterrows():
            for m in ["score", "bleu", "rougeL", "semantic"]:
                if m in row:
                    reg.log_metric(run.run_id, str(row["id"]), str(row["model"]), m, float(row[m]))
        console.print(f"[cyan]Logged run {run.run_id} to {registry}[/cyan]")


@app.command("runs-ls")
def runs_ls(registry: str = typer.Option(...)):
    reg = Registry(registry)
    rows = reg.list_runs()
    table = Table(title="QuillEval — Runs")
    table.add_column("run_id"); table.add_column("created_ts"); table.add_column("git_commit"); table.add_column("notes")
    for r in rows:
        table.add_row(str(r[0]), f"{r[1]:.0f}", str(r[2]), str(r[3]))
    console.print(table)


@app.command("runs-show")
def runs_show(registry: str = typer.Option(...), run_id: str = typer.Option(...)):
    reg = Registry(registry)
    r = reg.get_run(run_id)
    console.print(r)


if __name__ == "__main__":
    app()
