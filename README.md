# QuillEval — Prompt & LLM Output Evaluation

QuillEval is a research-grade framework to **evaluate and compare LLM prompts and model outputs**.
It combines *reference-based* (BLEU, ROUGE), *embedding-based* (SBERT cosine), and *distribution-aware* analysis
(bootstrap confidence intervals, per-slice breakdowns), with a modular plug‑in metric API.

**Why this exists**  
Most toolkits focus on one metric. QuillEval treats evaluation as an experiment: inputs, config,
metrics, and results are tracked in a lightweight SQLite registry for reproducibility.

## Features
- 🧩 **Plugin metric API**: add custom metrics via a simple decorator
- 🧪 **Confidence intervals** via nonparametric bootstrap
- 🔍 **Per-slice analysis**: tag records (e.g., `math`, `safety`) and get per-tag summaries
- 🧠 **Semantic similarity**: Sentence-Transformers cosine with configurable model
- 📈 **Reports**: CSV + pretty console tables
- 🗄️ **Run registry** (SQLite): store config, git info, scores; list/compare later
- 🧰 **CLI** with subcommands: `evaluate`, `report`, `runs ls`, `runs show`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Evaluate the example dataset (multi-candidate + slices)
python -m quilleval.cli evaluate   --data examples/demo.jsonl   --out results.csv   --sbert-model sentence-transformers/all-MiniLM-L6-v2   --weights bleu=0.15 rougeL=0.35 semantic=0.5   --registry quilleval.db
```

## Data format (JSONL/CSV/JSON)
Each record:
- `id`: str/int
- `prompt`: str
- `reference`: str or list[str]
- `candidate`: str or dict `{name: output}`
- `tags` (optional): list[str] for slice analysis

## License
MIT
