# QuillEval — Prompt & LLM Output Evaluation

QuillEval is a framework for evaluating and comparing large language model (LLM) prompts and outputs.  
It combines traditional reference-based metrics (BLEU, ROUGE), embedding-based similarity (SBERT cosine), and statistical analysis (bootstrap confidence intervals, per-tag summaries).  
The design treats evaluation as a reproducible experiment: inputs, configuration, metrics, and results are stored in a lightweight SQLite registry.

---

## Why QuillEval?
Most existing tools only provide a single type of metric or focus on one-off comparisons.  
QuillEval is built for iteration and reproducibility:

- Multiple metrics can be combined with configurable weights.
- Results are summarized both overall and by tag (for specific categories such as `math`, `safety`, etc.).
- Every run is logged to an experiment registry for later inspection.

---

## Features
- Plugin metric API: extend QuillEval with your own custom metrics.
- Confidence intervals: scores include nonparametric bootstrap confidence intervals.
- Per-tag analysis: tag records (e.g., `math`, `security`) and get summaries for each group.
- Semantic similarity: SBERT cosine similarity with configurable model.
- Reports: outputs are saved to CSV and displayed in rich console tables.
- Run registry (SQLite): store run configuration, metrics, and git information; list and compare runs later.
- Command-line interface (CLI): includes `evaluate`, `runs-ls`, and `runs-show` commands.

---

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On macOS/Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install QuillEval (editable mode)
pip install -e .




## Evaluate

pip install -e .
python -m quilleval.cli evaluate `
  --data .\examples\qa.jsonl `
  --out qa.csv `
  --weights bleu=0.3 `
  --weights rougeL=0.3 `
  --weights semantic=0.4 `
  --registry quilleval.db



