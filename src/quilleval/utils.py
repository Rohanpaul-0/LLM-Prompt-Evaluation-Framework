from __future__ import annotations
from typing import Any, Iterable, List, Dict, Union
import json, csv, pandas as pd

def to_list(x):
    if x is None: return []
    if isinstance(x, str): return [x]
    return list(x)

def load_any(path: str):
    if path.endswith(".jsonl"):
        rows = []
        # accept UTF-8 with BOM as well
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                # strip BOM if present on the first line
                line = line.lstrip("\ufeff").strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        # also accept BOM for .json
        return json.load(open(path, "r", encoding="utf-8-sig"))
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("Unsupported extension. Use .jsonl/.json/.csv")
