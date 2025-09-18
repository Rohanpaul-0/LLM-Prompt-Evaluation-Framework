from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
from .utils import to_list
from .metrics_base import get_metric, all_metrics
from . import text_metrics, semantic  # register plugins
from .bootstrap import bootstrap_ci

@dataclass
class Weights:
    bleu: float = 0.2
    rougeL: float = 0.3
    semantic: float = 0.5

def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: v/s if s>0 else 0 for k,v in d.items()}

def evaluate(records: List[Dict[str, Any]], weights: Dict[str,float]) -> pd.DataFrame:
    weights = _normalize(weights)
    metrics = all_metrics()
    rows = []
    for rec in records:
        refs = to_list(rec.get("reference",""))
        cand = rec.get("candidate")
        tags = rec.get("tags", [])
        items = cand.items() if isinstance(cand, dict) else [("single", str(cand))]
        for name, text in items:
            mvals = {}
            for mname in weights.keys():
                m = metrics[mname]
                mvals[mname] = m.score(text, refs)
            score = sum(weights[k]*mvals[k] for k in weights.keys())
            rows.append({
                "id": rec.get("id"),
                "model": name,
                "score": score,
                **mvals,
                "tags": ",".join(tags) if isinstance(tags, list) else str(tags)
            })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def summarize_by_tag(df: pd.DataFrame, metric: str = "score"):
    if "tags" not in df.columns: return None
    # explode tags
    tmp = df.copy()
    tmp["tags"] = tmp["tags"].fillna("")
    tmp = tmp.assign(tag=tmp["tags"].str.split(",")).explode("tag")
    tmp["tag"] = tmp["tag"].str.strip()
    tmp = tmp[tmp["tag"]!=""]
    # aggregate + CIs per tag
    out = []
    for tag, grp in tmp.groupby("tag"):
        vals = grp[metric].astype(float).tolist()
        lo, hi = bootstrap_ci(vals)
        out.append({"tag": tag, "mean": float(sum(vals)/len(vals)), "lo": lo, "hi": hi, "n": len(vals)})
    import pandas as pd
    return pd.DataFrame(out).sort_values("mean", ascending=False)
