from __future__ import annotations
from typing import List, Dict
import sacrebleu
from rouge_score import rouge_scorer
from .metrics_base import Metric, register

@register("bleu")
class BleuMetric(Metric):
    def score(self, candidate: str, references: List[str]) -> float:
        bleu = sacrebleu.corpus_bleu([candidate], [references])
        return float(bleu.score) / 100.0

@register("rougeL")
class RougeLMetric(Metric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    def score(self, candidate: str, references: List[str]) -> float:
        vals = []
        for ref in references:
            s = self.scorer.score(ref, candidate)
            vals.append(s["rougeLsum"].fmeasure)
        return sum(vals)/max(len(vals),1)
