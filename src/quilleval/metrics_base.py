from __future__ import annotations
from typing import List, Dict, Callable
_REGISTRY: Dict[str, "Metric"] = {}

class Metric:
    name: str
    def score(self, candidate: str, references: List[str]) -> float:
        raise NotImplementedError

def register(name: str):
    def deco(cls):
        instance = cls()
        instance.name = name
        _REGISTRY[name] = instance
        return cls
    return deco

def get_metric(name: str) -> Metric:
    return _REGISTRY[name]

def all_metrics() -> Dict[str, Metric]:
    return dict(_REGISTRY)
