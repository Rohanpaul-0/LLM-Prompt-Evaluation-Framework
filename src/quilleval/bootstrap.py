from __future__ import annotations
import numpy as np
from typing import Callable, List, Tuple

def bootstrap_ci(values: List[float], iterations: int = 1000, alpha: float = 0.05) -> Tuple[float,float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    vals = np.array(values)
    n = len(vals)
    sims = []
    for _ in range(iterations):
        idx = np.random.randint(0, n, size=n)
        sims.append(np.mean(vals[idx]))
    sims = np.sort(sims)
    lo = sims[int((alpha/2)*iterations)]
    hi = sims[int((1-alpha/2)*iterations)]
    return float(lo), float(hi)
