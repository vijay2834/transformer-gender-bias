"""Word Embedding Association Test (Caliskan et al., 2017).

s(w, A, B)  = mean_a cos(w, a) - mean_b cos(w, b)
effect size = [mean_x s(x,A,B) - mean_y s(y,A,B)] / std_{w in X∪Y} s(w,A,B)
p-value     = one-sided permutation test over equal-size partitions of X∪Y
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.linalg.norm(M, axis=-1, keepdims=True)


@dataclass
class WEATResult:
    effect_size: float
    p_value: float
    n_permutations: int


def association(W: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    W, A, B = _unit(W), _unit(A), _unit(B)
    return (W @ A.T).mean(1) - (W @ B.T).mean(1)


def weat(X, Y, A, B, n_perm: int = 10_000, seed: int = 0) -> WEATResult:
    X, Y, A, B = map(np.asarray, (X, Y, A, B))
    s = association(np.vstack([X, Y]), A, B)
    sx, sy = s[: len(X)], s[len(X):]
    effect = (sx.mean() - sy.mean()) / s.std(ddof=1)
    stat = sx.sum() - sy.sum()
    rng = np.random.default_rng(seed)
    n, count = len(s), 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        count += (s[idx[: len(X)]].sum() - s[idx[len(X):]].sum()) >= stat
    return WEATResult(float(effect), (count + 1) / (n_perm + 1), n_perm)
