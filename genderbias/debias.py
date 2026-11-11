"""Post-hoc debiasing of embeddings.

* Hard-Debias (Bolukbasi et al., 2016): find the gender direction by PCA over
  definitional pairs, remove it from gender-neutral words, then *equalize*
  explicit pairs so they are equidistant from every neutral word.
* INLP (Ravfogel et al., 2020): Iterative Nullspace Projection. Repeatedly train a
  linear classifier to predict gender from embeddings and project onto its
  nullspace until gender is no longer linearly recoverable.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def gender_direction(emb: dict[str, np.ndarray], pairs) -> np.ndarray:
    diffs = []
    for f, m in pairs:
        if f in emb and m in emb:
            c = (emb[f] + emb[m]) / 2
            diffs += [emb[f] - c, emb[m] - c]
    g = PCA(n_components=1).fit(np.array(diffs)).components_[0]
    return g / np.linalg.norm(g)


def hard_debias(emb: dict[str, np.ndarray], neutral: set[str], def_pairs, eq_pairs) -> dict[str, np.ndarray]:
    g = gender_direction(emb, def_pairs)
    out = {}
    for w, v in emb.items():
        v = v / np.linalg.norm(v)
        out[w] = v - (v @ g) * g if w in neutral else v
        if w in neutral:
            out[w] /= np.linalg.norm(out[w])
    for a, b in eq_pairs:  # equalize
        if a in out and b in out:
            mu = (out[a] + out[b]) / 2
            nu = mu - (mu @ g) * g
            scale = np.sqrt(max(1 - nu @ nu, 0))
            for w in (a, b):
                sign = np.sign(out[w] @ g) or 1.0
                out[w] = nu + sign * scale * g
    return out


def inlp(emb: dict[str, np.ndarray], female: list[str], male: list[str], neutral_vocab: list[str],
         n_iter: int = 20, seed: int = 0) -> tuple[dict[str, np.ndarray], list[float]]:
    """Returns debiased embeddings and the gender-classification accuracy per iteration."""
    words = [w for w in female + male if w in emb]
    X = np.array([emb[w] for w in words]); y = np.array([w in female for w in words], int)
    d = X.shape[1]
    P = np.eye(d)
    accs = []
    for _ in range(n_iter):
        Xtr, Xte, ytr, yte = train_test_split(X @ P, y, test_size=0.3, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        accs.append(float(clf.score(Xte, yte)))
        w = clf.coef_ / np.linalg.norm(clf.coef_)
        P = P @ (np.eye(d) - w.T @ w)
        if accs[-1] <= 0.55:
            break
    out = {w: v @ P for w, v in emb.items() if w in set(neutral_vocab) | set(words)}
    return out, accs
