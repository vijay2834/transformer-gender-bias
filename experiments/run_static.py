"""WEAT on GloVe before/after Hard-Debias and INLP.

    python experiments/run_static.py --glove data/glove-wiki-gigaword-50.gz
Writes results/static_weat.md, results/static_weat.json and results/weat_effects.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genderbias import wordlists as W                       # noqa: E402
from genderbias.debias import gender_direction, hard_debias, inlp   # noqa: E402
from genderbias.embeddings import load_glove                # noqa: E402
from genderbias.weat import weat                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_weat(emb, n_perm):
    out = {}
    for name, (X, Y, A, B) in W.WEAT_TESTS.items():
        vec = lambda ws: np.array([emb[w] for w in ws if w in emb])
        r = weat(vec(X), vec(Y), vec(A), vec(B), n_perm=n_perm)
        out[name] = {"effect_size": round(r.effect_size, 3), "p_value": round(r.p_value, 4)}
    return out


def nn_overlap(before, after, words, k=10):
    def nn(emb, w, vocab, M):
        v = emb[w] / np.linalg.norm(emb[w])
        sims = M @ v
        return set(np.array(vocab)[np.argsort(-sims)[1:k + 1]])
    vocab = list(before)[:20000]
    Mb = np.array([before[w] for w in vocab]); Mb /= np.linalg.norm(Mb, axis=1, keepdims=True)
    va = [w for w in vocab if w in after]
    Ma = np.array([after[w] for w in va]); Ma /= np.linalg.norm(Ma, axis=1, keepdims=True)
    return float(np.mean([len(nn(before, w, vocab, Mb) & nn(after, w, va, Ma)) / k for w in words]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", default=str(ROOT / "data/glove-wiki-gigaword-50.gz"))
    ap.add_argument("--vocab", type=int, default=50_000)
    ap.add_argument("--perm", type=int, default=10_000)
    a = ap.parse_args()

    emb = load_glove(a.glove, a.vocab)
    print(f"loaded {len(emb)} vectors of dim {len(next(iter(emb.values())))}")

    gender_specific = set(w for p in W.DEFINITIONAL_PAIRS + W.EQUALIZE_PAIRS for w in p) | \
        set(W.MALE_TERMS + W.FEMALE_TERMS + W.MALE_NAMES + W.FEMALE_NAMES)
    neutral = set(emb) - gender_specific
    hd = hard_debias(emb, neutral, W.DEFINITIONAL_PAIRS, W.EQUALIZE_PAIRS)

    # INLP training words: the most gender-polarised words along the PCA gender direction
    g = gender_direction(emb, W.DEFINITIONAL_PAIRS)
    vocab = list(emb)[:20000]
    proj = np.array([emb[w] @ g / np.linalg.norm(emb[w]) for w in vocab])
    order = np.argsort(proj)
    side_a, side_b = [vocab[i] for i in order[:1000]], [vocab[i] for i in order[-1000:]]
    female_side = side_a if np.mean([emb[w] @ g for w in ("she", "her", "woman") ]) < 0 else side_b
    male_side = side_b if female_side is side_a else side_a
    inlp_emb, accs = inlp(emb, female_side, male_side, list(emb), n_iter=25)

    variants = {"Original GloVe": emb, "Hard-Debias": hd, "INLP": inlp_emb}
    results = {k: run_weat(v, a.perm) for k, v in variants.items()}
    probe = [w for w in ["computer", "doctor", "river", "music", "economy", "football", "cooking", "science",
                         "nurse", "engineer", "city", "history", "family", "money", "art"] if w in emb]
    utility = {k: round(nn_overlap(emb, v, probe), 3) for k, v in variants.items()}
    results["_utility_nn_overlap@10"] = utility
    results["_inlp_gender_probe_accuracy"] = [round(x, 3) for x in accs]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/static_weat.json").write_text(json.dumps(results, indent=2))

    tests = list(W.WEAT_TESTS)
    lines = ["| Test | Original d (p) | Hard-Debias d (p) | INLP d (p) |", "|---|---|---|---|"]
    for t in tests:
        cells = [f"{results[v][t]['effect_size']:+.2f} ({results[v][t]['p_value']:.3f})" for v in variants]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append(f"| *Utility: neighbour overlap@10* | {utility['Original GloVe']:.2f} | {utility['Hard-Debias']:.2f} | {utility['INLP']:.2f} |")
    md = (f"# WEAT on GloVe (wiki-gigaword, 50d, top {a.vocab:,} words)\n\n"
          "Effect size d (Cohen's d style; |d| > 0.8 is large) with one-sided permutation p-value "
          f"({a.perm:,} permutations). Positive d means the first attribute set (career, math, science) is closer to male targets.\n\n"
          + "\n".join(lines) +
          f"\n\nINLP gender-probe accuracy by iteration: {', '.join(f'{x:.2f}' for x in accs)}\n")
    (ROOT / "results/static_weat.md").write_text(md)
    print(md)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tests)); wbar = 0.26
    for i, (v, c) in enumerate(zip(variants, ["#c0392b", "#2980b9", "#27ae60"])):
        ax.bar(x + (i - 1) * wbar, [results[v][t]["effect_size"] for t in tests], wbar, label=v, color=c)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x, [t.split(":")[0] + "\n" + t.split(": ")[1] for t in tests], fontsize=9)
    ax.set_ylabel("WEAT effect size d"); ax.set_title("Gender bias in GloVe before and after debiasing"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(ROOT / "results/weat_effects.png", dpi=140)


if __name__ == "__main__":
    main()
