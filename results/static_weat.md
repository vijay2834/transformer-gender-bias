# WEAT on GloVe (wiki-gigaword, 50d, top 50,000 words)

Effect size d (Cohen's d style; |d| > 0.8 is large) with one-sided permutation p-value (10,000 permutations). Positive d means the first attribute set (career, math, science) is closer to male targets.

| Test | Original d (p) | Hard-Debias d (p) | INLP d (p) |
|---|---|---|---|
| WEAT 6: career vs family (names) | +1.67 (0.000) | +1.31 (0.004) | +1.03 (0.018) |
| WEAT 7: math vs arts (terms) | +1.50 (0.000) | -1.15 (0.992) | +0.87 (0.042) |
| WEAT 8: science vs arts (terms) | +1.59 (0.000) | -0.41 (0.774) | -0.83 (0.952) |
| *Utility: neighbour overlap@10* | 1.00 | 0.89 | 0.81 |

INLP gender-probe accuracy by iteration: 1.00, 0.70, 0.68, 0.68, 0.61, 0.62, 0.61, 0.55
