"""Counterfactual Data Augmentation (Lu et al., 2018; Zmigrod et al., 2019).

Each training sentence is paired with a gender-swapped copy, so a model sees
"she is a doctor" as often as "he is a doctor". Case is preserved. The ambiguous
"her" (his / him) is resolved with a light heuristic on the next token.
"""
from __future__ import annotations

import re

PAIRS = [("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"), ("man", "woman"),
         ("men", "women"), ("boy", "girl"), ("boys", "girls"), ("father", "mother"), ("fathers", "mothers"),
         ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"), ("brothers", "sisters"),
         ("husband", "wife"), ("husbands", "wives"), ("king", "queen"), ("kings", "queens"),
         ("uncle", "aunt"), ("nephew", "niece"), ("mr", "mrs"), ("male", "female"), ("gentleman", "lady"),
         ("actor", "actress"), ("waiter", "waitress"), ("grandfather", "grandmother")]
M2F = {m: f for m, f in PAIRS}
F2M = {f: m for m, f in PAIRS if f != "her"}
TOKEN = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
DETERMINERS_FOLLOW = re.compile(r"^[a-z]+$")


def _case(src: str, tgt: str) -> str:
    return tgt.upper() if src.isupper() and len(src) > 1 else tgt.capitalize() if src[0].isupper() else tgt


def swap(sentence: str) -> str:
    toks = TOKEN.findall(sentence)
    out = []
    for i, t in enumerate(toks):
        low = t.lower()
        if low == "her":
            nxt = next((x for x in toks[i + 1:] if x.strip()), "")
            # "her book" -> "his book"; "saw her." -> "saw him."
            rep = "his" if nxt and nxt[0].isalpha() and nxt.lower() not in M2F and nxt.lower() not in {"and", "or", "to", "with"} else "him"
            out.append(_case(t, rep))
        elif low in M2F:
            out.append(_case(t, M2F[low]))
        elif low in F2M:
            out.append(_case(t, F2M[low]))
        else:
            out.append(t)
    return "".join(out)


def augment(corpus: list[str], two_sided: bool = True) -> list[str]:
    swapped = [swap(s) for s in corpus]
    return corpus + swapped if two_sided else swapped
