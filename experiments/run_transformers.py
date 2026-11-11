"""SEAT (sentence-template WEAT) on transformer encoders: BERT, GPT-2, RoBERTa, XLNet.

    pip install torch transformers
    python experiments/run_transformers.py --models bert-base-uncased gpt2 roberta-base xlnet-base-cased

Words are inserted into neutral templates ("This is <word>.") and mean-pooled, following
May et al. (2019). Results are written to results/transformer_seat.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genderbias import wordlists as W                 # noqa: E402
from genderbias.embeddings import TransformerEncoder   # noqa: E402
from genderbias.weat import weat                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base-uncased", "gpt2", "roberta-base", "xlnet-base-cased"])
    ap.add_argument("--perm", type=int, default=10_000)
    a = ap.parse_args()
    res = {}
    for m in a.models:
        enc = TransformerEncoder(m)
        res[m] = {}
        for name, (X, Y, A, B) in W.WEAT_TESTS.items():
            r = weat(*(enc.encode_words(ws) for ws in (X, Y, A, B)), n_perm=a.perm)
            res[m][name] = {"effect_size": round(r.effect_size, 3), "p_value": round(r.p_value, 4)}
        print(m, res[m])
    (ROOT / "results/transformer_seat.json").write_text(json.dumps(res, indent=2))
    lines = ["| Model | " + " | ".join(W.WEAT_TESTS) + " |", "|---" * (len(W.WEAT_TESTS) + 1) + "|"]
    for m, r in res.items():
        lines.append(f"| {m} | " + " | ".join(f"{r[t]['effect_size']:+.2f} ({r[t]['p_value']:.3f})" for t in W.WEAT_TESTS) + " |")
    (ROOT / "results/transformer_seat.md").write_text("# SEAT on transformer encoders\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
