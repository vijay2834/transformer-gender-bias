"""Mitigation for contextual models: continue MLM pre-training on a CDA-augmented corpus.

    python experiments/cda_finetune.py --model bert-base-uncased --corpus corpus.txt --out models/bert-cda
then re-run run_transformers.py --models models/bert-cda to measure the change in SEAT effect sizes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from genderbias.cda import augment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--corpus", required=True, help="one sentence per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()

    from datasets import Dataset
    from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)
    sents = [l.strip() for l in open(a.corpus, encoding="utf8") if l.strip()]
    ds = Dataset.from_dict({"text": augment(sents)})
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])
    model = AutoModelForMaskedLM.from_pretrained(a.model)
    Trainer(model=model, args=TrainingArguments(a.out, num_train_epochs=a.epochs, per_device_train_batch_size=32,
                                                 learning_rate=2e-5, save_strategy="no", report_to=[]),
            train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm_probability=0.15)).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
