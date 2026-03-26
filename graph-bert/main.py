import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SOPDataset(Dataset):
    """
    Expects a csv/jsonl with fields:
      - query: str
      - sop: str
      - label: float in [0,1]  (for training/eval)
    """

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 256,
        with_labels: bool = True,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        query = str(item["query"]).strip()
        sop = str(item["sop"]).strip()

        encoded = self.tokenizer(
            query,
            sop,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        if self.with_labels:
            # Regression target: 0~1 usefulness score
            label = float(item["label"])
            encoded["labels"] = label

        return encoded


@dataclass
class ParsedData:
    train_records: Optional[List[Dict]] = None
    val_records: Optional[List[Dict]] = None
    test_records: Optional[List[Dict]] = None
    infer_records: Optional[List[Dict]] = None


def load_records(path: str) -> List[Dict]:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.to_dict("records")
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported file format: {path}. Use .csv or .jsonl")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # regression head gives shape [N, 1]
    preds = np.squeeze(preds)
    labels = np.squeeze(labels)

    # clamp predictions to [0, 1] for score-like interpretation
    preds = np.clip(preds, 0.0, 1.0)

    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))

    # Optional classification-style metrics for quick inspection
    pred_bin = (preds >= 0.5).astype(int)
    label_bin = (labels >= 0.5).astype(int)

    metrics = {
        "mse": mse,
        "mae": mae,
        "acc@0.5": accuracy_score(label_bin, pred_bin),
        "f1@0.5": f1_score(label_bin, pred_bin, zero_division=0),
    }

    # Some splits may contain only one class
    try:
        metrics["roc_auc"] = roc_auc_score(label_bin, preds)
    except Exception:
        pass

    try:
        metrics["pr_auc"] = average_precision_score(label_bin, preds)
    except Exception:
        pass

    return metrics


def save_predictions(records: List[Dict], scores: np.ndarray, out_path: str) -> None:
    rows = []
    for item, score in zip(records, scores):
        rows.append(
            {
                "query": item["query"],
                "sop": item["sop"],
                "score": float(np.clip(score, 0.0, 1.0)),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {out_path}")


def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    return tokenizer, model


def train(args):
    set_seed(args.seed)
    tokenizer, model = build_model_and_tokenizer(args.model_name)

    train_records = load_records(args.train_file)
    val_records = load_records(args.val_file) if args.val_file else None

    train_ds = SOPDataset(train_records, tokenizer, args.max_length, with_labels=True)
    val_ds = SOPDataset(val_records, tokenizer, args.max_length, with_labels=True) if val_records else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="mae",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if val_ds else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


@torch.no_grad()
def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    infer_records = load_records(args.infer_file)
    infer_ds = SOPDataset(infer_records, tokenizer, args.max_length, with_labels=False)

    device = next(model.parameters()).device
    scores = []

    for item in infer_ds:
        batch = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in item.items()}
        outputs = model(**batch)
        score = outputs.logits.squeeze().detach().cpu().item()
        score = max(0.0, min(1.0, float(score)))
        scores.append(score)

    save_predictions(infer_records, np.array(scores), args.out_file)


def predict_one(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    encoded = tokenizer(
        args.query,
        args.sop,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        score = outputs.logits.squeeze().item()
        score = max(0.0, min(1.0, float(score)))

    print(json.dumps({"query": args.query, "sop": args.sop, "score": score}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="BERT reranker for Query-SOP usefulness scoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_file", type=str, required=True, help="CSV/JSONL with query,sop,label")
    train_parser.add_argument("--val_file", type=str, default=None, help="Optional validation CSV/JSONL")
    train_parser.add_argument("--model_name", type=str, default="hfl/chinese-macbert-base")
    train_parser.add_argument("--output_dir", type=str, default="./outputs/macbert-sop-reranker")
    train_parser.add_argument("--max_length", type=int, default=256)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--learning_rate", type=float, default=2e-5)
    train_parser.add_argument("--weight_decay", type=float, default=0.01)
    train_parser.add_argument("--logging_steps", type=int, default=20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no_fp16", action="store_true")

    # infer on file
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model_dir", type=str, required=True)
    infer_parser.add_argument("--infer_file", type=str, required=True, help="CSV/JSONL with query,sop")
    infer_parser.add_argument("--out_file", type=str, default="./predictions.csv")
    infer_parser.add_argument("--max_length", type=int, default=256)

    # predict one pair
    one_parser = subparsers.add_parser("predict_one")
    one_parser.add_argument("--model_dir", type=str, required=True)
    one_parser.add_argument("--query", type=str, required=True)
    one_parser.add_argument("--sop", type=str, required=True)
    one_parser.add_argument("--max_length", type=int, default=256)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "predict_one":
        predict_one(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
