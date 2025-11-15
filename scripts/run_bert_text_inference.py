#!/usr/bin/env python
"""Run inference with a fine-tuned BERT text classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.lift.models.bert.text_classifier import (
    BertTextClassificationDataset,
    BertTextClassifier,
    BertTextClassifierConfig,
    extract_label,
    load_text_classification_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions using a BERT text classifier.")
    parser.add_argument("--input-file", required=True, help="Path to the dataset (JSON/JSONL/CSV).")
    parser.add_argument("--model-dir", required=True, help="Directory containing the fine-tuned model.")
    parser.add_argument("--output-file", help="Optional path to store predictions (JSON or JSONL).")

    parser.add_argument("--text-field", help="Optional text field to read from the dataset.")
    parser.add_argument("--label-field", help="Optional label field for evaluation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--no-cuda", action="store_true", help="Force execution on CPU even if CUDA is available.")
    return parser.parse_args()


def collate_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def save_predictions(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    model_dir = Path(args.model_dir)
    config_path = model_dir / "bert_text_classifier_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing configuration file at {config_path}. Ensure the model directory is correct."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)
    config = BertTextClassifierConfig.from_dict(config_data)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = BertTextClassifier.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    examples = load_text_classification_examples(Path(args.input_file), require_labels=False)

    gold_labels: List[Optional[str]] = []
    for example in examples:
        try:
            label = extract_label(example, explicit_field=args.label_field)
            gold_labels.append(label)
        except KeyError:
            gold_labels.append(None)
    has_labels = all(label is not None for label in gold_labels)

    dataset = BertTextClassificationDataset(
        examples,
        tokenizer,
        max_length=config.max_length,
        label2id=config.label2id if has_labels else None,
        text_field=args.text_field,
        label_field=args.label_field if has_labels else None,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    id2label = {idx: label for label, idx in config.label2id.items()}
    predictions: List[Dict[str, object]] = []
    total_correct = 0
    total_examples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
        batch_on_device = collate_to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch_on_device)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)

        for i in range(predicted_ids.size(0)):
            example_index = batch_idx * dataloader.batch_size + i
            if example_index >= len(examples):
                break
            pred_id = predicted_ids[i].item()
            pred_label = id2label.get(pred_id, str(pred_id))
            probs = probabilities[i].cpu().tolist()
            record: Dict[str, object] = {
                "prediction": pred_label,
                "probabilities": {
                    id2label.get(idx, str(idx)): float(probs[idx]) for idx in range(len(probs))
                },
            }
            gold_label = gold_labels[example_index]
            if gold_label is not None:
                record["label"] = gold_label
                if pred_label == gold_label:
                    total_correct += 1
            predictions.append(record)
            total_examples += 1

    if args.output_file:
        save_predictions(Path(args.output_file), predictions)

    if total_examples == 0:
        print("No examples processed.")
        return

    if has_labels:
        accuracy = total_correct / total_examples
        print(json.dumps({"accuracy": accuracy}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"num_examples": total_examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
