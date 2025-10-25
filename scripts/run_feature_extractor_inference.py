#!/usr/bin/env python
"""Run inference using a frozen LLM feature extractor with a trained classifier head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from src.lift.models.gptj.feature_extractor_classifier import (
    ClassifierHeadConfig,
    InstructionClassificationDataset,
    LLMFeatureExtractorClassifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions for instruction-style classification datasets using a frozen LLM "
            "feature extractor and a trained MLP head."
        )
    )
    parser.add_argument("--data-file", required=True, help="Path to the dataset (JSON/JSONL).")
    parser.add_argument(
        "--classifier-dir",
        required=True,
        help="Directory containing classifier.pt and classifier_config.json from training.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct", help="Backbone model identifier.")
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument("--adapter-path", help="Optional directory containing LoRA adapters to load.")
    parser.add_argument("--no-adapter", action="store_true", help="Disable LoRA adapters and use the base model only.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantisation for the backbone model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for feature extraction.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")
    parser.add_argument(
        "--label-field",
        help=(
            "Optional explicit label field for datasets containing gold labels. "
            "Predictions are generated regardless of labels being present."
        ),
    )
    parser.add_argument(
        "--output-file",
        help="Optional file to store predictions (supports .json or .jsonl).",
    )
    return parser.parse_args()


def load_examples(file_path: str) -> List[Dict[str, Any]]:
    data_path = Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' does not exist.")

    raw_content = data_path.read_text(encoding="utf-8").strip()
    if not raw_content:
        raise ValueError(f"Dataset file '{file_path}' is empty.")

    try:
        return [json.loads(line) for line in raw_content.splitlines() if line.strip()]
    except json.JSONDecodeError:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        raise ValueError(
            "Unsupported dataset format: expected a JSON object, list, or newline-delimited entries."
        )


def save_predictions(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    else:
        output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    metadata = LLMFeatureExtractorClassifier.load_metadata(args.classifier_dir)
    head_cfg = ClassifierHeadConfig.from_dict(metadata.get("head_config", {}))
    label2id = metadata.get("label2id", {})
    if not label2id:
        raise ValueError("Classifier metadata missing label2id mapping.")

    saved_backbone = metadata.get("backbone_dir")
    backbone_path: Path | None = None
    if saved_backbone:
        candidate = Path(saved_backbone)
        if not candidate.is_absolute():
            candidate = Path(args.classifier_dir) / candidate
        if candidate.exists():
            backbone_path = candidate
        else:
            print(
                f"Warning: Saved backbone directory '{candidate}' not found. Falling back to CLI arguments."
            )

    use_lora = metadata.get("use_lora", not args.no_adapter)
    if args.no_adapter:
        use_lora = False

    model = LLMFeatureExtractorClassifier(
        model_name=args.model_name,
        adapter=use_lora,
        model_path=args.adapter_path or args.model_name,
        load_in_4bit=args.load_in_4bit,
        model_provider=args.model_provider,
        num_labels=len(label2id),
        classifier_config=head_cfg,
    )

    if backbone_path is not None:
        model.load_networks(str(backbone_path))
        model.freeze_backbone()
    elif args.adapter_path and not args.no_adapter:
        model.load_networks(args.adapter_path)
        model.freeze_backbone()

    model.load_classifier(args.classifier_dir)

    examples = load_examples(args.data_file)

    dataset = InstructionClassificationDataset(
        examples,
        model.tokenizer,
        label2id=None,
        label_field=args.label_field,
        max_length=args.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions: List[str] = []
    for batch in dataloader:
        input_ids = batch[0].to(model.device)
        attention_mask = batch[1].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(model.id2label[idx] for idx in pred_ids)

    results: List[Dict[str, Any]] = []
    for example, prediction in zip(examples, predictions):
        row = dict(example)
        row["prediction"] = prediction
        results.append(row)

    for idx, row in enumerate(results, start=1):
        print(f"[{idx:03d}] Prediction: {row['prediction']}")

    if args.output_file:
        save_predictions(args.output_file, results)


if __name__ == "__main__":
    main()
