#!/usr/bin/env python
"""Run inference for the LLM + structured fusion classifier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score

from src.lift.models.gptj.feature_extractor_classifier import ClassifierHeadConfig
from src.lift.models.gptj.llm_structured_fusion_classifier import (
    LLMStructuredFusionClassifier,
    StructuredEncoderConfig,
    StructuredFusionDataset,
)
from scripts.health_csv_utils import load_health_dataset_from_csv


TAB_FEATURES_FIELD = "tab_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate predictions for datasets that include both textual prompts and "
            "structured numeric/categorical features using a fusion classifier."
        )
    )
    parser.add_argument("--data-file", required=True, help="Path to the dataset (JSON/JSONL/CSV).")
    parser.add_argument(
        "--classifier-dir",
        required=True,
        help="Directory containing classifier.pt and classifier_config.json.",
    )
    parser.add_argument("--output-file", help="Optional file to store predictions (JSON or JSONL).")

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

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")
    parser.add_argument("--label-field", help="Optional explicit label field for evaluation.")
    return parser.parse_args()


def load_examples(
    file_path: str,
    *,
    numeric_fields: Sequence[str],
    categorical_fields: Sequence[str],
    tab_features_field: str = TAB_FEATURES_FIELD,
    require_label: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data_path = Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' does not exist.")

    if data_path.suffix.lower() == ".csv":
        (
            examples,
            inferred_label_field,
            feature_columns,
            _,
        ) = load_health_dataset_from_csv(
            data_path,
            tab_features_field=tab_features_field,
            feature_columns=None,
            require_label=require_label,
        )

        column_lookup = {column: idx for idx, column in enumerate(feature_columns)}
        required_columns = list(dict.fromkeys(list(numeric_fields) + list(categorical_fields)))
        for example in examples:
            tab_values = example.get(tab_features_field)
            if not isinstance(tab_values, (list, tuple)):
                continue
            for column in required_columns:
                idx = column_lookup.get(column)
                if idx is None or idx >= len(tab_values):
                    continue
                value = tab_values[idx]
                if isinstance(value, float) and math.isnan(value):
                    example[column] = None
                else:
                    example[column] = value

        return examples, inferred_label_field

    raw_content = data_path.read_text(encoding="utf-8").strip()
    if not raw_content:
        raise ValueError(f"Dataset file '{file_path}' is empty.")

    try:
        examples = [json.loads(line) for line in raw_content.splitlines() if line.strip()]
    except json.JSONDecodeError:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            examples = [parsed]
        elif isinstance(parsed, list):
            examples = parsed
        else:
            raise ValueError(
                "Unsupported dataset format: expected a JSON object, list, or newline-delimited entries."
            )

    return examples, None


def build_dataset(
    *,
    examples: List[Dict[str, Any]],
    tokenizer,
    numeric_fields: Sequence[str],
    categorical_fields: Sequence[str],
    numeric_stats: Dict[str, Sequence[float]],
    categorical_vocabs: Dict[str, Dict[str, int]],
    categorical_missing_indices: Dict[str, int],
    label2id: Optional[Dict[str, int]],
    label_field: Optional[str],
    max_length: int,
) -> StructuredFusionDataset:
    return StructuredFusionDataset(
        examples,
        tokenizer,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        numeric_stats=numeric_stats,
        categorical_vocabs=categorical_vocabs,
        categorical_missing_indices=categorical_missing_indices,
        label2id=label2id,
        label_field=label_field,
        max_length=max_length,
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

    metadata = LLMStructuredFusionClassifier.load_metadata(args.classifier_dir)
    head_cfg = ClassifierHeadConfig.from_dict(metadata.get("head_config", {}))
    label2id = metadata.get("label2id", {})
    if not label2id:
        raise ValueError("Classifier metadata missing label2id mapping.")
    id2label = {int(idx): label for label, idx in label2id.items()}

    structured_config_data = metadata.get("structured_config", {})
    structured_config = StructuredEncoderConfig.from_dict(structured_config_data)

    structured_metadata = metadata.get("structured_metadata", {})
    numeric_fields = structured_metadata.get("numeric_fields", [])
    categorical_fields = structured_metadata.get("categorical_fields", [])
    numeric_stats = structured_metadata.get("numeric_stats", {"mean": [], "std": []})
    categorical_vocabs = structured_metadata.get("categorical_vocabs", {})
    categorical_missing_indices = structured_metadata.get("categorical_missing_indices", {})

    label_field = args.label_field
    if not label_field:
        training_args = metadata.get("training_args", {})
        label_field = training_args.get("label_field")

    model = LLMStructuredFusionClassifier(
        num_labels=len(label2id),
        classifier_config=head_cfg,
        structured_config=structured_config,
        fusion_mode=metadata.get("fusion_mode", "concat"),
        freeze_backbone=True,
        wrap_backbone_with_ddp=True,
        model_name=args.model_name,
        model_provider=args.model_provider,
        adapter_path=args.adapter_path,
        adapter=not args.no_adapter,
        load_in_4bit=args.load_in_4bit,
    )
    model.set_label_mapping({label: int(idx) for label, idx in label2id.items()})
    model.load_classifier(args.classifier_dir)
    model.eval()

    require_label = bool(label_field)
    examples, inferred_label_field = load_examples(
        args.data_file,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        tab_features_field=TAB_FEATURES_FIELD,
        require_label=require_label,
    )
    if not label_field:
        label_field = inferred_label_field
    dataset = build_dataset(
        examples=examples,
        tokenizer=model.tokenizer,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        numeric_stats=numeric_stats,
        categorical_vocabs=categorical_vocabs,
        categorical_missing_indices=categorical_missing_indices,
        label2id=label2id if label_field else None,
        label_field=label_field,
        max_length=args.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    predictions: List[Dict[str, Any]] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            if dataset.labels is None:
                input_ids, attention_mask, numeric_features, numeric_mask, categorical_features, categorical_mask = batch
                labels = None
            else:
                (
                    input_ids,
                    attention_mask,
                    numeric_features,
                    numeric_mask,
                    categorical_features,
                    categorical_mask,
                    labels,
                ) = batch

            input_ids = input_ids.to(model.device, non_blocking=True)
            attention_mask = attention_mask.to(model.device, non_blocking=True)
            numeric_features = numeric_features.to(model.device, non_blocking=True)
            numeric_mask = numeric_mask.to(model.device, non_blocking=True)
            categorical_features = categorical_features.to(model.device, non_blocking=True)
            categorical_mask = categorical_mask.to(model.device, non_blocking=True)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features,
                numeric_mask=numeric_mask,
                categorical_features=categorical_features,
                categorical_mask=categorical_mask,
            )
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

            for idx in range(pred_ids.size(0)):
                row: Dict[str, Any] = {
                    "prediction": id2label[pred_ids[idx].item()],
                    "probabilities": {id2label[i]: float(probs[idx, i].item()) for i in range(probs.size(1))},
                }
                if labels is not None:
                    row["label"] = id2label[int(labels[idx].item())]
                    all_targets.append(int(labels[idx].item()))
                predictions.append(row)
                all_probs.append([float(p) for p in probs[idx].tolist()])

    if args.output_file:
        save_predictions(args.output_file, predictions)
    else:
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

    if dataset.labels is not None:
        pred_ids = [label2id[row["prediction"]] for row in predictions]
        metrics: Dict[str, Any] = {
            "accuracy": accuracy_score(all_targets, pred_ids),
        }
        try:
            if len(all_probs[0]) == 2:
                metrics["roc_auc"] = roc_auc_score(all_targets, [p[1] for p in all_probs])
            else:
                metrics["roc_auc"] = roc_auc_score(all_targets, all_probs, multi_class="ovo")
        except ValueError:
            metrics["roc_auc"] = float("nan")
        print(json.dumps({"metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
