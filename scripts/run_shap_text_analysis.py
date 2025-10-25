#!/usr/bin/env python
"""Utility script for running SHAP interpretability analysis on classifier predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import shap
import torch

from src.lift.models.gptj.feature_extractor_classifier import (
    ClassifierHeadConfig,
    LLMFeatureExtractorClassifier,
)
from src.lift.models.gptj.lora_gptj import build_instruction_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SHAP explanations for the feature extractor classifier and optionally "
            "export them to a JSON report."
        )
    )
    parser.add_argument("--data-file", required=True, help="Path to the dataset file (JSON/JSONL).")
    parser.add_argument(
        "--classifier-dir",
        required=True,
        help="Directory that contains classifier.pt and classifier_config.json produced during training.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct", help="Backbone model identifier.")
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument("--adapter-path", help="Optional path to the LoRA adapters.")
    parser.add_argument("--no-adapter", action="store_true", help="Disable LoRA adapters and use the base model only.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantisation for the backbone model.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for SHAP evaluation.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of examples from the dataset to explain (processed in dataset order).",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of tokens with highest absolute SHAP value to retain per example."
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=500,
        help="Maximum number of model evaluations per example used by the SHAP explainer.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to export the explanations as JSON for downstream processing.",
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


def build_prompts(examples: Iterable[Dict[str, Any]]) -> List[str]:
    prompts: List[str] = []
    for example in examples:
        prompt = build_instruction_prompt(example)
        prompts.append(prompt.strip())
    return prompts


def init_model(args: argparse.Namespace) -> LLMFeatureExtractorClassifier:
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
        wrap_backbone_with_ddp=False,
    )

    if backbone_path is not None:
        model.load_networks(str(backbone_path))
        model.freeze_backbone()
    elif args.adapter_path and not args.no_adapter:
        model.load_networks(args.adapter_path)
        model.freeze_backbone()

    model.load_classifier(args.classifier_dir)
    model.eval()
    return model


def predict(model: LLMFeatureExtractorClassifier, texts: Sequence[str], max_length: int) -> np.ndarray:
    tokens = model.tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(model.device)
    attention_mask = tokens["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()


def summarise_shap(
    shap_values: shap.Explanation,
    label_names: Sequence[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    values = shap_values.values
    tokens_data = shap_values.data

    summaries: List[Dict[str, Any]] = []
    for idx in range(values.shape[0]):
        entry: Dict[str, Any] = {"tokens": list(tokens_data[idx])}
        per_label: Dict[str, List[Dict[str, float]]] = {}
        if values.ndim == 3:
            for label_idx, label in enumerate(label_names):
                token_scores = values[idx, label_idx]
                ranked = sorted(
                    zip(tokens_data[idx], token_scores),
                    key=lambda item: abs(float(item[1])),
                    reverse=True,
                )
                per_label[label] = [
                    {"token": token, "score": float(score)} for token, score in ranked[:top_k]
                ]
        elif values.ndim == 2:
            token_scores = values[idx]
            ranked = sorted(
                zip(tokens_data[idx], token_scores),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )
            per_label[label_names[0] if label_names else "label"] = [
                {"token": token, "score": float(score)} for token, score in ranked[:top_k]
            ]
        else:
            raise RuntimeError(
                f"Unexpected SHAP values shape {values.shape}; expected rank 2 or 3 for text explanations."
            )

        entry["top_tokens"] = per_label
        summaries.append(entry)
    return summaries


def main() -> None:
    args = parse_args()

    examples = load_examples(args.data_file)
    if not examples:
        raise ValueError("No examples found in the dataset for analysis.")

    model = init_model(args)
    prompts = build_prompts(examples)

    num_to_explain = max(1, min(args.num_examples, len(prompts)))
    explain_prompts = prompts[:num_to_explain]

    label_names = [model.id2label[idx] for idx in sorted(model.id2label.keys())]

    masker = shap.maskers.Text(model.tokenizer)
    explainer = shap.Explainer(
        lambda inputs: predict(model, inputs, args.max_length),
        masker,
        output_names=label_names,
    )

    shap_values = explainer(
        explain_prompts,
        max_evals=args.max_evals,
        batch_size=args.batch_size,
    )

    summaries = summarise_shap(shap_values, label_names, args.top_k)

    for idx, prompt in enumerate(explain_prompts, start=1):
        prediction = predict(model, [prompt], args.max_length)[0]
        pred_label = label_names[int(np.argmax(prediction))]
        print(f"Example {idx}: predicted label = {pred_label}")
        for label, tokens in summaries[idx - 1]["top_tokens"].items():
            print(f"  Label: {label}")
            for token_info in tokens:
                print(f"    {token_info['token']!r:>15} : {token_info['score']:+.4f}")
        print()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "examples": explain_prompts,
            "label_names": label_names,
            "summaries": summaries,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved SHAP explanations to {output_path.resolve()}")


if __name__ == "__main__":
    main()
