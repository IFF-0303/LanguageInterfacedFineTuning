#!/usr/bin/env python
"""Run SHAP-based interpretability analysis for the feature extractor classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
            "Generate SHAP explanations for predictions made by the LLM feature extractor "
            "classifier."
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
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for SHAP evaluation.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of examples from the dataset to explain (processed in dataset order).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of tokens with the largest absolute SHAP values to display per example.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=500,
        help="Maximum number of model evaluations per example during SHAP computation.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional path to save an interactive SHAP report (HTML).",
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


def main() -> None:
    args = parse_args()

    examples = load_examples(args.data_file)
    if not examples:
        raise ValueError("No examples found in the dataset for analysis.")

    model = init_model(args)
    prompts = build_prompts(examples)

    num_to_explain = max(1, min(args.num_examples, len(prompts)))
    explain_prompts = prompts[:num_to_explain]

    # Pre-compute predictions for reporting and to ensure consistent class ordering.
    def predict_fn(texts: List[str]) -> np.ndarray:
        tokens = model.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(model.device)
        attention_mask = tokens["attention_mask"].to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    probs = predict_fn(explain_prompts)
    pred_ids = probs.argmax(axis=1)

    label_names = [model.id2label[idx] for idx in sorted(model.id2label.keys())]

    masker = shap.maskers.Text(model.tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=label_names)

    shap_values = explainer(
        explain_prompts,
        max_evals=args.max_evals,
        batch_size=args.batch_size,
    )

    values = shap_values.values
    data = shap_values.data

    print("Generated SHAP explanations for the following examples:\n")
    for idx, pred_idx in enumerate(pred_ids, start=1):
        print(f"Example {idx}: predicted label = {model.id2label[pred_idx]}")
        tokens = data[idx - 1]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        token_scores: np.ndarray
        if values.ndim == 3:
            token_scores = values[idx - 1, pred_idx]
        elif values.ndim == 2:
            token_scores = values[idx - 1]
        else:
            raise RuntimeError(
                f"Unexpected SHAP values shape {values.shape}; expected rank 2 or 3 for text explanations."
            )
        scores = list(zip(tokens, token_scores))
        scores.sort(key=lambda item: abs(float(item[1])), reverse=True)
        for token, score in scores[: args.top_k]:
            print(f"    {token!r:>15} : {score:+.4f}")
        print()

    if args.output_html:
        output_path = Path(args.output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.save_html(str(output_path), shap_values)
        print(f"Interactive SHAP report saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
