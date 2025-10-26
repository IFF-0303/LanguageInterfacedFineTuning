#!/usr/bin/env python
"""Run Captum-based interpretability analysis for the feature extractor classifier."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from captum.attr import LayerIntegratedGradients

from src.lift.models.gptj.feature_extractor_classifier import (
    ClassifierHeadConfig,
    LLMFeatureExtractorClassifier,
)
from src.lift.models.gptj.lora_gptj import build_instruction_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate token-level attributions for classifier predictions using Captum's "
            "Layer Integrated Gradients."
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
    parser.add_argument("--num-examples", type=int, default=8, help="Number of examples from the dataset to explain.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of tokens with the largest attributions to display.")
    parser.add_argument("--steps", type=int, default=50, help="Number of integration steps for Layer Integrated Gradients.")
    parser.add_argument(
        "--internal-batch-size",
        type=int,
        default=None,
        help="Optional internal batch size used during attribution computation.",
    )
    parser.add_argument(
        "--baseline-token",
        default="pad",
        help=(
            "Token used to construct the baseline sequence. Use 'pad' for the tokenizer pad token, "
            "'zero' for zero embeddings, or provide a literal token string present in the tokenizer vocabulary."
        ),
    )
    parser.add_argument(
        "--target-label",
        help=(
            "Optional label to explain instead of the model prediction. Provide either the label name "
            "or its integer id."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save a JSON report containing token attributions for each analysed example.",
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


def resolve_target_indices(
    target_label: str | None, predictions: Sequence[int], model: LLMFeatureExtractorClassifier
) -> List[int]:
    if target_label is None:
        return [int(idx) for idx in predictions]

    try:
        target_id = int(target_label)
        if target_id not in model.id2label:
            raise ValueError(f"Target label id {target_id} not present in classifier mapping.")
        return [target_id for _ in predictions]
    except ValueError:
        # Provided target_label is not an integer, treat as label name
        if target_label not in model.label2id:
            raise ValueError(
                f"Unknown target label {target_label!r}. Available labels: {sorted(model.label2id.keys())}."
            )
        resolved = model.label2id[target_label]
        return [int(resolved) for _ in predictions]


def build_baseline(
    input_ids: torch.Tensor, tokenizer, baseline_token: str, device: torch.device
) -> torch.Tensor:
    if baseline_token.lower() == "zero":
        return torch.zeros_like(input_ids, device=device)

    if baseline_token.lower() == "pad":
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer does not define a pad token; provide --baseline-token explicitly.")
        return torch.full_like(input_ids, pad_id, device=device)

    encoded = tokenizer.encode(baseline_token, add_special_tokens=False)
    if not encoded:
        raise ValueError(
            f"Baseline token {baseline_token!r} is not part of the tokenizer vocabulary; no ids were produced."
        )
    if len(encoded) > 1:
        raise ValueError(
            "Baseline token string must map to exactly one token id. Provide a shorter token or use 'pad'."
        )
    token_id = int(encoded[0])
    return torch.full_like(input_ids, token_id, device=device)


def tokens_from_ids(input_ids: torch.Tensor, tokenizer) -> List[List[str]]:
    tokens: List[List[str]] = []
    for row in input_ids.tolist():
        tokens.append(tokenizer.convert_ids_to_tokens(row))
    return tokens


def aggregate_attributions(attributions: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    summed = attributions.sum(dim=-1)
    return summed * attention_mask


@contextmanager
def temporarily_enable_backbone_gradients(model: LLMFeatureExtractorClassifier):
    """Ensure the backbone runs with gradients even if it was initialised as frozen."""

    was_frozen = getattr(model, "backbone_frozen", False)
    try:
        if was_frozen:
            # The flag gates a ``torch.no_grad`` region inside ``extract_features``; toggling
            # it off avoids silently detaching the backbone activations that Captum needs.
            model.backbone_frozen = False
        yield
    finally:
        if was_frozen:
            model.backbone_frozen = True


def main() -> None:
    args = parse_args()

    examples = load_examples(args.data_file)
    if not examples:
        raise ValueError("No examples found in the dataset for analysis.")

    model = init_model(args)
    prompts = build_prompts(examples)

    num_to_explain = max(1, min(args.num_examples, len(prompts)))
    explain_prompts = prompts[:num_to_explain]

    tokens = model.tokenizer(
        explain_prompts,
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
    pred_ids = probs.argmax(dim=-1)

    target_indices = resolve_target_indices(args.target_label, pred_ids.tolist(), model)

    baseline = build_baseline(input_ids, model.tokenizer, args.baseline_token, model.device)

    embedding_layer = model._unwrap_model().get_input_embeddings()
    original_requires_grad = getattr(embedding_layer.weight, "requires_grad", False)

    def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids, attention_mask=mask)

    tokens_per_example = tokens_from_ids(input_ids.detach().cpu(), model.tokenizer)
    attention_mask_cpu = attention_mask.detach().cpu()

    attribution_results: List[Dict[str, Any]] = []

    with temporarily_enable_backbone_gradients(model):
        if not original_requires_grad:
            embedding_layer.weight.requires_grad_(True)

        lig = LayerIntegratedGradients(forward_func, embedding_layer)

        for i in range(num_to_explain):
            ids_i = input_ids[i : i + 1]
            mask_i = attention_mask[i : i + 1]
            baseline_i = baseline[i : i + 1]
            target_i = target_indices[i]

            attr_i, delta_i = lig.attribute(
                inputs=ids_i,
                baselines=baseline_i,
                additional_forward_args=(mask_i,),
                target=target_i,
                n_steps=args.steps,
                internal_batch_size=args.internal_batch_size,
                return_convergence_delta=True,
            )

            aggregated_i = aggregate_attributions(attr_i, mask_i).detach().cpu().squeeze(0)
            delta_value = (
                float(delta_i.detach().cpu().item())
                if isinstance(delta_i, torch.Tensor)
                else None
            )

            attribution_results.append(
                {
                    "aggregated": aggregated_i,
                    "delta": delta_value,
                }
            )

            # Release tensors tied to the current iteration to free GPU memory eagerly.
            del attr_i, delta_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not original_requires_grad:
        embedding_layer.weight.requires_grad_(False)

    label_names = {idx: label for idx, label in model.id2label.items()}

    output_records: List[Dict[str, Any]] = []
    print("Generated Captum attributions for the following examples:\n")

    for i in range(num_to_explain):
        pred_idx = int(pred_ids[i].item())
        target_idx = int(target_indices[i])
        tokens_row = tokens_per_example[i]
        mask_row = attention_mask_cpu[i].tolist()
        scores_row = attribution_results[i]["aggregated"].tolist()

        scored_tokens = [
            {"token": token, "score": float(score)}
            for token, score, mask_val in zip(tokens_row, scores_row, mask_row)
            if mask_val > 0
        ]
        scored_tokens.sort(key=lambda item: abs(item["score"]), reverse=True)

        print(f"Example {i + 1}: predicted label = {label_names.get(pred_idx, pred_idx)}")
        print(f"  Target label explained = {label_names.get(target_idx, target_idx)}")
        print(f"  Predicted probability   = {float(probs[i, pred_idx].item()):.4f}")
        delta_value = attribution_results[i]["delta"]
        if delta_value is not None:
            print(f"  Convergence delta       = {delta_value:.6f}")
        print("  Top tokens:")
        for item in scored_tokens[: args.top_k]:
            print(f"    {item['token']!r:>15} : {item['score']:+.4f}")
        print()

        output_records.append(
            {
                "example_index": i,
                "prompt": explain_prompts[i],
                "predicted_label": label_names.get(pred_idx, str(pred_idx)),
                "target_label": label_names.get(target_idx, str(target_idx)),
                "predicted_probability": float(probs[i, pred_idx].item()),
                "convergence_delta": delta_value,
                "token_attributions": scored_tokens,
            }
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output_records, handle, ensure_ascii=False, indent=2)
        print(f"Attribution report saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
