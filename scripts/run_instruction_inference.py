#!/usr/bin/env python
"""CLI utility to run inference on instruction-style datasets with (LoRA) LLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.lift.models.gptj.lora_gptj import LoRaQGPTJ, build_instruction_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses for an instruction/input dataset using a fine-tuned model.",
    )
    parser.add_argument("--data-file", required=True, help="Path to the dataset to run inference on (JSON/JSONL).")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Base model identifier from Hugging Face or ModelScope.",
    )
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Directory containing the trained LoRA adapter weights (from fine-tuning).",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Disable LoRA adapters and run inference with the base model only.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization via bitsandbytes (Hugging Face models only).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of tokens to generate per example.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size to use during generation.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (only used when --deterministic is not set).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable sampling to produce deterministic outputs (greedy decoding).",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to save predictions (supports .json or .jsonl).",
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
        examples = [json.loads(line) for line in raw_content.splitlines() if line.strip()]
    except json.JSONDecodeError:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            examples = [parsed]
        elif isinstance(parsed, list):
            examples = parsed
        else:
            raise ValueError(
                "Unsupported dataset format: expected a JSON object, list, or JSONL entries.",
            )

    cleaned_examples: List[Dict[str, Any]] = []
    for example in examples:
        if not any(key in example for key in ("instruction", "prompt", "input", "context")):
            raise ValueError(
                "Each example must contain at least an 'instruction', 'prompt', 'input', or 'context' field.",
            )
        cleaned_examples.append(example)
    return cleaned_examples


def save_predictions(path: str, results: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in results:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if not args.no_adapter and not args.adapter_path:
        raise ValueError("--adapter-path must be provided unless --no-adapter is set.")

    model = LoRaQGPTJ(
        model_name=args.model_name,
        adapter=not args.no_adapter,
        model_path=args.adapter_path or args.model_name,
        load_in_4bit=args.load_in_4bit,
        model_provider=args.model_provider,
    )

    if args.adapter_path and not args.no_adapter:
        model.load_networks(args.adapter_path)

    examples = load_examples(args.data_file)
    prompts = [f"{build_instruction_prompt(example)} " for example in examples]

    generations = model.generate(
        prompts,
        deterministic=args.deterministic,
        max_token=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    if model.is_distributed and not model.is_main_process:
        return

    results: List[Dict[str, Any]] = []
    for example, prediction in zip(examples, generations):
        result = dict(example)
        result["prediction"] = prediction
        results.append(result)

    for idx, result in enumerate(results, start=1):
        instruction = result.get("instruction") or result.get("prompt") or "(no instruction)"
        print(f"[{idx:03d}] {instruction}\nPrediction: {result['prediction']}\n")

    if args.output_file:
        save_predictions(args.output_file, results)


if __name__ == "__main__":
    main()
