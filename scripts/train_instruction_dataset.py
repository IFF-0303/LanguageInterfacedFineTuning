#!/usr/bin/env python
"""Utility script for fine-tuning instruction datasets with LoRA adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.lift.models.gptj.lora_gptj import LoRaConfigParams, LoRaQGPTJ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an instruction-style dataset (instruction/input/output) "
            "using LoRA adapters on top of a causal language model."
        )
    )
    parser.add_argument("--train-file", required=True, help="Path to the training dataset (JSON/JSONL).")
    parser.add_argument("--val-file", required=True, help="Path to the validation dataset (JSON/JSONL).")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Model identifier from Hugging Face or ModelScope.",
    )
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument("--output-dir", default="outputs/lora", help="Directory to store the trained adapter.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for fine-tuning.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Warmup steps for the scheduler.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit quantization through bitsandbytes (Hugging Face models only).",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Disable LoRA adapters and fine-tune all model parameters (requires more memory).",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="Rank of the LoRA decomposition.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability.")
    parser.add_argument(
        "--target-modules",
        nargs="*",
        default=None,
        help="Specific module names to apply LoRA to (defaults to attention/MLP projections).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_config = LoRaConfigParams(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    trainer = LoRaQGPTJ(
        model_name=args.model_name,
        adapter=not args.no_adapter,
        model_path=str(output_dir),
        load_in_4bit=args.load_in_4bit,
        lora_config=lora_config,
        model_provider=args.model_provider,
    )

    train_configs = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
    }

    train_losses, val_losses = trainer.finetune(
        args.train_file,
        args.val_file,
        train_configs=train_configs,
        saving_checkpoint=True,
    )

    trainer.save_networks(str(output_dir))

    print("Training complete. Final train/val losses:")
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
