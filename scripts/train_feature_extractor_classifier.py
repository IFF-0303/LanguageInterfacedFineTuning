#!/usr/bin/env python
"""Train a frozen LLM feature extractor with a lightweight classification head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm import tqdm

from src.lift.models.gptj.feature_extractor_classifier import (
    ClassifierHeadConfig,
    InstructionClassificationDataset,
    LLMFeatureExtractorClassifier,
)
from src.lift.models.gptj.samplers import DistributedWeightedRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a small MLP head on top of an instruction-tuned language model, "
            "optionally updating the backbone."
        )
    )
    parser.add_argument("--train-file", required=True, help="Path to the training dataset (JSON/JSONL).")
    parser.add_argument("--val-file", required=True, help="Path to the validation dataset (JSON/JSONL).")
    parser.add_argument("--output-dir", required=True, help="Directory to store the classifier head and metadata.")

    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct", help="Backbone model identifier.")
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument("--adapter-path", help="Optional directory containing pre-trained LoRA adapters.")
    parser.add_argument("--no-adapter", action="store_true", help="Disable LoRA adapters entirely.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantisation for the backbone.")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Allow gradients to update all backbone parameters (full fine-tuning).",
    )
    parser.add_argument(
        "--train-lora",
        action="store_true",
        help="Fine-tune only LoRA adapter parameters in the backbone.",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for classifier training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for the MLP head.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="Optional learning rate for the backbone when --train-backbone is enabled."
        " Defaults to --learning-rate.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup steps for the scheduler.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when adapters are enabled.")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling factor (alpha) when adapters are enabled.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability applied inside LoRA adapters.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        help=(
            "Optional explicit module names to target with LoRA. If omitted, defaults from LoRaQGPTJ are used."
        ),
    )

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Hidden dimensions for the classifier head (e.g. --hidden-dims 512 128).",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability applied between hidden layers.")
    parser.add_argument(
        "--activation",
        choices=["relu", "gelu", "tanh"],
        default="gelu",
        help="Activation function for the classifier head.",
    )
    parser.add_argument(
        "--label-field",
        help=(
            "Optional explicit field name containing labels. "
            "If not provided the script searches for 'label', 'target', 'answer', 'category', or 'class'."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "sqrt_inv"],
        default="none",
        help=(
            "Strategy for rebalancing class distributions. "
            "Set to 'balanced' to inversely scale by class frequency or 'sqrt_inv' for a milder adjustment."
        ),
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


def build_label_mapping(examples: Iterable[Dict[str, Any]], label_field: str | None = None) -> Dict[str, int]:
    labels: List[str] = []
    for example in examples:
        if label_field:
            if label_field not in example:
                raise KeyError(
                    f"Specified label field '{label_field}' missing from example: {example.keys()}"
                )
            labels.append(str(example[label_field]))
            continue

        for key in ("label", "target", "answer", "category", "class"):
            if key in example:
                labels.append(str(example[key]))
                break
        else:
            raise KeyError(
                "Could not infer label from example. Provide --label-field or include one of "
                "'label', 'target', 'answer', 'category', or 'class'."
            )

    unique = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique)}


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def evaluate(
    model: LLMFeatureExtractorClassifier,
    forward_module: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    distributed: bool,
) -> Dict[str, float]:
    """Evaluate the model on a validation dataloader and return aggregate metrics."""

    model.eval()
    forward_module.eval()

    loss_total = torch.zeros((), dtype=torch.float64, device=model.device)
    correct_total = torch.zeros((), dtype=torch.float64, device=model.device)
    samples_total = torch.zeros((), dtype=torch.float64, device=model.device)

    for batch in dataloader:
        input_ids = batch[0].to(model.device, non_blocking=True)
        attention_mask = batch[1].to(model.device, non_blocking=True)
        labels = batch[2].to(model.device, non_blocking=True)

        with torch.no_grad():
            logits = forward_module(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        correct_total += (preds == labels).sum().to(torch.float64)
        samples_total += labels.size(0)
        loss_total += loss.detach().to(torch.float64) * labels.size(0)

    stats = torch.stack([loss_total, correct_total, samples_total])
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_loss, correct, total = stats.tolist()
    return {
        "loss": total_loss / max(total, 1.0),
        "accuracy": correct / max(total, 1.0),
    }


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "10086")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def build_dataloaders(
    train_dataset: InstructionClassificationDataset,
    val_dataset: InstructionClassificationDataset,
    batch_size: int,
    rank: int,
    world_size: int,
    *,
    class_weight_strategy: str | None = None,
) -> Tuple[DataLoader, DataLoader, Sampler | None]:
    train_sampler: Sampler | None = None
    val_sampler: DistributedSampler | None = None

    if world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    if class_weight_strategy:
        sample_weights = train_dataset.compute_sample_weights(strategy=class_weight_strategy)
        if world_size > 1:
            train_sampler = DistributedWeightedRandomSampler(
                sample_weights,
                num_replicas=world_size,
                rank=rank,
            )
        else:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
    elif world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def instantiate_model(
    args: argparse.Namespace,
    label2id: Dict[str, int],
    distributed: bool,
) -> LLMFeatureExtractorClassifier:
    """Create and configure the feature-extractor classifier model."""

    head_config = ClassifierHeadConfig(
        hidden_dims=args.hidden_dims or [],
        dropout=args.dropout,
        activation=args.activation,
    )

    lora_config = None
    if not args.no_adapter:
        from src.lift.models.gptj.lora_gptj import LoRaConfigParams

        lora_config = LoRaConfigParams(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules or None,
        )

    model = LLMFeatureExtractorClassifier(
        model_name=args.model_name,
        adapter=not args.no_adapter,
        model_path=args.adapter_path or args.model_name,
        load_in_4bit=args.load_in_4bit,
        model_provider=args.model_provider,
        num_labels=len(label2id),
        classifier_config=head_config,
        freeze_backbone=not (args.train_backbone or args.train_lora),
        lora_config=lora_config,
        wrap_backbone_with_ddp=not distributed,
    )

    if args.adapter_path and not args.no_adapter:
        model.load_networks(args.adapter_path)

    if args.train_backbone:
        model.unfreeze_backbone()
    elif args.train_lora:
        model.enable_lora_training()
    else:
        model.freeze_backbone()

    model.set_label_mapping(label2id)
    return model


def build_optimizer(
    model: LLMFeatureExtractorClassifier,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    """Create the AdamW optimizer with separate learning rates when requested."""

    param_groups: List[Dict[str, Any]] = []

    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": args.learning_rate})

    if args.train_backbone or args.train_lora:
        backbone_params = list(model.backbone_parameters())
        if backbone_params:
            backbone_lr = args.backbone_learning_rate or args.learning_rate
            param_groups.append({"params": backbone_params, "lr": backbone_lr})

    if not param_groups:
        raise ValueError(
            "No trainable parameters were found. Ensure the classifier head or backbone is unfrozen."
        )

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    distributed = torch.cuda.is_available() and world_size > 1

    if distributed:
        setup_distributed(rank, world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    train_examples = load_examples(args.train_file)
    val_examples = load_examples(args.val_file)

    label2id = build_label_mapping(train_examples, args.label_field)

    if args.train_backbone and args.train_lora:
        raise ValueError("--train-backbone and --train-lora are mutually exclusive options.")

    if args.train_lora and args.no_adapter:
        raise ValueError("--train-lora requires LoRA adapters. Remove --no-adapter to enable them.")

    model = instantiate_model(args, label2id, distributed)
    model.to(device)

    forward_module: torch.nn.Module = model
    if distributed:
        forward_module = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
        )

    train_dataset = InstructionClassificationDataset(
        train_examples,
        model.tokenizer,
        label2id=label2id,
        label_field=args.label_field,
        max_length=args.max_length,
    )
    val_dataset = InstructionClassificationDataset(
        val_examples,
        model.tokenizer,
        label2id=label2id,
        label_field=args.label_field,
        max_length=args.max_length,
    )

    class_weight_strategy = None if args.class_weight == "none" else args.class_weight

    train_loader, val_loader, train_sampler = build_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        rank,
        world_size,
        class_weight_strategy=class_weight_strategy,
    )

    model.device = device

    optimizer = build_optimizer(model, args)

    total_steps = len(train_loader) * max(args.epochs, 1)
    scheduler = None
    if total_steps > 0 and args.warmup_steps >= 0:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(args.warmup_steps, total_steps),
            num_training_steps=total_steps,
        )

    loss_weights = None
    if class_weight_strategy:
        loss_weights = train_dataset.compute_class_weights(strategy=class_weight_strategy)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=loss_weights.to(model.device) if loss_weights is not None else None
    )

    output_dir = Path(args.output_dir)
    is_main_process = rank == 0
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_metrics = {"loss": float("inf"), "accuracy": 0.0}
    best_classifier: Dict[str, Any] | None = None
    backbone_dir: Path | None = output_dir / "backbone" if (args.train_backbone or args.train_lora) else None

    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        forward_module.train()
        model.train()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process,
        )

        for batch in progress:
            input_ids = batch[0].to(model.device, non_blocking=True)
            attention_mask = batch[1].to(model.device, non_blocking=True)
            labels = batch[2].to(model.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = forward_module(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if is_main_process:
                progress.set_postfix(loss=loss.detach().item())

        metrics = evaluate(model, forward_module, val_loader, loss_fn, distributed)
        if is_main_process:
            print(
                f"Epoch {epoch + 1}: val_loss={metrics['loss']:.4f}, val_accuracy={metrics['accuracy']:.4f}"
            )

        if is_main_process and metrics["accuracy"] >= best_metrics["accuracy"]:
            best_metrics = metrics
            best_classifier = {
                "state": {k: v.cpu() for k, v in model.classifier.state_dict().items()},
                "label2id": dict(model.label2id),
            }
            if backbone_dir is not None:
                model.save_networks(str(backbone_dir))

    if is_main_process and best_classifier is None:
        best_classifier = {
            "state": {k: v.cpu() for k, v in model.classifier.state_dict().items()},
            "label2id": dict(model.label2id),
        }

    if is_main_process and best_classifier is not None:
        model.classifier.load_state_dict(best_classifier["state"])
        model.set_label_mapping(best_classifier["label2id"])

    if distributed:
        cleanup_distributed()

    if not is_main_process:
        return

    if backbone_dir is not None:
        model.save_networks(str(backbone_dir))

    extra_metadata = {"class_weight_strategy": class_weight_strategy} if class_weight_strategy else None
    model.save_classifier(output_dir, backbone_dir=backbone_dir, extra_metadata=extra_metadata)

    summary_path = output_dir / "training_metrics.json"
    summary_path.write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Training complete. Best validation metrics:", best_metrics)


def main() -> None:
    args = parse_args()

    cuda_available = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if cuda_available else 1

    if world_size <= 1:
        train(0, world_size, args)
        return

    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
