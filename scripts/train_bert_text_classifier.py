#!/usr/bin/env python
"""Fine-tune a BERT encoder for text classification tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm.auto import tqdm

from src.lift.models.bert.text_classifier import (
    BertTextClassificationDataset,
    BertTextClassifier,
    BertTextClassifierConfig,
    extract_label,
    load_text_classification_examples,
)
from src.lift.models.gptj.samplers import DistributedWeightedRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT text classifier.")
    parser.add_argument("--train-file", required=True, help="Path to the training dataset (JSON/JSONL/CSV).")
    parser.add_argument("--val-file", required=True, help="Path to the validation dataset (JSON/JSONL/CSV).")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and configuration.")

    parser.add_argument("--model-name", default="bert-base-chinese", help="Pretrained BERT checkpoint identifier.")
    parser.add_argument("--text-field", help="Optional text field to read from the dataset.")
    parser.add_argument("--label-field", help="Optional label field to read from the dataset.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability applied to the classifier head.")

    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Batch size used for evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Linear warmup ratio of total steps.")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Optional label smoothing factor.")
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "sqrt_inv"],
        default="none",
        help="Strategy to mitigate class imbalance via sampling and loss weighting.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-cuda", action="store_true", help="Force training on CPU even if CUDA is available.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, (total_steps - current_step) / max(1, total_steps - warmup_steps))

    return lr_lambda


def collate_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "10053")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def compute_class_weights(
    labels: torch.Tensor, num_classes: int, strategy: Optional[str]
) -> Optional[torch.Tensor]:
    if strategy is None or strategy == "none":
        return None

    if labels.numel() == 0:
        raise ValueError("Cannot compute class weights without any labels.")

    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    if torch.any(counts == 0):
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)

    total = counts.sum().item()
    if total <= 0:
        raise ValueError("Invalid class counts; ensure the dataset contains labelled samples.")

    if strategy == "balanced":
        weights = torch.tensor(total, dtype=torch.float32) / (counts * float(num_classes))
    elif strategy == "sqrt_inv":
        weights = torch.where(counts > 0, 1.0 / torch.sqrt(counts), torch.zeros_like(counts))
        if torch.sum(weights) > 0:
            weights = weights * (float(num_classes) / torch.sum(weights))
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    return weights


def compute_sample_weights(labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        raise ValueError("Cannot compute sample weights without any labels.")
    if class_weights.ndim != 1:
        raise ValueError("class_weights must be a 1D tensor of per-class weights")
    return class_weights[labels]


def build_dataloaders(
    train_dataset: BertTextClassificationDataset,
    val_dataset: BertTextClassificationDataset,
    *,
    batch_size: int,
    eval_batch_size: int,
    world_size: int,
    rank: int,
    num_labels: int,
    class_weight_strategy: Optional[str],
    seed: int,
) -> Tuple[DataLoader, DataLoader, Optional[Sampler], Optional[torch.Tensor]]:
    train_sampler: Optional[Sampler] = None
    val_sampler: Optional[DistributedSampler] = None
    class_weights: Optional[torch.Tensor] = None

    if train_dataset.labels is None:
        raise ValueError("Training dataset must include labels for supervised learning.")

    if class_weight_strategy and class_weight_strategy != "none":
        class_weights = compute_class_weights(
            train_dataset.labels,
            num_classes=num_labels,
            strategy=class_weight_strategy,
        )
        sample_weights = compute_sample_weights(train_dataset.labels, class_weights)
        if world_size > 1:
            train_sampler = DistributedWeightedRandomSampler(
                sample_weights,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
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

    if world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
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
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, class_weights


def gather_variable_length_tensor(
    tensor: torch.Tensor, world_size: int, device: torch.device
) -> torch.Tensor:
    if world_size == 1 or not is_distributed():
        return tensor.detach().cpu()

    tensor = tensor.to(device)
    local_size = torch.tensor([tensor.size(0)], device=device, dtype=torch.long)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = int(torch.max(torch.stack(size_list)).item())

    trailing_shape = tensor.shape[1:]
    padded_shape = (max_size,) + trailing_shape
    padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=device)
    if tensor.size(0) > 0:
        padded[: tensor.size(0)] = tensor

    gather_list = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gather_list, padded)

    results: List[torch.Tensor] = []
    for gathered, size in zip(gather_list, size_list):
        results.append(gathered[: int(size.item())])
    return torch.cat(results, dim=0).cpu()


def evaluate(
    model: BertTextClassifier,
    dataloader: DataLoader,
    device: torch.device,
    *,
    world_size: int,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    total_samples = 0
    prob_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = collate_to_device(batch, device)
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                label_smoothing=label_smoothing,
                reduction="sum",
            )

            loss_sum += loss.item()
            total_samples += labels.size(0)
            probabilities = F.softmax(logits, dim=-1)
            prob_chunks.append(probabilities.detach())
            label_chunks.append(labels.detach())

    device_loss = torch.tensor([loss_sum, float(total_samples)], device=device)
    if is_distributed():
        dist.all_reduce(device_loss, op=dist.ReduceOp.SUM)
    total_loss, total_count = device_loss.tolist()

    if prob_chunks:
        local_probs = torch.cat(prob_chunks)
        local_labels = torch.cat(label_chunks)
    else:
        num_labels = model.config.num_labels
        local_probs = torch.empty((0, num_labels), dtype=torch.float32, device=device)
        local_labels = torch.empty(0, dtype=torch.long, device=device)

    probs_all = gather_variable_length_tensor(local_probs, world_size, device)
    labels_all = gather_variable_length_tensor(local_labels, world_size, device)

    labels_list = labels_all.tolist()

    roc_auc = 0.0
    if labels_list and probs_all.numel() > 0:
        try:
            if probs_all.size(1) == 2:
                roc_auc = roc_auc_score(labels_list, probs_all[:, 1].numpy())
            else:
                roc_auc = roc_auc_score(
                    labels_list,
                    probs_all.numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        except ValueError:
            roc_auc = 0.0

    mean_loss = total_loss / max(total_count, 1.0)
    return {"loss": mean_loss, "roc_auc": roc_auc}


def main() -> None:
    args = parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    world_size = torch.cuda.device_count() if use_cuda else 1

    if world_size <= 1:
        train_worker(0, world_size, args, use_cuda)
        return

    mp.spawn(train_worker, args=(world_size, args, use_cuda), nprocs=world_size, join=True)


def train_worker(rank: int, world_size: int, args: argparse.Namespace, use_cuda: bool) -> None:
    distributed = use_cuda and world_size > 1

    if distributed:
        setup_distributed(rank, world_size)
    try:
        device = torch.device(f"cuda:{rank}") if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(device)

        set_seed(args.seed + rank)

        output_dir = Path(args.output_dir)
        is_main_process = rank == 0
        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)

        train_examples = load_text_classification_examples(Path(args.train_file), require_labels=True)
        val_examples = load_text_classification_examples(Path(args.val_file), require_labels=True)

        labels = sorted({extract_label(example, explicit_field=args.label_field) for example in train_examples})
        label2id = {label: idx for idx, label in enumerate(labels)}

        config = BertTextClassifierConfig(
            model_name=args.model_name,
            num_labels=len(labels),
            dropout=args.dropout,
            max_length=args.max_length,
            label2id=label2id,
        )
        tokenizer = BertTextClassifier.load_tokenizer(args.model_name)

        train_dataset = BertTextClassificationDataset(
            train_examples,
            tokenizer,
            max_length=args.max_length,
            label2id=label2id,
            text_field=args.text_field,
            label_field=args.label_field,
        )
        val_dataset = BertTextClassificationDataset(
            val_examples,
            tokenizer,
            max_length=args.max_length,
            label2id=label2id,
            text_field=args.text_field,
            label_field=args.label_field,
        )

        class_weight_strategy = args.class_weight if args.class_weight != "none" else None
        train_loader, val_loader, train_sampler, class_weights = build_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            world_size=world_size,
            rank=rank,
            num_labels=len(label2id),
            class_weight_strategy=class_weight_strategy,
            seed=args.seed,
        )

        model = BertTextClassifier(config=config, label_smoothing=0.0)
        model.to(device)

        train_module: torch.nn.Module = model
        if distributed:
            train_module = DistributedDataParallel(
                model,
                device_ids=[rank] if use_cuda else None,
                output_device=rank if use_cuda else None,
                find_unused_parameters=False,
            )

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.epochs // max(1, args.gradient_accumulation)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = LambdaLR(optimizer, compute_lr_lambda(total_steps, warmup_steps))

        class_weights_device = class_weights.to(device) if class_weights is not None else None

        best_metric = -float("inf")
        best_state: Dict[str, torch.Tensor] = {}

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch - 1)

            train_module.train()
            model.train()
            optimizer.zero_grad()
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                disable=not is_main_process,
            )
            accumulated_loss = 0.0

            for step, batch in enumerate(progress, start=1):
                batch = collate_to_device(batch, device)
                labels_tensor = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k != "labels"}

                outputs = train_module(**inputs)
                logits = outputs["logits"]
                loss = F.cross_entropy(
                    logits,
                    labels_tensor,
                    weight=class_weights_device,
                    label_smoothing=args.label_smoothing,
                )

                loss = loss / args.gradient_accumulation
                loss.backward()
                accumulated_loss += loss.item()

                if step % args.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if is_main_process:
                        progress.set_postfix({"train_loss": accumulated_loss})
                    accumulated_loss = 0.0

            metrics = evaluate(
                model,
                val_loader,
                device,
                world_size=world_size,
                class_weights=class_weights_device,
                label_smoothing=args.label_smoothing,
            )
            metric_score = metrics["roc_auc"]
            if is_main_process and metric_score > best_metric:
                best_metric = metric_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if is_main_process:
                metrics_path = output_dir / "metrics.json"
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, **metrics}, f, ensure_ascii=False, indent=2)

        if not best_state:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if is_main_process:
            model.load_state_dict(best_state)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            training_args = {
                "train_file": args.train_file,
                "val_file": args.val_file,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "gradient_accumulation": args.gradient_accumulation,
                "label_smoothing": args.label_smoothing,
                "seed": args.seed,
                "class_weight": args.class_weight,
            }
            with (output_dir / "training_args.json").open("w", encoding="utf-8") as f:
                json.dump(training_args, f, ensure_ascii=False, indent=2)
    finally:
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
