#!/usr/bin/env python
"""Train a fusion classifier that combines LLM and TabPFN feature extractors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm import tqdm

from src.lift.models.gptj.feature_extractor_classifier import ClassifierHeadConfig
from src.lift.models.gptj.llm_tabpfn_fusion_classifier import (
    FusionInstructionClassificationDataset,
    LLMTabPFNFusionClassifier,
    TabPFNFeatureExtractor,
    load_tabpfn_feature_tensor,
    save_tabpfn_feature_tensor,
)
from src.lift.models.gptj.samplers import DistributedWeightedRandomSampler
from scripts.health_csv_utils import load_health_dataset_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a fusion classifier that merges LLM text features and TabPFN tabular "
            "features using a lightweight MLP head."
        )
    )
    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to the training dataset (supports JSON/JSONL or CSV).",
    )
    parser.add_argument(
        "--val-file",
        required=True,
        help="Path to the validation dataset (supports JSON/JSONL or CSV).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints and metadata.")

    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct", help="Backbone LLM identifier.")
    parser.add_argument(
        "--model-provider",
        choices=["huggingface", "modelscope"],
        default="huggingface",
        help="Provider for loading the base model and tokenizer.",
    )
    parser.add_argument("--adapter-path", help="Optional directory containing pre-trained LoRA adapters.")
    parser.add_argument("--no-adapter", action="store_true", help="Disable LoRA adapters entirely.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantisation for the backbone.")
    parser.add_argument("--train-backbone", action="store_true", help="Unfreeze the backbone for joint training.")
    parser.add_argument(
        "--train-lora",
        action="store_true",
        help="Only fine-tune LoRA adapter parameters while keeping base weights frozen.",
    )

    parser.add_argument("--tab-features-field", default="tab_features", help="Field containing raw tabular features.")
    parser.add_argument(
        "--fusion-mode",
        choices=["concat", "gated_add"],
        default="concat",
        help="Strategy for combining LLM and TabPFN representations.",
    )
    parser.add_argument("--tabpfn-device", default="cpu", help="Device to run the TabPFN extractor on.")
    parser.add_argument(
        "--train-tabpfn-cache",
        help="Optional path to cached TabPFN features for the training split.",
    )
    parser.add_argument(
        "--val-tabpfn-cache",
        help="Optional path to cached TabPFN features for the validation split.",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="Optional learning rate for backbone/adapter parameters when train-backbone/train-lora is enabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for tokenisation.")

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when adapters are enabled.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling factor (alpha).")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout probability for LoRA adapters.")
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        help="Optional explicit module names to target with LoRA.",
    )

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Hidden dimensions for the fusion classifier head (e.g. --hidden-dims 512 128).",
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
            "Optional explicit field name containing labels. If not provided the script searches for "
            "'label', 'target', 'answer', 'category', or 'class'."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "sqrt_inv"],
        default="none",
        help="Strategy for rebalancing class distributions during training.",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()


def load_examples(
    file_path: str,
    *,
    tab_features_field: str,
    feature_columns: Optional[List[str]] = None,
    require_label: bool = False,
) -> Tuple[
    List[Dict[str, Any]],
    Optional[str],
    Optional[List[str]],
    Optional[List[str]],
]:
    data_path = Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' does not exist.")

    if data_path.suffix.lower() == ".csv":
        (
            examples,
            label_field,
            columns,
            categorical_columns,
        ) = load_health_dataset_from_csv(
            data_path,
            tab_features_field=tab_features_field,
            feature_columns=feature_columns,
            require_label=require_label,
        )
        return examples, label_field, list(columns), list(categorical_columns)

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
    return examples, None, feature_columns, None


def extract_tabular_matrix(examples: Iterable[Dict[str, Any]], field: str) -> np.ndarray:
    rows: List[List[float]] = []
    for example in examples:
        if field not in example:
            raise KeyError(
                f"Example missing tabular feature field '{field}'. Available keys: {list(example.keys())}"
            )
        values = example[field]
        if not isinstance(values, Iterable):
            raise TypeError(
                f"Tabular feature field '{field}' must be iterable. Received: {type(values)!r}."
            )
        row_values: List[float] = []
        for v in values:
            try:
                row_values.append(float(v))
            except (TypeError, ValueError):
                row_values.append(float("nan"))
        rows.append(row_values)
    if not rows:
        raise ValueError("No tabular features found to feed into TabPFN.")
    matrix = np.asarray(rows, dtype=np.float32)
    return matrix


def maybe_load_cached_features(path: Optional[str], expected_rows: int, split_name: str) -> Optional[torch.Tensor]:
    if not path:
        return None
    cache_path = Path(path)
    if not cache_path.exists():
        return None

    features = load_tabpfn_feature_tensor(cache_path)
    if features.size(0) != expected_rows:
        raise ValueError(
            f"Cached TabPFN features at '{cache_path}' contain {features.size(0)} rows, "
            f"but {expected_rows} were expected for the {split_name} split."
        )
    return features


def maybe_save_cached_features(path: Optional[str], features: torch.Tensor) -> None:
    if not path:
        return
    save_tabpfn_feature_tensor(path, features)


def build_label_mapping(
    examples: Iterable[Dict[str, Any]], label_field: str | None = None
) -> Dict[str, int]:
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
                "Unable to infer label field. Provide --label-field explicitly or include one of "
                "'label', 'target', 'answer', 'category', or 'class' in the dataset."
            )

    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def resolve_label(example: Dict[str, Any], label_field: str | None = None) -> str:
    if label_field:
        if label_field not in example:
            raise KeyError(
                f"Specified label field '{label_field}' missing from example: {example.keys()}"
            )
        return str(example[label_field])

    for key in ("label", "target", "answer", "category", "class"):
        if key in example:
            return str(example[key])
    raise KeyError(
        "Unable to infer label field. Provide --label-field explicitly or include one of "
        "'label', 'target', 'answer', 'category', or 'class' in the dataset."
    )


def configure_sampler(
    dataset: FusionInstructionClassificationDataset,
    class_weight_strategy: str,
    batch_size: int,
    *,
    distributed: bool,
) -> Tuple[Sampler[int] | None, torch.Tensor | None]:
    sample_weights = None
    if class_weight_strategy != "none":
        sample_weights = dataset.compute_sample_weights(strategy=class_weight_strategy)

    if distributed:
        if sample_weights is not None:
            sampler = DistributedWeightedRandomSampler(
                sample_weights, num_samples=len(dataset), replacement=True
            )
        else:
            sampler = DistributedSampler(dataset)
    else:
        if sample_weights is not None:
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        else:
            sampler = None

    if class_weight_strategy == "none":
        class_weights = None
    else:
        class_weights = dataset.compute_class_weights(strategy=class_weight_strategy)
    return sampler, class_weights


def setup_model(args: argparse.Namespace, num_labels: int, tab_feature_dim: int) -> LLMTabPFNFusionClassifier:
    head_cfg = ClassifierHeadConfig(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
    )

    model = LLMTabPFNFusionClassifier(
        model_name=args.model_name,
        adapter=not args.no_adapter,
        model_path=args.adapter_path or args.model_name,
        load_in_4bit=args.load_in_4bit,
        model_provider=args.model_provider,
        num_labels=num_labels,
        classifier_config=head_cfg,
        tab_feature_dim=tab_feature_dim,
        fusion_mode=args.fusion_mode,
    )

    if args.adapter_path and not args.no_adapter:
        model.load_networks(args.adapter_path)

    if args.train_backbone:
        model.unfreeze_backbone()
    elif args.train_lora:
        model.enable_lora_training()
    else:
        model.freeze_backbone()

    return model


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    (
        train_examples,
        inferred_label_field,
        feature_columns,
        categorical_columns,
    ) = load_examples(
        args.train_file,
        tab_features_field=args.tab_features_field,
        require_label=True,
    )
    val_examples, val_label_field, _, _ = load_examples(
        args.val_file,
        tab_features_field=args.tab_features_field,
        feature_columns=feature_columns,
        require_label=True,
    )

    effective_label_field = args.label_field or inferred_label_field or val_label_field
    if effective_label_field is None:
        raise ValueError(
            "Unable to determine the label field from the datasets. Provide --label-field explicitly."
        )

    label2id = build_label_mapping(train_examples, effective_label_field)
    num_labels = len(label2id)
    if num_labels < 2:
        raise ValueError("At least two distinct labels are required for classification.")

    train_tab_matrix = extract_tabular_matrix(train_examples, args.tab_features_field)
    val_tab_matrix = extract_tabular_matrix(val_examples, args.tab_features_field)

    train_tab_features = maybe_load_cached_features(
        args.train_tabpfn_cache, len(train_examples), "training"
    )
    if train_tab_features is not None:
        train_tab_features = train_tab_features.to(dtype=torch.float32, device="cpu")

    val_tab_features = maybe_load_cached_features(
        args.val_tabpfn_cache, len(val_examples), "validation"
    )
    if val_tab_features is not None:
        val_tab_features = val_tab_features.to(dtype=torch.float32, device="cpu")

    label_indices = [
        label2id[resolve_label(example, effective_label_field)] for example in train_examples
    ]

    categorical_indices: Optional[List[int]] = None
    if feature_columns is not None and categorical_columns:
        categorical_indices = [
            feature_columns.index(column)
            for column in categorical_columns
            if column in feature_columns
        ]

    tab_extractor: Optional[TabPFNFeatureExtractor] = None
    if train_tab_features is None or val_tab_features is None:
        tab_extractor = TabPFNFeatureExtractor(
            device=args.tabpfn_device,
            categorical_features=categorical_indices,
        )
        tab_extractor.fit(train_tab_matrix, label_indices)

        if train_tab_features is None:
            train_tab_features = tab_extractor.transform(train_tab_matrix)
            maybe_save_cached_features(args.train_tabpfn_cache, train_tab_features)

        if val_tab_features is None:
            val_tab_features = tab_extractor.transform(val_tab_matrix)
            maybe_save_cached_features(args.val_tabpfn_cache, val_tab_features)

    if train_tab_features is None or val_tab_features is None:
        raise RuntimeError("Failed to obtain TabPFN features for both training and validation splits.")

    train_tab_features = train_tab_features.to(dtype=torch.float32, device="cpu")
    val_tab_features = val_tab_features.to(dtype=torch.float32, device="cpu")
    if train_tab_features.size(1) != val_tab_features.size(1):
        raise ValueError(
            "Cached TabPFN features for training and validation have mismatched dimensionality: "
            f"{train_tab_features.size(1)} vs {val_tab_features.size(1)}"
        )

    model = setup_model(args, num_labels=num_labels, tab_feature_dim=train_tab_features.size(1))
    model.to(device)
    model.device = device
    model.set_label_mapping(label2id)

    tokenizer = model.tokenizer

    train_dataset = FusionInstructionClassificationDataset(
        train_examples,
        tokenizer,
        train_tab_features,
        label2id=label2id,
        label_field=effective_label_field,
        max_length=args.max_length,
    )
    val_dataset = FusionInstructionClassificationDataset(
        val_examples,
        tokenizer,
        val_tab_features,
        label2id=label2id,
        label_field=effective_label_field,
        max_length=args.max_length,
    )

    sampler, class_weights = configure_sampler(
        train_dataset,
        args.class_weight,
        args.batch_size,
        distributed=world_size > 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    optimizer_grouped_parameters: List[Dict[str, Any]] = [
        {"params": model.classifier.parameters(), "lr": args.learning_rate}
    ]

    backbone_lr = args.backbone_learning_rate or args.learning_rate
    if args.train_backbone or args.train_lora:
        optimizer_grouped_parameters.append(
            {"params": list(model.backbone_parameters()), "lr": backbone_lr}
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.warmup_steps:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.0,
            total_iters=max(args.warmup_steps, 1),
        )

    model.train()
    for epoch in range(args.epochs):
        if world_size > 1 and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=rank != 0)
        for batch in progress:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            tab_features = batch[2].to(device)
            labels = batch[3].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, tab_features=tab_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": epoch_loss / (progress.n or 1)})

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                tab_features = batch[2].to(device)
                labels = batch[3].to(device)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tab_features=tab_features,
                )
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / max(total, 1)
        if rank == 0:
            print(f"Validation accuracy after epoch {epoch + 1}: {accuracy:.4f}")
        model.train()

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "fusion_mode": args.fusion_mode,
            "tab_features_field": args.tab_features_field,
            "label_field": effective_label_field,
        }
        tabpfn_state_path: Optional[Path] = None
        if tab_extractor is not None:
            tabpfn_state_path = output_dir / "tabpfn.pkl"
            tab_extractor.save(tabpfn_state_path)
            metadata["tabpfn_state_path"] = tabpfn_state_path.name
        else:
            metadata["tabpfn_state_path"] = None
        if feature_columns is not None:
            metadata["tab_feature_columns"] = feature_columns
        if categorical_columns:
            metadata["categorical_feature_columns"] = list(categorical_columns)
        if categorical_indices:
            metadata["categorical_feature_indices"] = list(int(i) for i in categorical_indices)
        backbone_dir: Path | None = None
        if args.train_backbone or args.train_lora:
            backbone_dir = output_dir / "backbone"
            backbone_dir.mkdir(parents=True, exist_ok=True)
            model.save_networks(str(backbone_dir))
        elif args.adapter_path and not args.no_adapter:
            backbone_dir = Path(args.adapter_path)

        model.save_classifier(
            output_dir,
            backbone_dir=str(backbone_dir) if backbone_dir is not None else None,
            extra_metadata=metadata,
        )

    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(rank=0, world_size=world_size, args=args)


if __name__ == "__main__":
    main()
