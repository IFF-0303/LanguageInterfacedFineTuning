#!/usr/bin/env python
"""Train a classifier that fuses LLM text features with structured data embeddings."""

from __future__ import annotations

import argparse
import json
import os
import random
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from tqdm import tqdm

from sklearn.metrics import accuracy_score, roc_auc_score

from src.lift.models.gptj.feature_extractor_classifier import ClassifierHeadConfig
from src.lift.models.gptj.llm_structured_fusion_classifier import (
    LLMStructuredFusionClassifier,
    StructuredEncoderConfig,
    StructuredFeatureEncoder,
    StructuredFusionDataset,
    StructuredFusionHead,
)
from src.lift.models.gptj.samplers import DistributedWeightedRandomSampler
from src.lift.models.gptj.lora_gptj import LoRaConfigParams
from scripts.health_csv_utils import load_health_dataset_from_csv


TAB_FEATURES_FIELD = "tab_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an instruction-following LLM alongside structured feature embeddings "
            "with flexible fusion strategies."
        )
    )
    parser.add_argument("--train-file", required=True, help="Path to the training dataset (JSON/JSONL/CSV).")
    parser.add_argument("--val-file", required=True, help="Path to the validation dataset (JSON/JSONL/CSV).")
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
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank when adapters are enabled.")
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="Scaling factor applied within LoRA layers when adapters are enabled.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability used by LoRA adapters during training.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=None,
        help=(
            "Optional explicit module names to wrap with LoRA layers. "
            "Defaults to attention and MLP projections if omitted."
        ),
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantisation for the backbone.")
    parser.add_argument("--train-backbone", action="store_true", help="Unfreeze the backbone for joint training.")
    parser.add_argument(
        "--train-lora",
        action="store_true",
        help="Only fine-tune LoRA adapter parameters while keeping base weights frozen.",
    )

    parser.add_argument(
        "--numeric-fields",
        nargs="*",
        default=None,
        help="Columns containing numeric features to be normalised and embedded.",
    )
    parser.add_argument(
        "--categorical-fields",
        nargs="*",
        default=None,
        help="Columns containing categorical features that should be embedded.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenisation.",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for structured head parameters.")
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        help="Optional learning rate for backbone/adapter parameters when --train-backbone/--train-lora is enabled.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warmup steps for the scheduler.")

    parser.add_argument(
        "--structured-dim",
        type=int,
        default=256,
        help="Hidden dimension of the structured feature encoder output.",
    )
    parser.add_argument(
        "--categorical-embedding-dim",
        type=int,
        default=32,
        help="Embedding size for each categorical feature column.",
    )
    parser.add_argument(
        "--numeric-hidden-dim",
        type=int,
        help="Hidden dimension of the numeric encoder MLP. Defaults to max(2*num_numeric, structured_dim).",
    )
    parser.add_argument(
        "--structured-dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside the structured feature encoder.",
    )
    parser.add_argument(
        "--disable-numeric-missing-embeddings",
        action="store_true",
        help="Disable learnable embeddings for missing numeric values (still exposes masks).",
    )

    parser.add_argument(
        "--fusion-mode",
        choices=["concat", "gated_add", "film"],
        default="concat",
        help="Strategy for combining LLM and structured representations.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=None,
        help="Hidden dimensions for the final classifier head (e.g. --hidden-dims 512 128).",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability applied in the classifier head.")
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
    parser.add_argument(
        "--disable-ddp-find-unused-parameters",
        action="store_false",
        dest="ddp_find_unused_parameters",
        help=(
            "Disable DistributedDataParallel unused parameter detection. "
            "Enable this option if all parameters are guaranteed to receive gradients."
        ),
    )
    parser.set_defaults(ddp_find_unused_parameters=True)
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()


def load_examples(
    file_path: str,
    *,
    tab_features_field: str = TAB_FEATURES_FIELD,
    numeric_fields: Optional[Sequence[str]] = None,
    categorical_fields: Optional[Sequence[str]] = None,
    require_label: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str], List[str], List[str]]:
    data_path = Path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file '{file_path}' does not exist.")

    if data_path.suffix.lower() == ".csv":
        (
            examples,
            inferred_label_field,
            feature_columns,
            csv_categorical_fields,
        ) = load_health_dataset_from_csv(
            data_path,
            tab_features_field=tab_features_field,
            feature_columns=None,
            require_label=require_label,
        )

        categorical_list = (
            list(categorical_fields)
            if categorical_fields is not None
            else list(csv_categorical_fields)
        )
        categorical_set = set(categorical_list)
        numeric_list = (
            list(numeric_fields)
            if numeric_fields is not None
            else [col for col in feature_columns if col not in categorical_set]
        )

        for example in examples:
            tab_values = example.get(tab_features_field)
            if not isinstance(tab_values, (list, tuple)):
                continue
            for idx, column in enumerate(feature_columns):
                if idx >= len(tab_values):
                    break
                value = tab_values[idx]
                if isinstance(value, float) and math.isnan(value):
                    example[column] = None
                else:
                    example[column] = value

        return examples, inferred_label_field, numeric_list, categorical_list

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

    numeric_list = list(numeric_fields) if numeric_fields is not None else []
    categorical_list = list(categorical_fields) if categorical_fields is not None else []
    return examples, None, numeric_list, categorical_list


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
                "Unable to infer label field. Provide --label-field explicitly or include one of "
                "'label', 'target', 'answer', 'category', or 'class' in the dataset."
            )
    unique_labels = sorted(set(labels))
    if not unique_labels:
        raise ValueError("No labels found in the training dataset.")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "10086")
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if not is_distributed():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def build_datasets(
    *,
    model: LLMStructuredFusionClassifier,
    train_examples: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    label2id: Dict[str, int],
    numeric_fields: Optional[Sequence[str]],
    categorical_fields: Optional[Sequence[str]],
    label_field: Optional[str],
    max_length: int,
) -> Tuple[StructuredFusionDataset, StructuredFusionDataset]:
    train_dataset = StructuredFusionDataset(
        train_examples,
        model.tokenizer,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        label2id=label2id,
        label_field=label_field,
        max_length=max_length,
    )
    val_dataset = StructuredFusionDataset(
        val_examples,
        model.tokenizer,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        label2id=label2id,
        label_field=label_field,
        max_length=max_length,
        numeric_stats=train_dataset.numeric_stats,
        categorical_vocabs=train_dataset.categorical_vocabs,
        categorical_missing_indices=train_dataset.categorical_missing_indices,
    )
    return train_dataset, val_dataset


def configure_optimizer(
    model: LLMStructuredFusionClassifier,
    *,
    learning_rate: float,
    backbone_learning_rate: Optional[float],
    weight_decay: float,
    train_backbone: bool,
    train_lora: bool,
) -> AdamW:
    head_params = list(model.structured_parameters())
    for param in head_params:
        param.requires_grad_(True)

    param_groups = [
        {"params": head_params, "lr": learning_rate, "weight_decay": weight_decay},
    ]

    if train_backbone:
        model.unfreeze_backbone()
        backbone_params = list(model.backbone_parameters())
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_learning_rate or learning_rate,
                "weight_decay": weight_decay,
            }
        )
    elif train_lora:
        model.enable_lora_training()
        backbone_params = list(model.backbone_parameters())
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_learning_rate or learning_rate,
                "weight_decay": weight_decay,
            }
        )
    else:
        model.freeze_backbone()

    return AdamW(param_groups)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lambda _: 1.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def prepare_dataloaders(
    train_dataset: StructuredFusionDataset,
    val_dataset: StructuredFusionDataset,
    *,
    batch_size: int,
    num_workers: int,
    class_weight: str,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, Sampler | None]:
    class_weight_strategy = None if class_weight == "none" else class_weight

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
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler


def training_step(
    model: LLMStructuredFusionClassifier,
    forward_module: nn.Module,
    batch: Tuple[torch.Tensor, ...],
    loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch[0].to(model.device, non_blocking=True)
    attention_mask = batch[1].to(model.device, non_blocking=True)
    numeric_features = batch[2].to(model.device, non_blocking=True)
    numeric_mask = batch[3].to(model.device, non_blocking=True)
    categorical_features = batch[4].to(model.device, non_blocking=True)
    categorical_mask = batch[5].to(model.device, non_blocking=True)
    labels = batch[6].to(model.device, non_blocking=True)

    logits = forward_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        numeric_features=numeric_features,
        numeric_mask=numeric_mask,
        categorical_features=categorical_features,
        categorical_mask=categorical_mask,
    )
    loss = loss_fn(logits, labels)
    return loss, logits


def evaluate(
    model: LLMStructuredFusionClassifier,
    forward_module: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> Dict[str, float]:
    model.eval()
    forward_module.eval()

    loss_total = torch.zeros(1, dtype=torch.float64, device=model.device)
    correct_total = torch.zeros(1, dtype=torch.float64, device=model.device)
    count_total = torch.zeros(1, dtype=torch.float64, device=model.device)

    local_targets: List[int] = []
    local_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(model.device, non_blocking=True)
            attention_mask = batch[1].to(model.device, non_blocking=True)
            numeric_features = batch[2].to(model.device, non_blocking=True)
            numeric_mask = batch[3].to(model.device, non_blocking=True)
            categorical_features = batch[4].to(model.device, non_blocking=True)
            categorical_mask = batch[5].to(model.device, non_blocking=True)
            labels = batch[6].to(model.device, non_blocking=True)

            logits = forward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features,
                numeric_mask=numeric_mask,
                categorical_features=categorical_features,
                categorical_mask=categorical_mask,
            )
            loss = loss_fn(logits, labels)

            probs_batch = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs_batch, dim=-1)

            loss_total += loss.detach().to(torch.float64) * labels.size(0)
            correct_total += (preds == labels).sum().to(torch.float64)
            count_total += labels.size(0)

            local_targets.extend(labels.tolist())
            local_probs.extend(probs_batch.tolist())

    stats = torch.cat([loss_total, correct_total, count_total])
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_loss, total_correct, total_count = stats.tolist()
    metrics: Dict[str, float] = {
        "loss": total_loss / max(total_count, 1.0),
        "accuracy": total_correct / max(total_count, 1.0),
    }

    roc_auc_value = float("nan")
    if distributed:
        gathered: List[Dict[str, List[float]] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(
            gathered,
            {
                "targets": local_targets,
                "probs": local_probs,
            },
        )
        if rank == 0:
            all_targets: List[int] = []
            all_probs: List[List[float]] = []
            for item in gathered:
                if not item:
                    continue
                all_targets.extend(int(t) for t in item["targets"])
                all_probs.extend([[float(v) for v in row] for row in item["probs"]])
            try:
                if all_probs and len(all_probs[0]) == 2:
                    roc_auc_value = roc_auc_score(all_targets, [p[1] for p in all_probs])
                elif all_probs:
                    roc_auc_value = roc_auc_score(all_targets, all_probs, multi_class="ovo")
            except ValueError:
                roc_auc_value = float("nan")
        tensor_value = torch.tensor(
            [roc_auc_value], device=model.device, dtype=torch.float32
        )
        dist.broadcast(tensor_value, src=0)
        roc_auc_value = float(tensor_value.item())
    elif local_probs:
        try:
            if len(local_probs[0]) == 2:
                roc_auc_value = roc_auc_score(local_targets, [p[1] for p in local_probs])
            else:
                roc_auc_value = roc_auc_score(local_targets, local_probs, multi_class="ovo")
        except ValueError:
            roc_auc_value = float("nan")

    metrics["roc_auc"] = roc_auc_value
    return metrics


def train(rank: int, world_size: int, args: argparse.Namespace) -> None:
    distributed = torch.cuda.is_available() and world_size > 1

    if distributed:
        setup_distributed(rank, world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed + rank)

    (
        train_examples,
        inferred_label_field,
        numeric_fields,
        categorical_fields,
    ) = load_examples(
        args.train_file,
        tab_features_field=TAB_FEATURES_FIELD,
        numeric_fields=args.numeric_fields,
        categorical_fields=args.categorical_fields,
        require_label=True,
    )
    val_examples, _, _, _ = load_examples(
        args.val_file,
        tab_features_field=TAB_FEATURES_FIELD,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        require_label=True,
    )

    if args.train_backbone and args.train_lora:
        raise ValueError("--train-backbone and --train-lora cannot be enabled together.")
    if args.train_lora and args.no_adapter:
        raise ValueError("--train-lora requires LoRA adapters. Remove --no-adapter to enable them.")

    label_field = args.label_field or inferred_label_field
    label2id = build_label_mapping(train_examples, label_field=label_field)
    num_labels = len(label2id)

    classifier_config = ClassifierHeadConfig(
        hidden_dims=list(args.hidden_dims) if args.hidden_dims else [],
        dropout=args.dropout,
        activation=args.activation,
    )

    structured_config = StructuredEncoderConfig(
        numeric_feature_dim=len(numeric_fields),
        categorical_cardinalities=[],  # Placeholder, updated after dataset construction
        structured_dim=args.structured_dim,
        categorical_embedding_dim=args.categorical_embedding_dim,
        numeric_hidden_dim=args.numeric_hidden_dim,
        dropout=args.structured_dropout,
        use_numeric_missing_embeddings=not args.disable_numeric_missing_embeddings,
    )

    lora_config = None
    if not args.no_adapter:
        lora_config = LoRaConfigParams(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules or None,
        )

    model = LLMStructuredFusionClassifier(
        num_labels=num_labels,
        classifier_config=classifier_config,
        structured_config=structured_config,
        fusion_mode=args.fusion_mode,
        freeze_backbone=not args.train_backbone and not args.train_lora,
        wrap_backbone_with_ddp=not distributed,
        model_name=args.model_name,
        model_provider=args.model_provider,
        adapter_path=args.adapter_path,
        adapter=not args.no_adapter,
        load_in_4bit=args.load_in_4bit,
        lora_config=lora_config,
    )

    model.set_label_mapping(label2id)

    train_dataset, val_dataset = build_datasets(
        model=model,
        train_examples=train_examples,
        val_examples=val_examples,
        label2id=label2id,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        label_field=label_field,
        max_length=args.max_length,
    )

    structured_config.numeric_feature_dim = train_dataset.numeric_feature_dim
    structured_config.categorical_cardinalities = train_dataset.categorical_cardinalities
    model.structured_config = structured_config
    model.structured_encoder = StructuredFeatureEncoder(structured_config).to(model.device)
    hidden_size = int(getattr(model._unwrap_model().config, "hidden_size"))
    model.fusion_head = StructuredFusionHead(
        llm_hidden_dim=hidden_size,
        structured_dim=structured_config.structured_dim,
        num_labels=num_labels,
        config=classifier_config,
        fusion_mode=args.fusion_mode,
    ).to(model.device)
    model.classifier = model.fusion_head
    model.fusion_mode = args.fusion_mode

    model.set_structured_metadata(train_dataset.structured_metadata())

    model.device = device

    forward_module: nn.Module = model
    if distributed:
        forward_module = DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
            output_device=rank if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    train_loader, val_loader, train_sampler = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_weight=args.class_weight,
        rank=rank,
        world_size=world_size,
    )

    optimizer = configure_optimizer(
        model,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        train_backbone=args.train_backbone,
        train_lora=args.train_lora,
    )

    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)

    loss_weights = None
    if args.class_weight != "none":
        loss_weights = train_dataset.compute_class_weights(strategy=args.class_weight)
        loss_weights = loss_weights.to(model.device)

    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

    is_main_process = (not distributed) or rank == 0
    global_step = 0

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
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_step(model, forward_module, batch, loss_fn)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if is_main_process:
                progress.set_postfix({"loss": loss.item()})

        val_metrics = evaluate(
            model,
            forward_module,
            val_loader,
            loss_fn,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

        if is_main_process:
            print(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "metrics": val_metrics,
                    },
                    ensure_ascii=False,
                )
            )

    if distributed:
        cleanup_distributed()

    if not is_main_process:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.set_label_mapping(label2id)
    extra_metadata = {
        "training_args": {
            "train_file": args.train_file,
            "val_file": args.val_file,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "fusion_mode": args.fusion_mode,
            "numeric_fields": list(numeric_fields),
            "categorical_fields": list(categorical_fields),
            "label_field": label_field,
        }
    }
    model.save_classifier(output_dir, extra_metadata=extra_metadata)


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
