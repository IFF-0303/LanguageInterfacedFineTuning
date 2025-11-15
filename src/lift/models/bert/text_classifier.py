"""Simple BERT-based text classification utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

__all__ = [
    "BertTextClassifier",
    "BertTextClassifierConfig",
    "BertTextClassificationDataset",
    "load_text_classification_examples",
    "extract_label",
]


@dataclass
class BertTextClassifierConfig:
    """Configuration wrapper stored alongside the fine-tuned model."""

    model_name: str = "bert-base-chinese"
    num_labels: int = 2
    dropout: float = 0.1
    max_length: int = 512
    label2id: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_labels": int(self.num_labels),
            "dropout": float(self.dropout),
            "max_length": int(self.max_length),
            "label2id": dict(self.label2id),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BertTextClassifierConfig":
        return cls(
            model_name=str(data.get("model_name", "bert-base-chinese")),
            num_labels=int(data.get("num_labels", 2)),
            dropout=float(data.get("dropout", 0.1)),
            max_length=int(data.get("max_length", 512)),
            label2id=dict(data.get("label2id", {})),
        )


class BertTextClassificationDataset(Dataset):
    """Dataset that tokenises text/label pairs for classification."""

    def __init__(
        self,
        examples: Iterable[Mapping[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int = 512,
        label2id: Optional[Mapping[str, int]] = None,
        text_field: Optional[str] = None,
        label_field: Optional[str] = None,
    ) -> None:
        self.examples: List[Mapping[str, Any]] = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts: List[str] = [
            extract_text(example, text_field=text_field) for example in self.examples
        ]

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

        self.label2id = dict(label2id or {}) if label2id else None
        if self.label2id:
            labels: List[int] = []
            for example in self.examples:
                raw = extract_label(example, explicit_field=label_field)
                if raw not in self.label2id:
                    raise ValueError(
                        f"Label {raw!r} not present in label2id mapping: {sorted(self.label2id)}"
                    )
                labels.append(self.label2id[raw])
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class BertTextClassifier(nn.Module):
    """Thin wrapper around :class:`AutoModelForSequenceClassification`."""

    def __init__(
        self,
        *,
        config: BertTextClassifierConfig,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_smoothing = float(label_smoothing)

        id2label = {idx: label for label, idx in config.label2id.items()}
        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout,
            id2label=id2label,
            label2id=config.label2id,
        )
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=hf_config,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        if labels is not None and self.label_smoothing > 0 and outputs.loss is not None:
            loss = self._apply_label_smoothing(outputs.logits, labels)
            outputs.loss = loss

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def _apply_label_smoothing(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross entropy."""

        smoothing = self.label_smoothing
        if smoothing <= 0:
            raise ValueError("Label smoothing requested despite smoothing <= 0.")
        num_labels = logits.size(-1)
        log_probs = logits.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(smoothing / (num_labels - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        config_path = output_dir / "bert_text_classifier_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: Path) -> "BertTextClassifier":
        model_dir = Path(model_dir)
        config_path = model_dir / "bert_text_classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Expected configuration file not found at {config_path}."
            )
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = BertTextClassifierConfig.from_dict(config_data)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.config = config
        instance.label_smoothing = 0.0
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return instance

    @staticmethod
    def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def extract_text(example: Mapping[str, Any], text_field: Optional[str] = None) -> str:
    if text_field:
        if text_field not in example:
            raise KeyError(
                f"Text field '{text_field}' missing from example with keys: {list(example.keys())}"
            )
        value = example[text_field]
        if value is None:
            raise ValueError("Text field contains null value.")
        return str(value)

    for candidate in ("text", "instruction", "prompt", "content", "question"):
        if candidate in example and example[candidate] is not None:
            prefix = str(example[candidate])
            suffix = str(example.get("input", ""))
            if suffix:
                return f"{prefix}\n{suffix}".strip()
            return prefix.strip()

    raise KeyError(
        "Unable to infer text content from example. Provide --text-field explicitly when "
        "loading the dataset."
    )


def extract_label(example: Mapping[str, Any], explicit_field: Optional[str] = None) -> str:
    if explicit_field:
        if explicit_field not in example:
            raise KeyError(
                f"Label field '{explicit_field}' missing from example with keys: {list(example.keys())}"
            )
        return str(example[explicit_field])

    for candidate in ("label", "target", "answer", "output", "category", "class"):
        if candidate in example and example[candidate] is not None:
            return str(example[candidate])

    raise KeyError(
        "Unable to infer label from example. Provide --label-field explicitly when loading the dataset."
    )


def load_text_classification_examples(
    path: Path,
    *,
    require_labels: bool = True,
) -> List[Dict[str, Any]]:
    """Load text classification records from JSON/JSONL/CSV files."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    records: List[Dict[str, Any]] = []

    if suffix in {".json", ".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            if suffix == ".json":
                content = json.load(f)
                if isinstance(content, Mapping):
                    raise ValueError("JSON dataset must be a list of objects.")
                records = [dict(item) for item in content]
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
    elif suffix == ".csv":
        import csv

        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
    else:
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Supported formats: JSON, JSONL, CSV."
        )

    if require_labels:
        for record in records:
            extract_label(record)
    return records
