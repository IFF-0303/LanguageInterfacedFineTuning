"""Utilities for using causal LLMs as frozen feature extractors with an MLP head."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset

from .lora_gptj import LoRaQGPTJ, build_instruction_prompt


@dataclass
class ClassifierHeadConfig:
    """Configuration for the small MLP classification head."""

    hidden_dims: Optional[List[int]] = field(default_factory=list)
    dropout: float = 0.1
    activation: str = "gelu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dims": list(self.hidden_dims) if self.hidden_dims else [],
            "dropout": self.dropout,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassifierHeadConfig":
        return cls(
            hidden_dims=data.get("hidden_dims") or [],
            dropout=data.get("dropout", 0.1),
            activation=data.get("activation", "gelu"),
        )


class InstructionClassificationDataset(Dataset):
    """Dataset for instruction-style prompts paired with categorical labels."""

    def __init__(
        self,
        examples: Iterable[Dict[str, Any]],
        tokenizer,
        *,
        label2id: Optional[Dict[str, int]] = None,
        label_field: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.examples: List[Dict[str, Any]] = list(examples)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.label_field = label_field

        prompts: List[str] = []
        labels: List[int] = []
        for example in self.examples:
            prompt = build_instruction_prompt(example)
            prompts.append(prompt.strip())

            if self.label2id is not None:
                label = self._extract_label(example)
                if label not in self.label2id:
                    raise ValueError(f"Label {label!r} not present in label2id mapping.")
                labels.append(self.label2id[label])

        encodings = tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels else None

    def _extract_label(self, example: Dict[str, Any]) -> str:
        if self.label_field:
            if self.label_field not in example:
                raise KeyError(
                    f"Specified label field '{self.label_field}' missing from example: {example.keys()}"
                )
            return str(example[self.label_field])

        for key in ("label", "target", "answer", "category", "class"):
            if key in example:
                return str(example[key])
        raise KeyError(
            "Unable to infer label field. Provide --label-field explicitly or include one of "
            "'label', 'target', 'answer', 'category', or 'class' in the dataset."
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx]


class LLMFeatureExtractorClassifier(LoRaQGPTJ):
    """Wraps :class:`LoRaQGPTJ` with a configurable backbone and MLP head."""

    classifier_config: ClassifierHeadConfig

    def __init__(
        self,
        *,
        num_labels: int,
        classifier_config: Optional[ClassifierHeadConfig] = None,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.classifier_config = classifier_config or ClassifierHeadConfig()
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.backbone_frozen: bool = False
        self.backbone_train_mode: str = "frozen"

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        self.classifier = self._build_classifier(num_labels)
        self.classifier.to(self.device)

    def freeze_backbone(self) -> None:
        """Ensure the underlying language model is used as a frozen feature extractor."""

        backbone = self._unwrap_model()
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad_(False)
        self.backbone_frozen = True
        self.backbone_train_mode = "frozen"

    def unfreeze_backbone(self) -> None:
        """Allow the backbone weights (or LoRA adapters) to receive gradients."""

        backbone = self._unwrap_model()
        backbone.train()
        for param in backbone.parameters():
            param.requires_grad_(True)
        self.backbone_frozen = False
        self.backbone_train_mode = "full"

    def enable_lora_training(self) -> None:
        """Train only LoRA adapter parameters while keeping base weights frozen."""

        if not self.use_lora:
            raise ValueError("LoRA adapters are not enabled for this model.")

        backbone = self._unwrap_model()
        for param in backbone.parameters():
            param.requires_grad_(False)

        trainable = 0
        for name, param in backbone.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad_(True)
                trainable += 1

        if trainable == 0:
            raise RuntimeError(
                "No LoRA parameters were found to fine-tune. Ensure adapters are initialised."
            )

        backbone.train()
        self.backbone_frozen = False
        self.backbone_train_mode = "lora"

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        """Yield parameters belonging to the backbone that require gradients."""

        backbone = self._unwrap_model()
        for param in backbone.parameters():
            if param.requires_grad:
                yield param

    def _build_classifier(self, num_labels: int) -> nn.Module:
        hidden_size = getattr(self._unwrap_model().config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Backbone model must define `config.hidden_size`." )

        activation_name = (self.classifier_config.activation or "gelu").lower()
        if activation_name not in {"relu", "gelu", "tanh"}:
            raise ValueError(
                "Unsupported activation for classifier head: "
                f"{self.classifier_config.activation!r}. Use 'relu', 'gelu', or 'tanh'."
            )

        activation_layer: nn.Module
        if activation_name == "relu":
            activation_layer = nn.ReLU()
        elif activation_name == "tanh":
            activation_layer = nn.Tanh()
        else:
            activation_layer = nn.GELU()

        dims = [hidden_size]
        if self.classifier_config.hidden_dims:
            dims.extend(int(d) for d in self.classifier_config.hidden_dims if int(d) > 0)
        dims.append(num_labels)

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != num_labels:
                layers.append(activation_layer)
                if self.classifier_config.dropout and self.classifier_config.dropout > 0:
                    layers.append(nn.Dropout(self.classifier_config.dropout))
        return nn.Sequential(*layers)

    def set_label_mapping(self, label2id: Dict[str, int]) -> None:
        self.label2id = dict(label2id)
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return pooled token representations."""

        backbone = self._unwrap_model()
        if self.backbone_frozen:
            backbone.eval()
            with torch.no_grad():
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
        else:
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1)
        masked_sum = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = masked_sum / lengths
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(input_ids, attention_mask)
        return self.classifier(features)

    def save_classifier(self, output_dir: str | Path, backbone_dir: str | Path | None = None) -> None:
        if not self.label2id:
            raise ValueError("Label mapping is empty. Call set_label_mapping() before saving.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        head_path = output_path / "classifier.pt"
        torch.save(self.classifier.state_dict(), head_path)

        metadata = {
            "label2id": self.label2id,
            "id2label": {str(idx): label for idx, label in self.id2label.items()},
            "head_config": self.classifier_config.to_dict(),
            "model_name": self.model_name,
            "backbone_finetuned": not self.backbone_frozen,
            "use_lora": self.use_lora,
            "backbone_train_mode": self.backbone_train_mode,
        }
        if backbone_dir is not None:
            backbone_path = Path(backbone_dir)
            try:
                backbone_path = backbone_path.relative_to(output_path)
            except ValueError:
                pass
            metadata["backbone_dir"] = str(backbone_path)
        config_path = output_path / "classifier_config.json"
        config_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_classifier(self, directory: str | Path) -> Dict[str, Any]:
        directory_path = Path(directory)
        config_path = directory_path / "classifier_config.json"
        head_path = directory_path / "classifier.pt"

        if not config_path.exists() or not head_path.exists():
            raise FileNotFoundError(
                f"Classifier files not found in {directory_path}. Expected classifier.pt and classifier_config.json."
            )

        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        head_config = metadata.get("head_config", {})
        self.classifier_config = ClassifierHeadConfig.from_dict(head_config)

        label2id = metadata.get("label2id", {})
        if not label2id:
            raise ValueError("Loaded metadata does not contain a label2id mapping.")
        # Ensure integer ids
        processed_label2id = {str(label): int(idx) for label, idx in label2id.items()}
        self.set_label_mapping(processed_label2id)

        num_labels = len(self.label2id)
        self.classifier = self._build_classifier(num_labels).to(self.device)
        state_dict = torch.load(head_path, map_location=self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()
        self.freeze_backbone()
        return metadata

    @staticmethod
    def load_metadata(directory: str | Path) -> Dict[str, Any]:
        config_path = Path(directory) / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No classifier_config.json found in {directory}."
            )
        return json.loads(config_path.read_text(encoding="utf-8"))
