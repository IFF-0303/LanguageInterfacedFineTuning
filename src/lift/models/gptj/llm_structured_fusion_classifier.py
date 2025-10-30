"""Structured-data fusion classifier that augments LLM representations with tabular features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from .feature_extractor_classifier import (
    ClassifierHeadConfig,
    InstructionClassificationDataset,
    LLMFeatureExtractorClassifier,
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (float, int)):
        return math.isnan(float(value))
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):  # pragma: no cover - optional dependency
            return True
    except Exception:
        pass
    return False


def _to_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StructuredEncoderConfig:
    """Configuration for encoding structured numeric and categorical features."""

    numeric_feature_dim: int = 0
    categorical_cardinalities: List[int] = field(default_factory=list)
    structured_dim: int = 256
    categorical_embedding_dim: int = 32
    numeric_hidden_dim: Optional[int] = None
    dropout: float = 0.1
    use_numeric_missing_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_feature_dim": self.numeric_feature_dim,
            "categorical_cardinalities": list(self.categorical_cardinalities),
            "structured_dim": self.structured_dim,
            "categorical_embedding_dim": self.categorical_embedding_dim,
            "numeric_hidden_dim": self.numeric_hidden_dim,
            "dropout": self.dropout,
            "use_numeric_missing_embeddings": self.use_numeric_missing_embeddings,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StructuredEncoderConfig":
        return cls(
            numeric_feature_dim=int(data.get("numeric_feature_dim", 0)),
            categorical_cardinalities=list(data.get("categorical_cardinalities", [])),
            structured_dim=int(data.get("structured_dim", 256)),
            categorical_embedding_dim=int(data.get("categorical_embedding_dim", 32)),
            numeric_hidden_dim=data.get("numeric_hidden_dim"),
            dropout=float(data.get("dropout", 0.1)),
            use_numeric_missing_embeddings=bool(data.get("use_numeric_missing_embeddings", True)),
        )


class StructuredFeatureEncoder(nn.Module):
    """Embed numeric and categorical features into a dense representation."""

    def __init__(self, config: StructuredEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.numeric_dim = int(config.numeric_feature_dim)
        self.categorical_cardinalities = list(config.categorical_cardinalities)
        self.num_categorical = len(self.categorical_cardinalities)
        self.structured_dim = int(config.structured_dim)

        activation = nn.GELU()
        if self.numeric_dim > 0:
            numeric_hidden = config.numeric_hidden_dim or max(self.numeric_dim * 2, self.structured_dim)
            self.numeric_encoder = nn.Sequential(
                nn.Linear(self.numeric_dim * 2, numeric_hidden),
                activation,
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(numeric_hidden, self.structured_dim),
            )
            if config.use_numeric_missing_embeddings:
                self.numeric_missing_embeddings = nn.Parameter(
                    torch.zeros(self.numeric_dim, self.structured_dim)
                )
            else:
                self.numeric_missing_embeddings = None
        else:
            self.numeric_encoder = None
            self.numeric_missing_embeddings = None

        if self.num_categorical > 0:
            self.categorical_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, config.categorical_embedding_dim)
                    for cardinality in self.categorical_cardinalities
                ]
            )
            cat_input_dim = self.num_categorical * config.categorical_embedding_dim
            self.categorical_encoder = nn.Sequential(
                nn.Linear(cat_input_dim, max(cat_input_dim, self.structured_dim)),
                activation,
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
                nn.Linear(max(cat_input_dim, self.structured_dim), self.structured_dim),
            )
        else:
            self.categorical_embeddings = nn.ModuleList()
            self.categorical_encoder = None

    def forward(
        self,
        *,
        numeric_features: Optional[torch.Tensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        categorical_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size: Optional[int] = None
        components: List[torch.Tensor] = []

        if self.numeric_encoder is not None and numeric_features is not None and numeric_mask is not None:
            if batch_size is None:
                batch_size = numeric_features.size(0)
            concat = torch.cat([numeric_features, numeric_mask], dim=-1)
            numeric_repr = self.numeric_encoder(concat)
            if self.numeric_missing_embeddings is not None:
                missing_mask = (1.0 - numeric_mask).unsqueeze(-1)
                missing_bias = torch.sum(
                    missing_mask * self.numeric_missing_embeddings.unsqueeze(0), dim=1
                )
                numeric_repr = numeric_repr + missing_bias
            components.append(numeric_repr)

        if (
            self.categorical_encoder is not None
            and categorical_features is not None
            and categorical_features.numel() > 0
        ):
            if batch_size is None:
                batch_size = categorical_features.size(0)
            embeddings: List[torch.Tensor] = []
            for idx, embedding_layer in enumerate(self.categorical_embeddings):
                feature = categorical_features[:, idx]
                emb = embedding_layer(feature)
                if categorical_mask is not None:
                    mask_column = categorical_mask[:, idx].unsqueeze(-1)
                    emb = emb * mask_column
                embeddings.append(emb)
            cat_concat = torch.cat(embeddings, dim=-1)
            cat_repr = self.categorical_encoder(cat_concat)
            components.append(cat_repr)

        if not components:
            if batch_size is None:
                raise ValueError("StructuredFeatureEncoder received no inputs to encode.")
            device = numeric_features.device if numeric_features is not None else categorical_features.device
            return torch.zeros(batch_size, self.structured_dim, device=device)

        if len(components) == 1:
            return components[0]
        return torch.stack(components, dim=0).mean(dim=0)


def _build_mlp(input_dim: int, num_labels: int, config: ClassifierHeadConfig) -> nn.Sequential:
    activation_name = (config.activation or "gelu").lower()
    if activation_name not in {"relu", "gelu", "tanh"}:
        raise ValueError(
            f"Unsupported activation {config.activation!r}. Choose from 'relu', 'gelu', or 'tanh'."
        )
    if activation_name == "relu":
        activation_layer: nn.Module = nn.ReLU()
    elif activation_name == "tanh":
        activation_layer = nn.Tanh()
    else:
        activation_layer = nn.GELU()

    dims = [input_dim]
    if config.hidden_dims:
        dims.extend(int(d) for d in config.hidden_dims if int(d) > 0)
    dims.append(num_labels)

    layers: List[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != num_labels:
            layers.append(activation_layer)
            if config.dropout and config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
    return nn.Sequential(*layers)


class StructuredFusionHead(nn.Module):
    """Classifier head that fuses LLM and structured representations."""

    def __init__(
        self,
        llm_hidden_dim: int,
        structured_dim: int,
        num_labels: int,
        config: ClassifierHeadConfig,
        *,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()
        fusion_mode = fusion_mode.lower()
        if fusion_mode not in {"concat", "gated_add", "film"}:
            raise ValueError(
                f"Unsupported fusion_mode {fusion_mode!r}. Choose from 'concat', 'gated_add', or 'film'."
            )
        self.fusion_mode = fusion_mode
        self.config = config

        if fusion_mode == "concat":
            input_dim = llm_hidden_dim + structured_dim
            self.tab_projection = None
            self.gate = None
            self.film_gamma = None
            self.film_beta = None
        elif fusion_mode == "gated_add":
            input_dim = llm_hidden_dim
            self.tab_projection = nn.Linear(structured_dim, llm_hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(llm_hidden_dim + structured_dim, llm_hidden_dim),
                nn.Sigmoid(),
            )
            self.film_gamma = None
            self.film_beta = None
        else:  # film
            input_dim = llm_hidden_dim
            self.tab_projection = None
            self.gate = None
            self.film_gamma = nn.Linear(structured_dim, llm_hidden_dim)
            self.film_beta = nn.Linear(structured_dim, llm_hidden_dim)

        self.classifier = _build_mlp(input_dim, num_labels, config)

    def forward(self, llm_features: torch.Tensor, structured_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            fused = torch.cat([llm_features, structured_features], dim=-1)
        elif self.fusion_mode == "gated_add":
            assert self.tab_projection is not None and self.gate is not None
            projected = self.tab_projection(structured_features)
            gate_values = self.gate(torch.cat([llm_features, structured_features], dim=-1))
            fused = gate_values * llm_features + (1.0 - gate_values) * projected
        else:  # film
            assert self.film_gamma is not None and self.film_beta is not None
            gamma = self.film_gamma(structured_features)
            beta = self.film_beta(structured_features)
            fused = (1.0 + gamma) * llm_features + beta
        return self.classifier(fused)


class StructuredFusionDataset(InstructionClassificationDataset):
    """Instruction dataset that also provides structured numeric and categorical features."""

    def __init__(
        self,
        examples: Iterable[Dict[str, Any]],
        tokenizer,
        *,
        numeric_fields: Sequence[str] | None = None,
        categorical_fields: Sequence[str] | None = None,
        numeric_stats: Optional[Mapping[str, Sequence[float]]] = None,
        categorical_vocabs: Optional[Mapping[str, Mapping[str, int]]] = None,
        categorical_missing_indices: Optional[Mapping[str, int]] = None,
        label2id: Optional[Dict[str, int]] = None,
        label_field: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        numeric_fields = list(numeric_fields or [])
        categorical_fields = list(categorical_fields or [])
        self.numeric_fields = numeric_fields
        self.categorical_fields = categorical_fields

        super().__init__(
            examples,
            tokenizer,
            label2id=label2id,
            label_field=label_field,
            max_length=max_length,
        )

        self.numeric_stats = self._build_numeric_stats(numeric_stats)
        self.categorical_vocabs, self.categorical_missing_indices = self._build_categorical_vocabs(
            categorical_vocabs, categorical_missing_indices
        )

        self.numeric_tensor, self.numeric_mask = self._encode_numeric_features()
        self.categorical_tensor, self.categorical_mask = self._encode_categorical_features()

    def _build_numeric_stats(
        self, numeric_stats: Optional[Mapping[str, Sequence[float]]]
    ) -> Dict[str, List[float]]:
        if not self.numeric_fields:
            return {"mean": [], "std": []}

        if numeric_stats is not None:
            mean = [float(v) for v in numeric_stats.get("mean", [])]
            std = [float(v) for v in numeric_stats.get("std", [])]
            if len(mean) != len(self.numeric_fields) or len(std) != len(self.numeric_fields):
                raise ValueError("numeric_stats dimensions do not match provided numeric_fields.")
            return {"mean": mean, "std": std}

        values_per_feature: List[List[float]] = [[] for _ in self.numeric_fields]
        for example in self.examples:
            for idx, field in enumerate(self.numeric_fields):
                value = _to_float(example.get(field))
                if value is not None:
                    values_per_feature[idx].append(value)

        mean: List[float] = []
        std: List[float] = []
        for values in values_per_feature:
            if not values:
                mean.append(0.0)
                std.append(1.0)
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            mean.append(tensor.mean().item())
            std_value = tensor.std(unbiased=False).item()
            if std_value <= 1e-6:
                std_value = 1.0
            std.append(std_value)
        return {"mean": mean, "std": std}

    def _build_categorical_vocabs(
        self,
        categorical_vocabs: Optional[Mapping[str, Mapping[str, int]]],
        categorical_missing_indices: Optional[Mapping[str, int]],
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        if not self.categorical_fields:
            return {}, {}

        if categorical_vocabs is not None:
            vocabs = {field: dict(mapping) for field, mapping in categorical_vocabs.items()}
            missing_indices = {field: int(idx) for field, idx in (categorical_missing_indices or {}).items()}
            return vocabs, missing_indices

        vocabs: Dict[str, Dict[str, int]] = {field: {} for field in self.categorical_fields}
        for example in self.examples:
            for field in self.categorical_fields:
                value = example.get(field)
                if _is_missing(value):
                    continue
                key = str(value)
                vocab = vocabs[field]
                if key not in vocab:
                    vocab[key] = len(vocab)

        missing_indices: Dict[str, int] = {}
        for field, vocab in vocabs.items():
            missing_indices[field] = len(vocab)
        return vocabs, missing_indices

    def _encode_numeric_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.numeric_fields:
            zeros = torch.zeros(len(self), 0, dtype=torch.float32)
            return zeros, zeros

        means = torch.tensor(self.numeric_stats["mean"], dtype=torch.float32)
        stds = torch.tensor(self.numeric_stats["std"], dtype=torch.float32)

        features = torch.zeros(len(self), len(self.numeric_fields), dtype=torch.float32)
        mask = torch.zeros(len(self), len(self.numeric_fields), dtype=torch.float32)

        for row_idx, example in enumerate(self.examples):
            for col_idx, field in enumerate(self.numeric_fields):
                value = _to_float(example.get(field))
                if value is None:
                    continue
                mask[row_idx, col_idx] = 1.0
                normalized = (value - means[col_idx].item()) / stds[col_idx].item()
                features[row_idx, col_idx] = normalized
        return features, mask

    def _encode_categorical_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.categorical_fields:
            zeros_float = torch.zeros(len(self), 0, dtype=torch.float32)
            zeros_long = torch.zeros(len(self), 0, dtype=torch.long)
            return zeros_long, zeros_float

        tensor = torch.zeros(len(self), len(self.categorical_fields), dtype=torch.long)
        mask = torch.zeros(len(self), len(self.categorical_fields), dtype=torch.float32)

        for row_idx, example in enumerate(self.examples):
            for col_idx, field in enumerate(self.categorical_fields):
                vocab = self.categorical_vocabs[field]
                missing_index = self.categorical_missing_indices[field]
                value = example.get(field)
                if _is_missing(value):
                    tensor[row_idx, col_idx] = missing_index
                    continue
                key = str(value)
                idx = vocab.get(key, missing_index)
                tensor[row_idx, col_idx] = idx
                mask[row_idx, col_idx] = 1.0
        return tensor, mask

    @property
    def numeric_feature_dim(self) -> int:
        return len(self.numeric_fields)

    @property
    def categorical_cardinalities(self) -> List[int]:
        return [len(self.categorical_vocabs[field]) + 1 for field in self.categorical_fields]

    def structured_metadata(self) -> Dict[str, Any]:
        return {
            "numeric_fields": list(self.numeric_fields),
            "numeric_stats": self.numeric_stats,
            "categorical_fields": list(self.categorical_fields),
            "categorical_vocabs": self.categorical_vocabs,
            "categorical_missing_indices": self.categorical_missing_indices,
        }

    def __getitem__(self, idx: int):  # type: ignore[override]
        base = super().__getitem__(idx)
        if self.labels is None:
            input_ids, attention_mask = base  # type: ignore[misc]
            label = None
        else:
            input_ids, attention_mask, label = base  # type: ignore[misc]

        numeric_row = self.numeric_tensor[idx]
        numeric_mask = self.numeric_mask[idx]
        categorical_row = self.categorical_tensor[idx]
        categorical_mask = self.categorical_mask[idx]

        if label is None:
            return (
                input_ids,
                attention_mask,
                numeric_row,
                numeric_mask,
                categorical_row,
                categorical_mask,
            )
        return (
            input_ids,
            attention_mask,
            numeric_row,
            numeric_mask,
            categorical_row,
            categorical_mask,
            label,
        )


class LLMStructuredFusionClassifier(LLMFeatureExtractorClassifier):
    """LLM classifier that fuses textual and structured features before classification."""

    def __init__(
        self,
        *,
        num_labels: int,
        classifier_config: Optional[ClassifierHeadConfig] = None,
        structured_config: Optional[StructuredEncoderConfig] = None,
        fusion_mode: str = "concat",
        freeze_backbone: bool = True,
        wrap_backbone_with_ddp: bool = True,
        **kwargs: Any,
    ) -> None:
        self.structured_config = structured_config or StructuredEncoderConfig()
        super().__init__(
            num_labels=num_labels,
            classifier_config=classifier_config,
            freeze_backbone=freeze_backbone,
            wrap_backbone_with_ddp=wrap_backbone_with_ddp,
            **kwargs,
        )

        hidden_size = getattr(self._unwrap_model().config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Backbone model must define hidden_size in its config.")

        self.structured_encoder = StructuredFeatureEncoder(self.structured_config).to(self.device)
        self.fusion_head = StructuredFusionHead(
            llm_hidden_dim=int(hidden_size),
            structured_dim=self.structured_config.structured_dim,
            num_labels=num_labels,
            config=self.classifier_config,
            fusion_mode=fusion_mode,
        ).to(self.device)
        self.fusion_mode = fusion_mode
        self.classifier = self.fusion_head
        self._structured_metadata: Dict[str, Any] = {}

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_features: Optional[torch.Tensor] = None,
        numeric_mask: Optional[torch.Tensor] = None,
        categorical_features: Optional[torch.Tensor] = None,
        categorical_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        llm_features = self.extract_features(input_ids, attention_mask)
        structured_features = self.structured_encoder(
            numeric_features=numeric_features,
            numeric_mask=numeric_mask,
            categorical_features=categorical_features,
            categorical_mask=categorical_mask,
        )
        return self.fusion_head(llm_features, structured_features)

    def structured_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.structured_encoder, self.fusion_head):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def set_structured_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._structured_metadata = dict(metadata)

    def save_classifier(
        self,
        output_dir: str | Path,
        backbone_dir: str | Path | None = None,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.label2id:
            raise ValueError("Label mapping is empty. Call set_label_mapping() before saving.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        head_path = output_path / "classifier.pt"
        torch.save(
            {
                "structured_encoder": self.structured_encoder.state_dict(),
                "fusion_head": self.fusion_head.state_dict(),
            },
            head_path,
        )

        metadata: Dict[str, Any] = {
            "label2id": self.label2id,
            "id2label": {str(idx): label for idx, label in self.id2label.items()},
            "head_config": self.classifier_config.to_dict(),
            "model_name": self.model_name,
            "backbone_finetuned": not self.backbone_frozen,
            "use_lora": self.use_lora,
            "backbone_train_mode": self.backbone_train_mode,
            "fusion_mode": self.fusion_mode,
            "structured_config": self.structured_config.to_dict(),
            "structured_metadata": self._structured_metadata,
        }
        if backbone_dir is not None:
            backbone_path = Path(backbone_dir)
            try:
                backbone_path = backbone_path.relative_to(output_path)
            except ValueError:
                pass
            metadata["backbone_dir"] = str(backbone_path)
        if extra_metadata:
            metadata.update(extra_metadata)
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

        structured_config_dict = metadata.get("structured_config", {})
        self.structured_config = StructuredEncoderConfig.from_dict(structured_config_dict)

        label2id = metadata.get("label2id", {})
        if not label2id:
            raise ValueError("Loaded metadata does not contain a label2id mapping.")
        processed_label2id = {str(label): int(idx) for label, idx in label2id.items()}
        self.set_label_mapping(processed_label2id)

        num_labels = len(self.label2id)
        hidden_size = getattr(self._unwrap_model().config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Backbone model must define hidden_size in its config.")

        self.structured_encoder = StructuredFeatureEncoder(self.structured_config).to(self.device)
        self.fusion_head = StructuredFusionHead(
            llm_hidden_dim=int(hidden_size),
            structured_dim=self.structured_config.structured_dim,
            num_labels=num_labels,
            config=self.classifier_config,
            fusion_mode=metadata.get("fusion_mode", "concat"),
        ).to(self.device)
        self.fusion_mode = metadata.get("fusion_mode", "concat")
        self.classifier = self.fusion_head

        state = torch.load(head_path, map_location=self.device)
        self.structured_encoder.load_state_dict(state["structured_encoder"])
        self.fusion_head.load_state_dict(state["fusion_head"])
        self.structured_encoder.eval()
        self.fusion_head.eval()
        self.freeze_backbone()

        structured_metadata = metadata.get("structured_metadata", {})
        self.set_structured_metadata(structured_metadata)
        return metadata

    @staticmethod
    def load_metadata(directory: str | Path) -> Dict[str, Any]:
        config_path = Path(directory) / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No classifier_config.json found in {directory}.")
        return json.loads(config_path.read_text(encoding="utf-8"))
