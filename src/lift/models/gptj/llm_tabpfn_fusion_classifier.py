"""Fusion model that combines LLM and TabPFN feature extractors for classification."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .feature_extractor_classifier import (
    ClassifierHeadConfig,
    InstructionClassificationDataset,
    LLMFeatureExtractorClassifier,
)


class FusionClassifierHead(nn.Module):
    """Fusion-aware MLP head that combines text and tabular features."""

    def __init__(
        self,
        llm_hidden_dim: int,
        tab_feature_dim: int,
        num_labels: int,
        config: ClassifierHeadConfig,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()

        if tab_feature_dim <= 0:
            raise ValueError("tab_feature_dim must be a positive integer.")

        self.fusion_mode = fusion_mode
        self.config = config

        activation_name = (config.activation or "gelu").lower()
        if activation_name not in {"relu", "gelu", "tanh"}:
            raise ValueError(
                "Unsupported activation for classifier head: "
                f"{config.activation!r}. Use 'relu', 'gelu', or 'tanh'."
            )

        if activation_name == "relu":
            activation_layer: nn.Module = nn.ReLU()
        elif activation_name == "tanh":
            activation_layer = nn.Tanh()
        else:
            activation_layer = nn.GELU()

        self.activation = activation_layer

        if fusion_mode == "concat":
            input_dim = llm_hidden_dim + tab_feature_dim
            self.tab_projection = None
            self.gate = None
        elif fusion_mode == "gated_add":
            input_dim = llm_hidden_dim
            self.tab_projection = nn.Linear(tab_feature_dim, llm_hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(llm_hidden_dim + tab_feature_dim, llm_hidden_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(
                f"Unsupported fusion mode: {fusion_mode!r}. Use 'concat' or 'gated_add'."
            )

        dims: List[int] = [input_dim]
        if config.hidden_dims:
            dims.extend(int(d) for d in config.hidden_dims if int(d) > 0)
        dims.append(num_labels)

        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != num_labels:
                layers.append(self.activation)
                if config.dropout and config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))

        self.classifier = nn.Sequential(*layers)

    def forward(self, llm_features: torch.Tensor, tab_features: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "concat":
            fused = torch.cat([llm_features, tab_features], dim=-1)
        else:  # gated_add
            assert self.tab_projection is not None and self.gate is not None
            projected_tab = self.tab_projection(tab_features)
            gate_values = self.gate(torch.cat([llm_features, tab_features], dim=-1))
            fused = gate_values * llm_features + (1.0 - gate_values) * projected_tab
        return self.classifier(fused)


class TabPFNFeatureExtractor:
    """Light wrapper around TabPFNClassifier to produce tabular embeddings."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        categorical_features: Optional[Sequence[int]] = None,
    ) -> None:
        try:
            from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
        except ImportError as exc:  # pragma: no cover - dependency not always present
            raise ImportError(
                "tabpfn is required for TabPFNFeatureExtractor. Install via `pip install tabpfn`."
            ) from exc

        self.device = device
        if categorical_features is not None:
            categorical_indices = [int(idx) for idx in categorical_features]
        else:
            categorical_indices = None
        self.categorical_features_indices = categorical_indices
        tabpfn_kwargs: Dict[str, Any] = {"device": device}
        if categorical_indices is not None:
            tabpfn_kwargs["categorical_features_indices"] = categorical_indices
        self.model = TabPFNClassifier(**tabpfn_kwargs)
        self._fitted = False

    @staticmethod
    def _ensure_numpy(array_like: Iterable[Iterable[float]]) -> np.ndarray:
        arr = np.asarray(list(array_like), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Tabular features must be a 2D array-like structure.")
        return arr

    def fit(self, features: Iterable[Iterable[float]], labels: Iterable[int]) -> None:
        X = self._ensure_numpy(features)
        y = np.asarray(list(labels))
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of feature rows and labels must match for TabPFN training.")
        self.model.fit(X, y)
        self._fitted = True

    def transform(self, features: Iterable[Iterable[float]]) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("TabPFNFeatureExtractor must be fitted before calling transform().")
        X = self._ensure_numpy(features)
        proba = self.model.predict_proba(X)
        return torch.from_numpy(np.asarray(proba, dtype=np.float32))

    def save(self, output_path: str | Path) -> None:
        state = {
            "model": self.model,
            "fitted": self._fitted,
            "device": self.device,
            "categorical_features_indices": self.categorical_features_indices,
        }
        with Path(output_path).open("wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabPFNFeatureExtractor":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.model = state["model"]
        obj._fitted = state.get("fitted", False)
        obj.device = state.get("device", "cpu")
        obj.categorical_features_indices = state.get("categorical_features_indices")
        return obj


class FusionInstructionClassificationDataset(InstructionClassificationDataset):
    """Instruction dataset augmented with TabPFN features for fusion models."""

    def __init__(
        self,
        examples: Iterable[Dict[str, Any]],
        tokenizer,
        tab_features: torch.Tensor,
        *,
        label2id: Optional[Dict[str, int]] = None,
        label_field: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            examples,
            tokenizer,
            label2id=label2id,
            label_field=label_field,
            max_length=max_length,
        )

        if len(self.examples) != len(tab_features):
            raise ValueError(
                "Number of tabular feature rows must match the number of text examples."
            )
        if not isinstance(tab_features, torch.Tensor):
            tab_features = torch.tensor(tab_features, dtype=torch.float32)
        self.tab_features = tab_features.to(dtype=torch.float32)

    def __getitem__(self, idx: int):
        base = super().__getitem__(idx)
        if self.labels is not None:
            input_ids, attention_mask, labels = base
            return input_ids, attention_mask, self.tab_features[idx], labels
        input_ids, attention_mask = base
        return input_ids, attention_mask, self.tab_features[idx]


class LLMTabPFNFusionClassifier(LLMFeatureExtractorClassifier):
    """Extends :class:`LLMFeatureExtractorClassifier` with TabPFN fusion."""

    def __init__(
        self,
        *,
        num_labels: int,
        tab_feature_dim: int,
        fusion_mode: str = "concat",
        classifier_config: Optional[ClassifierHeadConfig] = None,
        freeze_backbone: bool = True,
        wrap_backbone_with_ddp: bool = True,
        **kwargs: Any,
    ) -> None:
        self.tab_feature_dim = int(tab_feature_dim)
        if self.tab_feature_dim <= 0:
            raise ValueError("tab_feature_dim must be a positive integer.")
        self.fusion_mode = fusion_mode
        super().__init__(
            num_labels=num_labels,
            classifier_config=classifier_config,
            freeze_backbone=freeze_backbone,
            wrap_backbone_with_ddp=wrap_backbone_with_ddp,
            **kwargs,
        )

    def _build_classifier(self, num_labels: int) -> nn.Module:
        hidden_size = getattr(self._unwrap_model().config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Backbone model must define `config.hidden_size`.")
        return FusionClassifierHead(
            llm_hidden_dim=hidden_size,
            tab_feature_dim=self.tab_feature_dim,
            num_labels=num_labels,
            config=self.classifier_config,
            fusion_mode=self.fusion_mode,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tab_features: torch.Tensor,
    ) -> torch.Tensor:
        llm_features = self.extract_features(input_ids, attention_mask)
        return self.classifier(llm_features, tab_features)

    def save_classifier(
        self,
        output_dir: str | Path,
        backbone_dir: str | Path | None = None,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = {
            "fusion_mode": self.fusion_mode,
            "tab_feature_dim": self.tab_feature_dim,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        super().save_classifier(output_dir, backbone_dir=backbone_dir, extra_metadata=metadata)

    def load_classifier(self, directory: str | Path) -> Dict[str, Any]:
        directory_path = Path(directory)
        config_path = directory_path / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No classifier_config.json found in {directory_path} for fusion classifier."
            )
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        fusion_mode = metadata.get("fusion_mode", self.fusion_mode)
        tab_dim = metadata.get("tab_feature_dim", self.tab_feature_dim)
        self.fusion_mode = fusion_mode
        self.tab_feature_dim = int(tab_dim)
        return super().load_classifier(directory)
