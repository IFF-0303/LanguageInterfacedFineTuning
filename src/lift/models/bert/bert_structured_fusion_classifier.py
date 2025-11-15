"""BERT-based structured fusion classifier that combines text and structured features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.lift.models.gptj.feature_extractor_classifier import ClassifierHeadConfig
from src.lift.models.gptj.llm_structured_fusion_classifier import (
    StructuredEncoderConfig,
    StructuredFeatureEncoder,
    StructuredFusionDataset,
    StructuredFusionHead,
)


class BertStructuredFusionClassifier(nn.Module):
    """Wrap a BERT encoder with structured feature fusion for classification."""

    def __init__(
        self,
        *,
        model_name: str,
        num_labels: int,
        classifier_config: Optional[ClassifierHeadConfig] = None,
        structured_config: Optional[StructuredEncoderConfig] = None,
        fusion_mode: str = "concat",
        freeze_backbone: bool = True,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classifier_config = classifier_config or ClassifierHeadConfig()
        self.structured_config = structured_config or StructuredEncoderConfig()
        self.fusion_mode = fusion_mode
        self.num_labels = int(num_labels)

        load_path = model_path or model_name
        self.config = AutoConfig.from_pretrained(load_path)
        if getattr(self.config, "num_labels", None) != self.num_labels:
            self.config.num_labels = self.num_labels
        self.backbone = AutoModel.from_pretrained(load_path, config=self.config)
        self.backbone.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("BERT backbone must define `config.hidden_size`.")

        self.structured_encoder = StructuredFeatureEncoder(self.structured_config).to(self.device)
        self.fusion_head = StructuredFusionHead(
            llm_hidden_dim=int(hidden_size),
            structured_dim=self.structured_config.structured_dim,
            num_labels=self.num_labels,
            config=self.classifier_config,
            fusion_mode=self.fusion_mode,
        ).to(self.device)
        self.classifier = self.fusion_head

        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self._structured_metadata: Dict[str, Any] = {}
        self.backbone_frozen: bool = False

        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    # ---------------------------------------------------------------------
    # Backbone control helpers
    # ---------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()
        self.backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad_(True)
        self.backbone.train()
        self.backbone_frozen = False

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    # ---------------------------------------------------------------------
    # Structured feature helpers
    # ---------------------------------------------------------------------
    def structured_parameters(self) -> Iterable[nn.Parameter]:
        for module in (self.structured_encoder, self.fusion_head):
            for param in module.parameters():
                if param.requires_grad:
                    yield param

    def set_structured_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._structured_metadata = dict(metadata)

    # ---------------------------------------------------------------------
    # Label utilities
    # ---------------------------------------------------------------------
    def set_label_mapping(self, label2id: Mapping[str, int]) -> None:
        self.label2id = {str(label): int(idx) for label, idx in label2id.items()}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    # ---------------------------------------------------------------------
    # Forward pass helpers
    # ---------------------------------------------------------------------
    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        if getattr(outputs, "last_hidden_state", None) is not None:
            last_hidden = outputs.last_hidden_state
        elif getattr(outputs, "hidden_states", None):
            last_hidden = outputs.hidden_states[-1]
        else:
            raise AttributeError(
                "Backbone must return `last_hidden_state` when `output_hidden_states=True`."
            )

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        masked_sum = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / lengths

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

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
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
        self.classifier_config = ClassifierHeadConfig.from_dict(metadata.get("head_config", {}))
        self.structured_config = StructuredEncoderConfig.from_dict(
            metadata.get("structured_config", {})
        )
        self.fusion_mode = metadata.get("fusion_mode", "concat")

        label2id = metadata.get("label2id", {})
        if not label2id:
            raise ValueError("Loaded metadata does not contain a label2id mapping.")
        self.set_label_mapping(label2id)

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("BERT backbone must define `config.hidden_size`.")

        self.structured_encoder = StructuredFeatureEncoder(self.structured_config).to(self.device)
        self.fusion_head = StructuredFusionHead(
            llm_hidden_dim=int(hidden_size),
            structured_dim=self.structured_config.structured_dim,
            num_labels=len(self.label2id),
            config=self.classifier_config,
            fusion_mode=self.fusion_mode,
        ).to(self.device)
        self.classifier = self.fusion_head

        state = torch.load(head_path, map_location=self.device)
        self.structured_encoder.load_state_dict(state["structured_encoder"])
        self.fusion_head.load_state_dict(state["fusion_head"])
        self.structured_encoder.eval()
        self.fusion_head.eval()
        self.freeze_backbone()

        self.set_structured_metadata(metadata.get("structured_metadata", {}))
        return metadata

    @staticmethod
    def load_metadata(directory: str | Path) -> Dict[str, Any]:
        config_path = Path(directory) / "classifier_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No classifier_config.json found in {directory}.")
        return json.loads(config_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        device = kwargs.get("device") or (args[0] if args else None)
        if device is not None:
            self.device = torch.device(device)
        return self


__all__ = [
    "BertStructuredFusionClassifier",
    "StructuredFusionDataset",
    "StructuredEncoderConfig",
    "StructuredFeatureEncoder",
    "StructuredFusionHead",
]
