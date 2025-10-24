"""Thin wrapper re-exporting the generic LoRA fine-tuner for downstream tasks.

The classification experiments historically imported this module. To keep the
same import surface while switching to an open-source LLM backend we simply
forward the implementations from :mod:`gptj.lora_gptj`.
"""

from lift.models.gptj.lora_gptj import AverageMeter, GPTJDataset, LoRaQGPTJ  # noqa: F401

__all__ = ["AverageMeter", "GPTJDataset", "LoRaQGPTJ"]
