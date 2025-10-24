"""Distributed adapter around the Qwen-based LoRA fine-tuner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup

from gptj.lora_gptj import LoRaQGPTJ as _BaseLoRaQGPTJ


@dataclass
class DistributedConfig:
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    world_size: int = 1
    local_rank: int = 0


class LoRaQGPTJ(_BaseLoRaQGPTJ):
    """LoRA fine-tuner with optional DistributedDataParallel support."""

    def finetune(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        train_configs: Dict | DistributedConfig | None = None,
        saving_checkpoint: bool = False,
    ):
        cfg = (
            train_configs
            if isinstance(train_configs, DistributedConfig)
            else DistributedConfig(**(train_configs or {}))
        )

        train_data = self.prepare_data(train_jsonl_path)
        val_data = self.prepare_data(val_jsonl_path)

        train_sampler = DistributedSampler(
            train_data,
            num_replicas=cfg.world_size,
            rank=cfg.local_rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_data,
            num_replicas=cfg.world_size,
            rank=cfg.local_rank,
            shuffle=False,
        )

        data_loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_data, batch_size=cfg.batch_size, sampler=val_sampler)
        total_steps = len(data_loader) * cfg.epochs

        model = DistributedDataParallel(
            self.model,
            device_ids=[cfg.local_rank] if self.device.type == "cuda" else None,
            output_device=cfg.local_rank if self.device.type == "cuda" else None,
            find_unused_parameters=True,
        )

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )

        best_loss = np.inf
        train_losses, val_losses = np.zeros(cfg.epochs), np.zeros(cfg.epochs)

        for epoch in range(cfg.epochs):
            train_sampler.set_epoch(epoch)
            tqdm_object = tqdm(data_loader, total=len(data_loader), desc=f"Epoch: {epoch + 1}")
            loss_meter = 0.0
            seen = 0

            for batch in tqdm_object:
                model.zero_grad(set_to_none=True)
                inputs = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch_size = inputs.size(0)
                loss_meter += loss.detach().item() * batch_size
                seen += batch_size
                tqdm_object.set_postfix(train_loss=loss_meter / max(seen, 1))

            train_losses[epoch] = loss_meter / max(seen, 1)
            self.model = model.module
            val_loss = super().validate(val_loader)
            val_losses[epoch] = val_loss

            if saving_checkpoint and val_loss < best_loss:
                best_loss = val_loss
                self.save_networks(self.model_path)

        return train_losses, val_losses
