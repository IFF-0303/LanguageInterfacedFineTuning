import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, distributed
from tqdm import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM as HFAutoModelForCausalLM,
    AutoTokenizer as HFAutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

try:
    from modelscope import (
        AutoModelForCausalLM as MsAutoModelForCausalLM,
        AutoTokenizer as MsAutoTokenizer,
    )
except ImportError:  # pragma: no cover - optional dependency for ModelScope users
    MsAutoModelForCausalLM = None
    MsAutoTokenizer = None

try:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError as exc:  # pragma: no cover - dependency message for users
    raise ImportError(
        "The 'peft' library is required for LoRA-based fine-tuning."
        " Please install it with `pip install peft`."
    ) from exc


class AverageMeter(object):
    """Utility class for tracking streaming metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def build_instruction_prompt(example: Dict[str, Any]) -> str:
    """Construct a standard prompt from an instruction-style example."""

    instruction = example.get("instruction") or example.get("prompt")
    inp = example.get("input") or example.get("context")

    prompt_sections: List[str] = []
    if instruction:
        prompt_sections.append(f"Instruction:\n{str(instruction).strip()}")
    if inp:
        prompt_sections.append(f"Input:\n{str(inp).strip()}")

    if prompt_sections:
        return "\n\n".join(prompt_sections) + "\n\nResponse:"
    return "Response:"


class GPTJDataset(Dataset):
    """Generic dataset wrapper for instruction-style JSONL data."""

    def __init__(self, json_lst: Iterable[dict], tokenizer, max_length: int = 1024):
        texts: List[str] = []
        completion_lens: List[int] = []
        for row in json_lst:
            completion = row.get("completion") or row.get("output") or row.get("response")
            if completion is None:
                raise ValueError(
                    "Each training example must include a 'completion', 'output', or 'response' field."
                )

            prompt_text = build_instruction_prompt(row)
            text = f"{prompt_text} {completion}".strip()
            texts.append(text)

            completion_ids = tokenizer(
                str(completion),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            completion_lens.append(int(completion_ids.shape[-1]))

        tokens = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone()
            label_trim = max(0, self.input_ids[i].shape[0] - completion_lens[i])
            b_labels[:label_trim] = -100
            self.labels.append(b_labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


@dataclass
class LoRaConfigParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None


class LoRaQGPTJ:
    """LoRA fine-tuner targeting open-source LLMs via Hugging Face or ModelScope."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        adapter: bool = True,
        device: Optional[torch.device] = None,
        model_path: str = "../results/qwen/",
        cache_dir: Optional[str] = None,
        load_in_4bit: bool = False,
        lora_config: Optional[LoRaConfigParams] = None,
        trust_remote_code: bool = True,
        model_provider: str = "huggingface",
    ) -> None:
        self.model_name = model_name
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._global_rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", str(self._global_rank)))
        self._distributed = False

        if torch.cuda.is_available():
            if self._world_size > 1:
                if not dist.is_available():
                    raise RuntimeError("torch.distributed is required for multi-GPU fine-tuning.")
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                self._distributed = True
                torch.cuda.set_device(self._local_rank)
                self.device = torch.device(f"cuda:{self._local_rank}")
            else:
                self.device = device or torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self._is_main_process = not self._distributed or self._global_rank == 0
        self.model_path = model_path
        self.use_lora = adapter
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.model_provider = model_provider.lower()

        if self.model_provider not in {"huggingface", "modelscope"}:
            raise ValueError(
                "model_provider must be either 'huggingface' or 'modelscope', "
                f"got {model_provider!r}."
            )

        if self.model_provider == "modelscope" and MsAutoTokenizer is None:
            raise ImportError(
                "The 'modelscope' package is required for ModelScope-backed models. "
                "Install it with `pip install modelscope`."
            )

        self.tokenizer = self._load_tokenizer(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quant_config = None
        if load_in_4bit:
            if self.model_provider == "modelscope":
                raise ValueError(
                    "4-bit quantization via bitsandbytes is not currently supported "
                    "for ModelScope-provided models."
                )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        base_model = self._load_model(model_name, quant_config)

        if quant_config is None:
            base_model = base_model.to(self.device)

        base_model.config.use_cache = False

        if adapter:
            if quant_config is not None:
                base_model = prepare_model_for_kbit_training(base_model)

            lora_cfg = lora_config or LoRaConfigParams()
            target_modules = lora_cfg.target_modules or self._default_target_modules()
            config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            base_model = get_peft_model(base_model, config)
        else:
            base_model = base_model.to(self.device)

        if not load_in_4bit:
            base_model = base_model.to(self.device)

        self.model = self._wrap_model(base_model)

    def _default_target_modules(self) -> List[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    def load_networks(self, model_path: str) -> None:
        if self.use_lora:
            base_model = self._load_model(self.model_name)
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = self._load_model(model_path)
        model = model.to(self.device)
        self.model = self._wrap_model(model)

    def _load_tokenizer(self, identifier: str):
        if self.model_provider == "modelscope":
            return MsAutoTokenizer.from_pretrained(
                identifier,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
            )
        return HFAutoTokenizer.from_pretrained(
            identifier,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )

    def _load_model(self, identifier: str, quant_config: BitsAndBytesConfig | None = None):
        if self.model_provider == "modelscope":
            if MsAutoModelForCausalLM is None:
                raise ImportError(
                    "The 'modelscope' package is required for ModelScope-backed models. "
                    "Install it with `pip install modelscope`."
                )
            if quant_config is not None:
                raise ValueError(
                    "Quantization via BitsAndBytes is unavailable for ModelScope models."
                )
            return MsAutoModelForCausalLM.from_pretrained(
                identifier,
                cache_dir=self.cache_dir,
                trust_remote_code=self.trust_remote_code,
            )

        return HFAutoModelForCausalLM.from_pretrained(
            identifier,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if quant_config is not None else None,
        )

    def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap the base model to utilise all visible GPUs on a single machine."""

        if self._distributed:
            return torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=False,
            )
        return model

    def _unwrap_model(self) -> torch.nn.Module:
        """Return the underlying model regardless of any wrappers."""

        return self.model.module if hasattr(self.model, "module") else self.model

    @property
    def is_main_process(self) -> bool:
        return self._is_main_process

    @property
    def is_distributed(self) -> bool:
        return self._distributed

    def prepare_data(self, jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as json_file:
            raw_content = json_file.read().strip()

        if not raw_content:
            raise ValueError(f"Dataset file '{jsonl_path}' is empty.")

        txt_list: List[dict]
        try:
            # Try jsonlines first
            txt_list = [json.loads(line) for line in raw_content.splitlines() if line.strip()]
        except json.JSONDecodeError:
            parsed = json.loads(raw_content)
            if isinstance(parsed, dict):
                txt_list = [parsed]
            elif isinstance(parsed, list):
                txt_list = parsed
            else:
                raise ValueError(
                    "Unsupported dataset format: expected a JSON object, list, or JSONL entries."
                )

        for row in txt_list:
            if "completion" not in row and "output" in row:
                row["completion"] = row["output"]

        data = GPTJDataset(txt_list, self.tokenizer)

        return data

    def finetune(
        self,
        train_jsonl_path,
        val_jsonl_path,
        train_configs={'batch_size': 8, 'epochs': 20, 'learning_rate': 1e-3, 'weight_decay': 0.01, 'warmup_steps': 20},
        saving_checkpoint: bool = False,
    ):
        train_data = self.prepare_data(train_jsonl_path)
        val_data = self.prepare_data(val_jsonl_path)

        train_sampler: Optional[Sampler] = None
        val_sampler: Optional[Sampler] = None
        if self._distributed:
            train_sampler = distributed.DistributedSampler(train_data, shuffle=True)
            val_sampler = distributed.DistributedSampler(val_data, shuffle=False)

        data_loader = DataLoader(
            train_data,
            batch_size=train_configs['batch_size'],
            shuffle=train_sampler is None,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=train_configs['batch_size'],
            shuffle=False,
            sampler=val_sampler,
        )
        total_steps = len(data_loader) * train_configs['epochs']

        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_configs['learning_rate'],
            weight_decay=train_configs.get('weight_decay', 0.01),
        )
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = train_configs['warmup_steps'],
                                            num_training_steps = total_steps)

        best_loss = np.inf
        # with torch.cuda.amp.autocast():
        train_losses, val_losses = np.zeros(train_configs['epochs']), np.zeros(train_configs['epochs'])
        for epoch in range(train_configs['epochs']):
            if self._distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            self.model.train()
            tqdm_object = tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Epoch: {epoch + 1}",
                disable=not self._is_main_process,
            )
            loss_meter = AverageMeter()
            for batch in tqdm_object:
                self.model.zero_grad(set_to_none=True)
                inputs = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                outputs = self.model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.detach().item(), batch[0].shape[0])
                if self._is_main_process:
                    tqdm_object.set_postfix(train_loss=loss_meter.avg)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            train_stats = torch.tensor(
                [loss_meter.sum, loss_meter.count],
                device=self.device,
                dtype=torch.float64,
            )
            if self._distributed:
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            global_train_loss = (train_stats[0] / train_stats[1]).item()

            val_loss = self.validate(val_loader)
            val_losses[epoch] = val_loss
            train_losses[epoch] = global_train_loss
            if saving_checkpoint and val_loss < best_loss:
                if self._is_main_process:
                    print('Saving the best model with loss {:.4f}'.format(val_loss))
                    self.save_networks(self.model_path)
                best_loss = val_loss
        return train_losses, val_losses
                
        

    def validate(self, val_loader):
        # ========================================
        #               Validation
        # ========================================
        self.model.eval()
        # Evaluate data for one epoch
        loss_meter = AverageMeter()
        tqdm_object = tqdm(
            val_loader,
            total=len(val_loader),
            desc='Validation',
            disable=not self._is_main_process,
        )
        for batch in tqdm_object:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch[0].to(self.device),
                    attention_mask=batch[1].to(self.device),
                    labels=batch[2].to(self.device),
                )
                loss = outputs.loss

            loss_meter.update(loss.detach().item(), batch[0].shape[0])
            if self._is_main_process:
                tqdm_object.set_postfix(val_loss=loss_meter.avg)

        val_stats = torch.tensor(
            [loss_meter.sum, loss_meter.count],
            device=self.device,
            dtype=torch.float64,
        )
        if self._distributed:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        return (val_stats[0] / val_stats[1]).item()

    def generate(self, text_lst, deterministic=True, max_token=10, batch_size=10, temperature: float = 1.0):
        self.model.eval()
        if self._distributed and not self._is_main_process:
            return []
        outputs: List[str] = []
        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors='pt',
            )
            prompt = {key: value.to(self.device) for key, value in prompt.items()}
            generation_kwargs = dict(
                max_new_tokens=max_token,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=not deterministic,
                early_stopping=True,
                temperature=temperature,
            )
            model = self._unwrap_model()
            model.eval()
            sequences = model.generate(**prompt, **generation_kwargs)
            input_lengths = prompt["attention_mask"].sum(dim=1)
            for seq, length in zip(sequences, input_lengths):
                generated = seq[int(length):]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                outputs.append(text.strip())
        return outputs


    def save_networks(self, output_dir = '../results/qwen/'):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self._distributed and not self._is_main_process:
            dist.barrier()
            return

        print("Saving model to %s" % output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self._distributed:
            dist.barrier()


def test(texts, previous_token, end_token):
    y = [txt.split(end_token)[0].split(previous_token)[-1] for txt in texts]
    return y

# if __name__ == '__main__':
#     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#     gpt = LoRaQGPTJ(adapter=True, device=device)
#     train_jsonl = f"../datasets/test/compas_train.jsonl"
#     val_jsonl = f"../datasets/test/compas_test.jsonl"
#     test_jsonl = f"../datasets/test/compas_test.jsonl"

#     train_configs={'batch_size': 4, 'epochs': 10, 'learning_rate': 1e-4, 'weight_decay': 0.01, 'warmup_steps': 6}

#     gpt.finetune(train_jsonl, val_jsonl, train_configs)
    
#     texts = "The defendant, a 69-year-old male, was arrested for a felony. The specific charge is Aggravated Assault w/Firearm. The defendant has committed 0 juvenile misdemeanors, 0 juvenile felonies, 0 other juvenile delinquencies, and 0 prior convictions for other offenses. Will this defendant reoffend in two years? ###"
#     output = gpt.generate(texts)
#     print(output)
