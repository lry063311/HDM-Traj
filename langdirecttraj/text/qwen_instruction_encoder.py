from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB, MobilityTokenVocab


def parse_torch_dtype(name: str):
    name = str(name).lower()
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_qwen_tokenizer(
    model_name: str,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Qwen planner requires transformers. Install transformers first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def batch_encode_qwen_texts(
    tokenizer,
    texts: Sequence[str],
    max_len: int = 128,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_len),
        return_tensors="pt",
    )
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded


@dataclass
class QwenPlannerConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    plan_dim: int = 256
    num_plan_tokens: int = 8
    num_attention_heads: int = 4
    dropout: float = 0.1
    pooling: str = "last"
    freeze_backbone: bool = True
    torch_dtype: str = "auto"
    local_files_only: bool = False
    trust_remote_code: bool = True
    gradient_checkpointing: bool = False
    qwen_lora_enabled: bool = False
    qwen_lora_r: int = 8
    qwen_lora_alpha: int = 16
    qwen_lora_dropout: float = 0.05
    qwen_lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    semantic_control_enabled: bool = False
    field_control_fields: tuple[str, ...] = ("direction", "curvature", "reference_adherence")
    field_control_latent_dim: int = 128
    field_control_num_act_classes: int = 8

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "plan_dim": self.plan_dim,
            "num_plan_tokens": self.num_plan_tokens,
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "freeze_backbone": self.freeze_backbone,
            "torch_dtype": self.torch_dtype,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "gradient_checkpointing": self.gradient_checkpointing,
            "qwen_lora_enabled": self.qwen_lora_enabled,
            "qwen_lora_r": self.qwen_lora_r,
            "qwen_lora_alpha": self.qwen_lora_alpha,
            "qwen_lora_dropout": self.qwen_lora_dropout,
            "qwen_lora_target_modules": list(self.qwen_lora_target_modules),
            "semantic_control_enabled": self.semantic_control_enabled,
            "field_control_fields": list(self.field_control_fields),
            "field_control_latent_dim": self.field_control_latent_dim,
            "field_control_num_act_classes": self.field_control_num_act_classes,
        }


class QwenInstructionEncoder(nn.Module):
    """Frozen Qwen backbone with trainable semantic-plan projections."""

    def __init__(self, config: QwenPlannerConfig):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Qwen planner requires transformers. Install transformers first.") from exc

        self.config = config
        dtype = parse_torch_dtype(config.torch_dtype)
        backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        if bool(config.qwen_lora_enabled):
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Qwen LoRA requires peft. Install peft first.") from exc
            lora_cfg = LoraConfig(
                r=int(config.qwen_lora_r),
                lora_alpha=int(config.qwen_lora_alpha),
                lora_dropout=float(config.qwen_lora_dropout),
                target_modules=list(config.qwen_lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            backbone = get_peft_model(backbone, lora_cfg)
        self.backbone = backbone
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.pooling = str(config.pooling).lower()

        if config.gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if config.freeze_backbone:
            self.backbone.eval()
            for name, param in self.backbone.named_parameters():
                if bool(config.qwen_lora_enabled) and "lora_" in name:
                    param.requires_grad = True
                    continue
                param.requires_grad = False

        self.pool_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, config.plan_dim),
            nn.SiLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(config.plan_dim, config.plan_dim),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, config.plan_dim),
        )
        self.plan_queries = nn.Parameter(torch.randn(config.num_plan_tokens, config.plan_dim) * 0.02)
        self.plan_attention = nn.MultiheadAttention(
            embed_dim=config.plan_dim,
            num_heads=config.num_attention_heads,
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.plan_norm = nn.LayerNorm(config.plan_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        has_trainable_backbone = any(param.requires_grad for param in self.backbone.parameters())
        if self.config.freeze_backbone and not has_trainable_backbone:
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        return self

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.bool()
        if self.pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
            return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
        if self.pooling != "last":
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(dim=1, index=gather_index).squeeze(1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        qwen_body = getattr(self.backbone, "model", self.backbone)
        requires_grad = any(param.requires_grad for param in qwen_body.parameters())
        with torch.set_grad_enabled(requires_grad):
            outputs = qwen_body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if not hidden_states:
                    raise AttributeError("Qwen planner backbone output has neither last_hidden_state nor hidden_states.")
                hidden = hidden_states[-1]

        hidden = hidden.to(self.plan_queries.dtype)
        pooled = self.pool_proj(self._pool_hidden(hidden, attention_mask))
        token_features = self.token_proj(hidden)
        queries = self.plan_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        semantic_plan_tokens, _ = self.plan_attention(
            query=queries,
            key=token_features,
            value=token_features,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        semantic_plan_tokens = self.plan_norm(semantic_plan_tokens + queries)
        return {
            "pooled": pooled,
            "semantic_plan_tokens": semantic_plan_tokens,
            "token_hidden": token_features,
        }


class QwenMobilityTokenPlanner(nn.Module):
    """Qwen semantic planner with five trainable mobility-token heads."""

    def __init__(
        self,
        config: QwenPlannerConfig,
        mobility_vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB,
    ):
        super().__init__()
        self.mobility_vocab = mobility_vocab
        self.fields = list(mobility_vocab.fields)
        self.semantic_control_enabled = bool(config.semantic_control_enabled)
        self.field_control_fields = [field for field in config.field_control_fields if field in mobility_vocab.fields]
        self.field_control_latent_dim = int(config.field_control_latent_dim)
        self.field_control_num_act_classes = int(config.field_control_num_act_classes)
        self.encoder = QwenInstructionEncoder(config)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(config.plan_dim, mobility_vocab.sizes[field]) for field in self.fields}
        )
        self.global_semantic_head = nn.Sequential(
            nn.LayerNorm(config.plan_dim),
            nn.Linear(config.plan_dim, config.plan_dim),
            nn.SiLU(),
            nn.Linear(config.plan_dim, config.plan_dim),
        )
        self.field_control_latent_heads = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.LayerNorm(config.plan_dim),
                    nn.Linear(config.plan_dim, self.field_control_latent_dim),
                    nn.SiLU(),
                    nn.Linear(self.field_control_latent_dim, self.field_control_latent_dim),
                )
                for field in self.field_control_fields
            }
        )
        self.field_control_logit_heads = nn.ModuleDict(
            {
                field: nn.Linear(config.plan_dim, self.field_control_num_act_classes)
                for field in self.field_control_fields
            }
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Dict | torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = {field: self.heads[field](encoded["pooled"]) for field in self.fields}
        global_semantic_latent = self.global_semantic_head(encoded["pooled"])
        field_control_latents = {
            field: self.field_control_latent_heads[field](encoded["pooled"])
            for field in self.field_control_fields
        }
        field_control_logits = {
            field: self.field_control_logit_heads[field](encoded["pooled"])
            for field in self.field_control_fields
        }
        return {
            "logits": logits,
            "pooled": encoded["pooled"],
            "semantic_plan_tokens": encoded["semantic_plan_tokens"],
            "global_semantic_latent": global_semantic_latent,
            "field_control_latents": field_control_latents,
            "field_control_logits": field_control_logits,
            "semantic_control_enabled": self.semantic_control_enabled,
            "field_control_fields": list(self.field_control_fields),
        }

    def predict_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        return torch.stack([logits[field].argmax(dim=-1) for field in self.fields], dim=1)

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]
