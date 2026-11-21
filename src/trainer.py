"""Memory-safe pure PyTorch GRPO trainer for Tesla T4 (16 GB VRAM) and CPU testing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import CountdownPuzzle, format_prompt
from src.grpo import (
    build_completion_mask,
    compute_group_advantages,
    compute_grpo_loss,
    compute_per_token_logprobs,
)
from src.verifier import compute_reward


DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class GRPOConfig:
    """Configuration for GRPO RLVR training and evaluation."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    group_size: int = 4
    micro_batch_size: int = 2
    num_steps: int = 100
    num_grpo_epochs: int = 2
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    max_new_tokens: int = 224
    temperature: float = 0.8
    top_p: float = 0.95
    eval_batch_size: int = 12
    seed: int = 42


def get_compute_dtype(device: torch.device) -> torch.dtype:
    """Select hardware-aware precision: FP16 on Turing/Volta (major < 8), BF16 on Ampere+."""
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def attach_lora(
    base_model: torch.nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGET_MODULES,
    enable_gradient_checkpointing: bool = True,
) -> torch.nn.Module:
    """Attach PEFT LoRA adapters, keep trainable LoRA weights in FP32, and enable GC."""
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    # Keep trainable LoRA weights in FP32 so AdamW updates never underflow under FP16 training.
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch.float32)

    if enable_gradient_checkpointing:
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        if hasattr(peft_model, "config"):
            peft_model.config.use_cache = False

    return peft_model


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: torch.device | None = None,
    r: int = 16,
    lora_alpha: int = 32,
    enable_gradient_checkpointing: bool = True,
) -> tuple[torch.nn.Module, Any]:
    """Load base CausalLM in hardware-aware dtype and attach FP32 LoRA adapters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compute_dtype = get_compute_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=compute_dtype)
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=compute_dtype)

    base_model.to(device)
    model = attach_lora(
        base_model,
        r=r,
        lora_alpha=lora_alpha,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
    )
    return model, tokenizer


def _build_sequence_attention_mask(
    sequences: torch.Tensor,
    prompt_len: int,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Create `[B, L]` attention mask keeping prompt + completion through first EOS active."""
    batch_size, seq_len = sequences.shape
    attn = torch.ones((batch_size, seq_len), dtype=torch.long, device=sequences.device)
    if seq_len <= prompt_len:
        return attn

    comp = sequences[:, prompt_len:]
    active = torch.ones_like(comp, dtype=torch.bool)
    if eos_token_id is not None:
        is_eos = comp == eos_token_id
        eos_cumsum = is_eos.cumsum(dim=-1)
        after_first_eos = (eos_cumsum - is_eos.to(eos_cumsum.dtype)) > 0
        active = active & (~after_first_eos)
    if pad_token_id is not None and pad_token_id != eos_token_id:
        active = active & (comp != pad_token_id)

    attn[:, prompt_len:] = active.to(dtype=torch.long)
    return attn


def _count_completion_tokens(
    completion_ids: torch.Tensor,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> int:
    """Count generated tokens up to and including the first EOS token (excluding trailing pad)."""
    tokens = completion_ids.tolist()
    count = 0
    for tok in tokens:
        if pad_token_id is not None and pad_token_id != eos_token_id and tok == pad_token_id:
            break
        count += 1
        if eos_token_id is not None and tok == eos_token_id:
            break
    return count


class GRPOTrainer:
    """Pure PyTorch GRPO trainer with zero-VRAM `disable_adapter()` reference policy."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: GRPOConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.compute_dtype = get_compute_dtype(device)
        self.use_autocast = device.type == "cuda" and self.compute_dtype in (
            torch.float16,
            torch.bfloat16,
        )

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not self.trainable_params:
            raise ValueError("Model has no trainable parameters.")

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        total_opt_steps = max(config.num_steps * config.num_grpo_epochs, 1)
        warmup_steps = max(int(total_opt_steps * config.warmup_ratio), 1)

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(max(total_opt_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and self.compute_dtype == torch.float16),
        )

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def pad_token_id(self) -> int | None:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        return pad_id if pad_id is not None else self.eos_token_id

    @torch.no_grad()
    def _compute_logprobs_microbatched(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        disable_adapter: bool = False,
    ) -> torch.Tensor:
        """Compute per-token log-probs in micro-batches (and optionally with adapters disabled)."""
        prev_use_cache = getattr(self.model.config, "use_cache", False)
        self.model.eval()
        self.model.config.use_cache = False
        batch_size, seq_len = sequences.shape
        if seq_len < 2:
            self.model.config.use_cache = prev_use_cache
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=self.device)

        chunks: list[torch.Tensor] = []
        mb_size = max(1, self.config.micro_batch_size)

        try:
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_seq = sequences[start:end]
                mb_attn = attention_mask[start:end]

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.compute_dtype,
                    enabled=self.use_autocast,
                ):
                    if disable_adapter:
                        with self.model.disable_adapter():
                            outputs = self.model(input_ids=mb_seq, attention_mask=mb_attn)
                    else:
                        outputs = self.model(input_ids=mb_seq, attention_mask=mb_attn)

                logprobs = compute_per_token_logprobs(outputs.logits, mb_seq)
                chunks.append(logprobs)
        finally:
            self.model.config.use_cache = prev_use_cache

        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def generate_rollouts(
        self,
        puzzle: CountdownPuzzle,
    ) -> tuple[torch.Tensor, torch.Tensor, int, list[str], list[int]]:
        """Sample a group of `G` completions for a single Countdown puzzle."""
        prev_training = self.model.training
        prev_use_cache = getattr(self.model.config, "use_cache", False)
        prev_padding_side = getattr(self.tokenizer, "padding_side", "left")

        self.tokenizer.padding_side = "left"
        prompt_text = format_prompt(puzzle, self.tokenizer)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(self.device)
        prompt_attn = encoded["attention_mask"].to(self.device)
        prompt_len = int(prompt_ids.shape[1])

        group_size = self.config.group_size
        batch_ids = prompt_ids.repeat(group_size, 1)
        batch_attn = prompt_attn.repeat(group_size, 1)

        self.model.eval()
        self.model.config.use_cache = True
        try:
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
                enabled=self.use_autocast,
            ):
                sequences = self.model.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_attn,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    use_cache=True,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                )
        finally:
            self.tokenizer.padding_side = prev_padding_side
            self.model.config.use_cache = prev_use_cache
            if prev_training:
                self.model.train()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        seq_attention_mask = _build_sequence_attention_mask(
            sequences=sequences,
            prompt_len=prompt_len,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        completion_ids = sequences[:, prompt_len:]
        completion_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        completion_lengths = [
            _count_completion_tokens(completion_ids[i], self.eos_token_id, self.pad_token_id)
            for i in range(group_size)
        ]

        return sequences, seq_attention_mask, prompt_len, completion_texts, completion_lengths

    def grpo_update_from_rollouts(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        rewards: Sequence[float] | torch.Tensor,
    ) -> dict[str, float | bool]:
        """Run micro-batched GRPO optimization steps on pre-collected rollouts."""
        if not isinstance(rewards, torch.Tensor):
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards_tensor = rewards.to(dtype=torch.float32, device=self.device)

        advantages, is_zero_var = compute_group_advantages(rewards_tensor)
        if is_zero_var:
            return {
                "loss": 0.0,
                "pg_loss": 0.0,
                "kl": 0.0,
                "clip_frac": 0.0,
                "mean_ratio": 1.0,
                "skipped": True,
            }

        completion_mask = build_completion_mask(
            sequences=sequences,
            prompt_len=prompt_len,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        if int(completion_mask.sum().item()) == 0:
            return {
                "loss": 0.0,
                "pg_loss": 0.0,
                "kl": 0.0,
                "clip_frac": 0.0,
                "mean_ratio": 1.0,
                "skipped": True,
            }

        # Reference log-probs use zero extra VRAM via `model.disable_adapter()`.
        old_logprobs = self._compute_logprobs_microbatched(
            sequences, attention_mask, disable_adapter=False
        )
        ref_logprobs = self._compute_logprobs_microbatched(
            sequences, attention_mask, disable_adapter=True
        )

        group_size = int(sequences.shape[0])
        mb_size = max(1, min(self.config.micro_batch_size, group_size))

        prev_use_cache = getattr(self.model.config, "use_cache", False)
        self.model.config.use_cache = False
        self.model.train()

        agg_metrics = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "mean_ratio": 0.0}
        try:
            for _ in range(self.config.num_grpo_epochs):
                self.optimizer.zero_grad(set_to_none=True)
                epoch_metrics = {k: 0.0 for k in agg_metrics}

                for start in range(0, group_size, mb_size):
                    end = min(start + mb_size, group_size)
                    mb_weight = float(end - start) / float(group_size)

                    mb_seq = sequences[start:end]
                    mb_attn = attention_mask[start:end]
                    mb_mask = completion_mask[start:end]
                    mb_old = old_logprobs[start:end]
                    mb_ref = ref_logprobs[start:end]
                    mb_adv = advantages[start:end]

                    # Trim trailing padding columns per micro-batch to lower peak T4 VRAM.
                    max_active = min(
                        max(int(mb_attn.sum(dim=1).max().item()), prompt_len, 2),
                        int(mb_seq.shape[1]),
                    )
                    mb_seq = mb_seq[:, :max_active]
                    mb_attn = mb_attn[:, :max_active]
                    mb_mask = mb_mask[:, : max_active - 1]
                    mb_old = mb_old[:, : max_active - 1]
                    mb_ref = mb_ref[:, : max_active - 1]

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.compute_dtype,
                        enabled=self.use_autocast,
                    ):
                        outputs = self.model(input_ids=mb_seq, attention_mask=mb_attn)
                    policy_logprobs = compute_per_token_logprobs(outputs.logits, mb_seq)

                    mb_loss, mb_metrics = compute_grpo_loss(
                        policy_logprobs=policy_logprobs,
                        old_logprobs=mb_old,
                        ref_logprobs=mb_ref,
                        advantages=mb_adv,
                        completion_mask=mb_mask,
                        clip_eps=self.config.clip_eps,
                        beta_kl=self.config.beta_kl,
                    )
                    scaled_loss = mb_loss * mb_weight
                    self.scaler.scale(scaled_loss).backward()

                    for key in epoch_metrics:
                        epoch_metrics[key] += mb_metrics[key] * mb_weight

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.max_grad_norm)
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= prev_scale:
                    self.scheduler.step()

                for key in agg_metrics:
                    agg_metrics[key] += epoch_metrics[key]
        finally:
            self.model.config.use_cache = prev_use_cache

        num_epochs = float(max(self.config.num_grpo_epochs, 1))
        result: dict[str, float | bool] = {k: v / num_epochs for k, v in agg_metrics.items()}
        result["skipped"] = False
        return result

    def train_step(self, puzzle: CountdownPuzzle) -> dict[str, Any]:
        """Execute one full RLVR step: rollout -> AST verify -> micro-batched GRPO update."""
        sequences, attention_mask, prompt_len, completion_texts, completion_lengths = (
            self.generate_rollouts(puzzle)
        )

        rewards: list[float] = []
        correct_flags: list[bool] = []
        format_flags: list[bool] = []
        for text in completion_texts:
            r, is_corr, has_fmt = compute_reward(
                text, numbers=puzzle.numbers, target=puzzle.target
            )
            rewards.append(r)
            correct_flags.append(is_corr)
            format_flags.append(has_fmt)

        update_metrics = self.grpo_update_from_rollouts(
            sequences=sequences,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            rewards=rewards,
        )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        n = float(len(rewards))
        return {
            "puzzle_id": puzzle.puzzle_id,
            "mean_reward": sum(rewards) / n,
            "reward_std": float(torch.tensor(rewards, dtype=torch.float32).std(unbiased=False).item()),
            "accuracy": sum(1 for c in correct_flags if c) / n,
            "format_rate": sum(1 for f in format_flags if f) / n,
            "mean_completion_len": sum(completion_lengths) / n,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            **update_metrics,
        }

    @torch.no_grad()
    def evaluate(
        self,
        puzzles: Sequence[CountdownPuzzle],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate held-out pass@1 (greedy decoding) on a list of Countdown puzzles."""
        if not puzzles:
            return {
                "pass_at_1": 0.0,
                "format_rate": 0.0,
                "mean_reward": 0.0,
                "mean_completion_len": 0.0,
                "num_puzzles": 0,
            }, []

        prev_training = self.model.training
        prev_use_cache = getattr(self.model.config, "use_cache", False)
        prev_padding_side = getattr(self.tokenizer, "padding_side", "right")

        self.model.eval()
        self.model.config.use_cache = True
        self.tokenizer.padding_side = "left"

        records: list[dict[str, Any]] = []
        batch_size = max(1, self.config.eval_batch_size)

        try:
            for start in range(0, len(puzzles), batch_size):
                batch_puzzles = puzzles[start : start + batch_size]
                prompts = [format_prompt(p, self.tokenizer) for p in batch_puzzles]
                encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                padded_prompt_len = int(input_ids.shape[1])

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.compute_dtype,
                    enabled=self.use_autocast,
                ):
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                    )

                gen_tokens = outputs[:, padded_prompt_len:]
                decoded = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

                for i, puzzle in enumerate(batch_puzzles):
                    text = decoded[i]
                    reward, is_correct, has_format = compute_reward(
                        text, numbers=puzzle.numbers, target=puzzle.target
                    )
                    comp_len = _count_completion_tokens(
                        gen_tokens[i], self.eos_token_id, self.pad_token_id
                    )
                    records.append(
                        {
                            "puzzle_id": puzzle.puzzle_id,
                            "numbers": list(puzzle.numbers),
                            "target": puzzle.target,
                            "reference_solution": puzzle.solution,
                            "completion": text,
                            "reward": reward,
                            "is_correct": is_correct,
                            "has_valid_format": has_format,
                            "completion_len": comp_len,
                        }
                    )
        finally:
            self.tokenizer.padding_side = prev_padding_side
            self.model.config.use_cache = prev_use_cache
            if prev_training:
                self.model.train()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        total = float(len(records))
        summary = {
            "pass_at_1": sum(1 for r in records if r["is_correct"]) / total,
            "format_rate": sum(1 for r in records if r["has_valid_format"]) / total,
            "mean_reward": sum(r["reward"] for r in records) / total,
            "mean_completion_len": sum(r["completion_len"] for r in records) / total,
            "num_puzzles": len(records),
        }
        return summary, records
