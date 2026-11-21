"""Pure PyTorch GRPO implementation for Countdown RLVR."""

from src.dataset import CountdownPuzzle, SYSTEM_PROMPT, build_messages, format_prompt, generate_dataset
from src.grpo import (
    build_completion_mask,
    compute_group_advantages,
    compute_grpo_loss,
    compute_per_token_logprobs,
    compute_schulman_kl,
)
from src.trainer import GRPOConfig, GRPOTrainer, attach_lora, get_compute_dtype, load_model_and_tokenizer
from src.verifier import compute_reward, extract_think_and_answer, safe_eval_countdown

__all__ = [
    "CountdownPuzzle",
    "GRPOConfig",
    "GRPOTrainer",
    "SYSTEM_PROMPT",
    "attach_lora",
    "build_completion_mask",
    "build_messages",
    "compute_group_advantages",
    "compute_grpo_loss",
    "compute_per_token_logprobs",
    "compute_reward",
    "compute_schulman_kl",
    "extract_think_and_answer",
    "format_prompt",
    "generate_dataset",
    "get_compute_dtype",
    "load_model_and_tokenizer",
    "safe_eval_countdown",
]
