"""Pure PyTorch Group Relative Policy Optimization (GRPO) math primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_completion_mask(
    sequences: torch.Tensor,
    prompt_len: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Build boolean mask over shifted next-token positions `[B, L - 1]`.

    Strictly masks out:
    - All prompt tokens (indices `< prompt_len - 1` in `sequences[:, 1:]`).
    - All tokens strictly after the first `eos_token_id` in the completion portion.
    - Any `pad_token_id` tokens in the completion portion (when distinct from `eos_token_id`).
    """
    if sequences.ndim != 2:
        raise ValueError(f"Expected 2D sequences tensor [B, L], got shape {tuple(sequences.shape)}")

    batch_size, seq_len = sequences.shape
    if seq_len < 2:
        return torch.zeros((batch_size, 0), dtype=torch.bool, device=sequences.device)

    shifted_len = seq_len - 1
    mask = torch.zeros((batch_size, shifted_len), dtype=torch.bool, device=sequences.device)

    # Position `prompt_len - 1` in shifted targets corresponds to `sequences[:, prompt_len]`,
    # i.e., the very first generated completion token.
    comp_start = max(prompt_len - 1, 0)
    if comp_start >= shifted_len:
        return mask

    completion_targets = sequences[:, comp_start + 1 :]
    comp_mask = torch.ones_like(completion_targets, dtype=torch.bool)

    if eos_token_id is not None:
        is_eos = completion_targets == eos_token_id
        # Include the first EOS token itself (so the model learns to stop), exclude everything after.
        eos_cumsum = is_eos.cumsum(dim=-1)
        after_first_eos = (eos_cumsum - is_eos.to(eos_cumsum.dtype)) > 0
        comp_mask = comp_mask & (~after_first_eos)

    if pad_token_id is not None and pad_token_id != eos_token_id:
        comp_mask = comp_mask & (completion_targets != pad_token_id)

    mask[:, comp_start:] = comp_mask
    return mask


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
    zero_var_threshold: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Compute group-relative normalized advantages `(r - mean) / (std + eps)`.

    Returns:
        `(advantages, is_zero_variance)` where `is_zero_variance` is True when `std < zero_var_threshold`,
        signaling the caller to skip the gradient update gracefully.
    """
    rewards_f32 = rewards.to(dtype=torch.float32)
    if rewards_f32.numel() <= 1:
        return torch.zeros_like(rewards_f32), True

    std = rewards_f32.std(unbiased=False)
    if float(std.item()) < zero_var_threshold:
        return torch.zeros_like(rewards_f32), True

    mean = rewards_f32.mean()
    advantages = (rewards_f32 - mean) / (std + eps)
    return advantages, False


def compute_per_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token log-probabilities in float32 to prevent FP16 softmax underflow/NaNs."""
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape[:2])} does not match input_ids {tuple(input_ids.shape)}"
        )
    if input_ids.shape[1] < 2:
        return torch.zeros((input_ids.shape[0], 0), dtype=torch.float32, device=logits.device)

    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs


def compute_schulman_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    clamp_val: float = 10.0,
) -> torch.Tensor:
    """DeepSeekMath / Schulman unbiased non-negative per-token KL estimator (k3).

    `D_KL(pi_theta || pi_ref) = exp(log_ref - log_policy) - (log_ref - log_policy) - 1`
    Clamped to `[-10.0, 10.0]` in float32 by default to prevent FP16 GradScaler backward overflow.
    """
    log_ratio = torch.clamp(
        ref_logprobs.float() - policy_logprobs.float(),
        min=-clamp_val,
        max=clamp_val,
    )
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    # Clamp tiny negative FP32 rounding noise near 0 while preserving exact 0 when log_ratio == 0.
    return torch.clamp(kl, min=0.0)


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute clipped GRPO surrogate loss + unbiased Schulman KL penalty over completion tokens."""
    mask_f32 = completion_mask.to(dtype=torch.float32)
    adv = advantages.to(dtype=torch.float32, device=policy_logprobs.device).view(-1, 1)

    log_diff = torch.clamp(
        policy_logprobs.float() - old_logprobs.float(),
        min=-10.0,
        max=10.0,
    )
    ratio = torch.exp(log_diff)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    pg_loss_per_token = -torch.minimum(surr1, surr2)

    kl_per_token = compute_schulman_kl(policy_logprobs, ref_logprobs)
    total_per_token = pg_loss_per_token + beta_kl * kl_per_token

    seq_token_counts = mask_f32.sum(dim=-1)
    valid_seqs = seq_token_counts > 0

    if not bool(valid_seqs.any().item()):
        zero = (policy_logprobs.float() * 0.0).sum()
        return zero, {
            "loss": 0.0,
            "pg_loss": 0.0,
            "kl": 0.0,
            "clip_frac": 0.0,
            "mean_ratio": 1.0,
        }

    safe_counts = seq_token_counts.clamp(min=1.0)
    per_seq_total = (total_per_token * mask_f32).sum(dim=-1) / safe_counts
    per_seq_pg = (pg_loss_per_token * mask_f32).sum(dim=-1) / safe_counts
    per_seq_kl = (kl_per_token * mask_f32).sum(dim=-1) / safe_counts

    loss = per_seq_total[valid_seqs].mean()

    with torch.no_grad():
        total_valid_tokens = mask_f32.sum().clamp(min=1.0)
        is_clipped = ((ratio - 1.0).abs() > clip_eps).to(dtype=torch.float32)
        clip_frac = float(((is_clipped * mask_f32).sum() / total_valid_tokens).item())
        mean_ratio = float(((ratio * mask_f32).sum() / total_valid_tokens).item())
        metrics = {
            "loss": float(loss.detach().item()),
            "pg_loss": float(per_seq_pg[valid_seqs].mean().item()),
            "kl": float(per_seq_kl[valid_seqs].mean().item()),
            "clip_frac": clip_frac,
            "mean_ratio": mean_ratio,
        }

    return loss, metrics
