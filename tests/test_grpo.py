"""Fast CPU unit and end-to-end integration tests for tinyzero-grpo (<10s runtime)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.plot_results import generate_all_plots
from src.dataset import CountdownPuzzle, format_prompt, generate_dataset
from src.grpo import (
    build_completion_mask,
    compute_group_advantages,
    compute_grpo_loss,
    compute_per_token_logprobs,
    compute_schulman_kl,
)
from src.trainer import GRPOConfig, GRPOTrainer, attach_lora, get_compute_dtype
from src.verifier import compute_reward, extract_think_and_answer, safe_eval_countdown


class _TinyDummyTokenizer:
    """Minimal byte-level tokenizer for fast CPU testing without network access."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<eos>"
        self.padding_side = "right"
        self.chat_template = None

    def _encode_one(self, text: str) -> list[int]:
        # Map bytes into [2, vocab_size - 1] so token IDs 0 and 1 remain special/EOS/PAD.
        span = self.vocab_size - 2
        ids = [2 + (b % span) for b in text.encode("utf-8")]
        return ids[:48] if len(ids) > 48 else (ids or [2])

    def __call__(
        self,
        texts: str | list[str],
        return_tensors: str = "pt",
        padding: bool = False,
    ) -> dict[str, torch.Tensor]:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        encoded = [self._encode_one(t) for t in items]
        max_len = max(len(x) for x in encoded) if padding else len(encoded[0])

        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for ids in encoded:
            pad_n = max_len - len(ids)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_n + ids)
                masks.append([0] * pad_n + [1] * len(ids))
            else:
                padded.append(ids + [self.pad_token_id] * pad_n)
                masks.append([1] * len(ids) + [0] * pad_n)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        out: list[str] = []
        for row in token_ids.tolist():
            if skip_special_tokens:
                row = [t for t in row if t not in (self.eos_token_id, self.pad_token_id)]
            out.append(" ".join(str(t) for t in row))
        return out


# ---------------------------------------------------------------------------
# 1. Dataset Tests
# ---------------------------------------------------------------------------

def test_dataset_deterministic_and_disjoint() -> None:
    train_a, test_a = generate_dataset(num_train=500, num_test=60, seed=42)
    train_b, test_b = generate_dataset(num_train=500, num_test=60, seed=42)

    assert len(train_a) == 500
    assert len(test_a) == 60
    assert train_a == train_b
    assert test_a == test_b

    train_keys = {(tuple(sorted(p.numbers)), p.target) for p in train_a}
    test_keys = {(tuple(sorted(p.numbers)), p.target) for p in test_a}
    assert train_keys.isdisjoint(test_keys)

    for puzzle in train_a + test_a:
        assert len(puzzle.numbers) == 3
        assert all(1 <= n <= 25 for n in puzzle.numbers)
        assert 10 <= puzzle.target <= 100
        ref_completion = f"<think>Check solution.</think>\n<answer>{puzzle.solution}</answer>"
        reward, is_correct, has_fmt = compute_reward(
            ref_completion, numbers=puzzle.numbers, target=puzzle.target
        )
        assert has_fmt is True
        assert is_correct is True
        assert reward == pytest.approx(1.15)


def test_format_prompt_with_and_without_chat_template() -> None:
    puzzle = CountdownPuzzle(puzzle_id=0, numbers=(4, 7, 10), target=38, solution="(4 * 7) + 10")
    plain = format_prompt(puzzle, tokenizer=None)
    assert "4, 7, 10" in plain
    assert "38" in plain

    class _FakeChatTok:
        chat_template = "dummy"

        def apply_chat_template(self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            return "|".join(m["content"] for m in messages)

    templated = format_prompt(puzzle, tokenizer=_FakeChatTok())
    assert "4, 7, 10" in templated
    assert "38" in templated


# ---------------------------------------------------------------------------
# 2. Verifier Tests
# ---------------------------------------------------------------------------

def test_verifier_valid_and_division() -> None:
    completion = "<think>4 * 7 = 28, plus 10 is 38.</think>\n<answer>(4 * 7) + 10</answer>"
    reward, is_correct, has_fmt = compute_reward(completion, numbers=[4, 7, 10], target=38)
    assert (reward, is_correct, has_fmt) == (pytest.approx(1.15), True, True)

    div_comp = "<think>24 / 6 = 4, times 5 = 20.</think><answer>(24 / 6) * 5</answer>"
    reward, is_correct, has_fmt = compute_reward(div_comp, numbers=[6, 24, 5], target=20)
    assert (reward, is_correct, has_fmt) == (pytest.approx(1.15), True, True)


def test_verifier_rejects_reused_missing_extra_numbers() -> None:
    # Reused 4 instead of 7 (4 * 4 + 10 = 26)
    r, ok, fmt = compute_reward("<think>t</think><answer>4 * 4 + 10</answer>", [4, 7, 10], 26)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Missing 10 (4 * 7 = 28)
    r, ok, fmt = compute_reward("<think>t</think><answer>4 * 7</answer>", [4, 7, 10], 28)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Extra number 1
    r, ok, fmt = compute_reward("<think>t</think><answer>4 * 7 + 10 + 1</answer>", [4, 7, 10], 39)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Duplicate numbers in puzzle must match exact multiplicity
    r, ok, fmt = compute_reward("<think>t</think><answer>(5 + 5) * 3</answer>", [5, 5, 3], 30)
    assert (r, ok, fmt) == (pytest.approx(1.15), True, True)


def test_verifier_division_by_zero_and_malformed_and_injection() -> None:
    # Division by zero
    r, ok, fmt = compute_reward("<think>t</think><answer>10 / (4 - 4)</answer>", [10, 4, 4], 10)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Malformed syntax
    r, ok, fmt = compute_reward("<think>t</think><answer>4 * + 7</answer>", [4, 7, 10], 38)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Code injection attempts (must never execute and must return False)
    injections = [
        "__import__('os').system('echo hacked')",
        "open('requirements.txt').read()",
        "(lambda: 38)()",
        "True + 37",
        "2 ** 5 + 6",
        "-4 + 42",
        "4 * 7 + 10 = 38",
    ]
    for expr in injections:
        assert safe_eval_countdown(expr, [4, 7, 10], 38) is False
        r, ok, fmt = compute_reward(f"<think>t</think><answer>{expr}</answer>", [4, 7, 10], 38)
        assert (r, ok, fmt) == (pytest.approx(0.15), False, True)


def test_verifier_format_only_and_invalid_tags() -> None:
    # Wrong arithmetic, valid format -> +0.15
    r, ok, fmt = compute_reward("<think>guess</think>\n<answer>4 + 7 + 10</answer>", [4, 7, 10], 38)
    assert (r, ok, fmt) == (pytest.approx(0.15), False, True)

    # Invalid/missing/nested tags -> 0.0
    bad_outputs = [
        "4 * 7 + 10",
        "<think>unfinished</think> 4 * 7 + 10",
        "<think>missing close <answer>4*7+10</answer>",
        "<think>ok</think><answer></answer>",
        "<think>nested <think>x</think></think><answer>4*7+10</answer>",
        "<think>ok</think><answer>4*7+10</answer> extra trailing text",
    ]
    for text in bad_outputs:
        assert extract_think_and_answer(text) == (None, None)
        assert compute_reward(text, [4, 7, 10], 38) == (0.0, False, False)


# ---------------------------------------------------------------------------
# 3. GRPO Math Tests
# ---------------------------------------------------------------------------

def test_group_advantages_exact_and_zero_variance() -> None:
    rewards = torch.tensor([0.0, 0.15, 1.15, 1.15], dtype=torch.float32)
    adv, is_zero = compute_group_advantages(rewards)
    assert is_zero is False
    expected = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    assert torch.allclose(adv, expected, atol=1e-6)
    assert adv.mean().abs().item() < 1e-6

    same = torch.tensor([0.15, 0.15, 0.15, 0.15], dtype=torch.float32)
    zero_adv, is_zero_same = compute_group_advantages(same)
    assert is_zero_same is True
    assert torch.equal(zero_adv, torch.zeros_like(same))


def test_completion_mask_and_prompt_exclusion() -> None:
    # prompt_len = 4, total len = 9 -> shifted len = 8
    # Row 0: generated [10, 11, EOS(1), PAD(1), PAD(1)] -> active shifted indices [3, 4, 5], masked [0,1,2,6,7]
    # Row 1: generated [12, 13, 14, 15, 16] (no EOS) -> active shifted indices [3, 4, 5, 6, 7]
    sequences = torch.tensor(
        [
            [5, 6, 7, 8, 10, 11, 1, 1, 1],
            [5, 6, 7, 8, 12, 13, 14, 15, 16],
        ],
        dtype=torch.long,
    )
    mask = build_completion_mask(sequences, prompt_len=4, eos_token_id=1, pad_token_id=1)
    expected = torch.tensor(
        [
            [False, False, False, True, True, True, False, False],
            [False, False, False, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)

    # Verify loss & gradients are strictly invariant to arbitrary values on masked-out tokens.
    policy_lp = torch.randn((2, 8), dtype=torch.float32, requires_grad=True)
    old_lp = policy_lp.detach().clone()
    ref_lp = policy_lp.detach().clone() - 0.1
    adv = torch.tensor([1.0, -1.0], dtype=torch.float32)

    loss_1, _ = compute_grpo_loss(policy_lp, old_lp, ref_lp, adv, mask, clip_eps=0.2, beta_kl=0.04)
    grad_1 = torch.autograd.grad(loss_1, policy_lp)[0]
    assert torch.all(grad_1[~mask] == 0.0)

    # Corrupt masked positions in old_lp and ref_lp; loss and gradients must stay identical.
    old_lp_corrupt = old_lp.clone()
    ref_lp_corrupt = ref_lp.clone()
    old_lp_corrupt[~mask] = -999.0
    ref_lp_corrupt[~mask] = 999.0

    loss_2, _ = compute_grpo_loss(
        policy_lp, old_lp_corrupt, ref_lp_corrupt, adv, mask, clip_eps=0.2, beta_kl=0.04
    )
    grad_2 = torch.autograd.grad(loss_2, policy_lp)[0]
    assert torch.allclose(loss_1, loss_2, atol=1e-7)
    assert torch.allclose(grad_1, grad_2, atol=1e-7)


def test_schulman_kl_properties_and_clipping() -> None:
    lp = torch.tensor([[-1.2, -0.4, -2.5]], dtype=torch.float32)
    kl_zero = compute_schulman_kl(lp, lp)
    assert torch.equal(kl_zero, torch.zeros_like(lp))

    ref_diff = torch.tensor([[-0.5, -1.8, -0.1]], dtype=torch.float32)
    kl_pos = compute_schulman_kl(lp, ref_diff)
    assert torch.all(kl_pos > 0.0)
    assert torch.all(torch.isfinite(kl_pos))

    # Extreme log-prob differences must stay finite and non-negative
    kl_extreme = compute_schulman_kl(
        torch.tensor([[-500.0, 500.0]]), torch.tensor([[500.0, -500.0]])
    )
    assert torch.all(torch.isfinite(kl_extreme))
    assert torch.all(kl_extreme >= 0.0)

    # Clipping check: ratio > 1 + eps with positive advantage yields zero PG gradient
    policy_lp = torch.tensor([[0.5, 0.0]], dtype=torch.float32, requires_grad=True)
    old_lp = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # exp(0.5) ~ 1.648 > 1.2
    ref_lp = old_lp.clone()
    adv = torch.tensor([1.0], dtype=torch.float32)
    mask = torch.tensor([[True, True]], dtype=torch.bool)

    loss, metrics = compute_grpo_loss(
        policy_lp, old_lp, ref_lp, adv, mask, clip_eps=0.2, beta_kl=0.0
    )
    grad = torch.autograd.grad(loss, policy_lp)[0]
    assert grad[0, 0].item() == pytest.approx(0.0, abs=1e-7)  # clipped token
    assert grad[0, 1].item() < 0.0  # unclipped token pushes logprob up
    assert metrics["clip_frac"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. End-to-End Tiny Qwen2Config LoRA + Gradient Flow Test
# ---------------------------------------------------------------------------

def test_end_to_end_tiny_qwen2_grpo_gradient_flow() -> None:
    assert get_compute_dtype(torch.device("cpu")) == torch.float32

    torch.manual_seed(0)
    tiny_cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    base_model = Qwen2ForCausalLM(tiny_cfg)
    model = attach_lora(base_model, r=4, lora_alpha=8, enable_gradient_checkpointing=True)

    base_params: list[tuple[str, torch.Tensor]] = []
    lora_params: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.dtype == torch.float32
            assert "lora_" in name
            lora_params.append((name, param.detach().clone()))
        else:
            base_params.append((name, param.detach().clone()))

    assert len(lora_params) > 0
    assert len(base_params) > 0

    tokenizer = _TinyDummyTokenizer(vocab_size=256)
    cfg = GRPOConfig(
        group_size=4,
        micro_batch_size=2,
        num_steps=4,
        num_grpo_epochs=2,
        lr=1e-2,
        beta_kl=0.04,
        max_new_tokens=8,
        eval_batch_size=2,
    )
    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, config=cfg, device=torch.device("cpu"))

    puzzle = CountdownPuzzle(puzzle_id=0, numbers=(4, 7, 10), target=38, solution="(4 * 7) + 10")
    sequences, attn_mask, prompt_len, texts, lengths = trainer.generate_rollouts(puzzle)
    assert sequences.shape[0] == 4
    assert len(texts) == 4
    assert len(lengths) == 4

    # 4a. Zero-variance group must gracefully skip update and leave weights unchanged.
    skip_metrics = trainer.grpo_update_from_rollouts(
        sequences=sequences,
        attention_mask=attn_mask,
        prompt_len=prompt_len,
        rewards=[0.15, 0.15, 0.15, 0.15],
    )
    assert skip_metrics["skipped"] is True
    for name, prev in lora_params:
        curr = dict(model.named_parameters())[name]
        assert torch.equal(curr.detach(), prev)

    # 4b. Verify explicit per-token logprobs work in float32 and disable_adapter() produces finite ref logits.
    ref_lp = trainer._compute_logprobs_microbatched(sequences, attn_mask, disable_adapter=True)
    assert ref_lp.dtype == torch.float32
    assert torch.all(torch.isfinite(ref_lp))

    # 4c. Mixed rewards must flow finite non-zero gradients into LoRA weights and keep base frozen.
    update_metrics = trainer.grpo_update_from_rollouts(
        sequences=sequences,
        attention_mask=attn_mask,
        prompt_len=prompt_len,
        rewards=[0.0, 0.15, 1.15, 1.15],
    )
    assert update_metrics["skipped"] is False
    assert torch.isfinite(torch.tensor(update_metrics["loss"]))
    assert torch.isfinite(torch.tensor(update_metrics["kl"]))

    named_after = dict(model.named_parameters())
    for name, prev in base_params:
        assert torch.equal(named_after[name].detach(), prev)

    changed_lora = 0
    for name, prev in lora_params:
        curr = named_after[name].detach()
        assert torch.all(torch.isfinite(curr))
        if not torch.equal(curr, prev):
            changed_lora += 1
    assert changed_lora > 0

    # 4d. Check raw `.grad` on a single micro-batch backward pass directly.
    model.zero_grad(set_to_none=True)
    mb_out = model(input_ids=sequences[:2], attention_mask=attn_mask[:2])
    mb_lp = compute_per_token_logprobs(mb_out.logits, sequences[:2])
    mb_mask = build_completion_mask(sequences[:2], prompt_len, eos_token_id=1, pad_token_id=1)
    mb_loss, _ = compute_grpo_loss(
        policy_logprobs=mb_lp,
        old_logprobs=mb_lp.detach(),
        ref_logprobs=ref_lp[:2],
        advantages=torch.tensor([-1.0, 1.0]),
        completion_mask=mb_mask,
    )
    mb_loss.backward()
    non_zero_grads = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))
            if float(param.grad.abs().max().item()) > 0.0:
                non_zero_grads += 1
    assert non_zero_grads > 0

    # 4e. Batched greedy evaluate smoke test.
    summary, records = trainer.evaluate([puzzle, puzzle])
    assert summary["num_puzzles"] == 2
    assert len(records) == 2


# ---------------------------------------------------------------------------
# 5. Plotting Smoke Test
# ---------------------------------------------------------------------------

def test_plot_results_smoke(tmp_path: Path) -> None:
    sample: dict[str, Any] = {
        "main_run": {
            "config": {"model_name": "Qwen2.5-1.5B-Instruct", "beta_kl": 0.04, "group_size": 4},
            "train_history": [
                {"step": s, "mean_reward": 0.2 + 0.02 * s, "format_rate": 0.8, "mean_completion_len": 110 + s, "kl": 0.01}
                for s in range(1, 11)
            ],
            "eval_history": [
                {"step": 0, "pass_at_1": 0.1, "format_rate": 0.7, "mean_completion_len": 100.0},
                {"step": 5, "pass_at_1": 0.25, "format_rate": 0.85, "mean_completion_len": 115.0},
                {"step": 10, "pass_at_1": 0.4, "format_rate": 0.95, "mean_completion_len": 122.0},
            ],
        },
        "ablation": {
            "horizon_steps": 5,
            "with_kl": {
                "beta_kl": 0.04,
                "train_history": [
                    {"step": s, "mean_reward": 0.2 + 0.03 * s, "kl": 0.01 * s} for s in range(1, 6)
                ],
                "eval_history": [{"step": 0, "pass_at_1": 0.1}, {"step": 5, "pass_at_1": 0.25}],
            },
            "without_kl": {
                "beta_kl": 0.0,
                "train_history": [
                    {"step": s, "mean_reward": 0.2 + 0.02 * s, "kl": 0.05 * s} for s in range(1, 6)
                ],
                "eval_history": [{"step": 0, "pass_at_1": 0.1}, {"step": 5, "pass_at_1": 0.20}],
            },
        },
    }
    res_path = tmp_path / "results.json"
    res_path.write_text(json.dumps(sample), encoding="utf-8")
    created = generate_all_plots(res_path, tmp_path / "assets")
    assert len(created) == 2
    for p in created:
        assert p.exists() and p.stat().st_size > 1000
