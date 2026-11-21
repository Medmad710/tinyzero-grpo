"""CLI experiment runner for Countdown GRPO RLVR training, held-out eval, and KL ablation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from src.dataset import CountdownPuzzle, generate_dataset
from src.trainer import GRPOConfig, GRPOTrainer, load_model_and_tokenizer


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_qualitative_traces(
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    num_traces: int = 3,
) -> list[dict[str, Any]]:
    """Select representative before/after reasoning traces, prioritizing solved improvements."""
    after_by_id = {rec["puzzle_id"]: rec for rec in after_records}
    improved: list[dict[str, Any]] = []
    still_correct: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for before in before_records:
        pid = before["puzzle_id"]
        after = after_by_id.get(pid)
        if after is None:
            continue

        entry = {
            "puzzle_id": pid,
            "numbers": before["numbers"],
            "target": before["target"],
            "reference_solution": before["reference_solution"],
            "before": {
                "completion": before["completion"],
                "reward": before["reward"],
                "is_correct": before["is_correct"],
                "has_valid_format": before["has_valid_format"],
                "completion_len": before["completion_len"],
            },
            "after": {
                "completion": after["completion"],
                "reward": after["reward"],
                "is_correct": after["is_correct"],
                "has_valid_format": after["has_valid_format"],
                "completion_len": after["completion_len"],
            },
        }
        if not before["is_correct"] and after["is_correct"]:
            improved.append(entry)
        elif after["is_correct"]:
            still_correct.append(entry)
        else:
            other.append(entry)

    combined = improved + still_correct + other
    return combined[:num_traces]


def _run_experiment(
    name: str,
    config: GRPOConfig,
    train_puzzles: list[CountdownPuzzle],
    test_puzzles: list[CountdownPuzzle],
    eval_steps: set[int],
    device: torch.device,
    r: int = 16,
    lora_alpha: int = 32,
) -> dict[str, Any]:
    """Load fresh model + LoRA, run GRPO training loop + periodic held-out evaluation."""
    set_seed(config.seed)
    t_start = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model_name,
        device=device,
        r=r,
        lora_alpha=lora_alpha,
        enable_gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, config=config, device=device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    before_records: list[dict[str, Any]] = []
    after_records: list[dict[str, Any]] = []

    if 0 in eval_steps:
        summary_0, records_0 = trainer.evaluate(test_puzzles)
        before_records = records_0
        eval_history.append({"step": 0, **summary_0})
        print(
            f"[{name}] step=0 | pass@1={summary_0['pass_at_1']:.3f} | "
            f"format={summary_0['format_rate']:.3f} | reward={summary_0['mean_reward']:.3f} | "
            f"len={summary_0['mean_completion_len']:.1f}"
        )

    rng = random.Random(config.seed)
    puzzle_order = list(train_puzzles)
    rng.shuffle(puzzle_order)

    for step in range(1, config.num_steps + 1):
        puzzle = puzzle_order[(step - 1) % len(puzzle_order)]
        step_metrics = trainer.train_step(puzzle)
        step_metrics["step"] = step
        train_history.append(step_metrics)

        if step == 1 or step % 10 == 0 or step in eval_steps:
            print(
                f"[{name}] step={step}/{config.num_steps} | "
                f"reward={step_metrics['mean_reward']:.3f} | "
                f"acc={step_metrics['accuracy']:.2f} | "
                f"fmt={step_metrics['format_rate']:.2f} | "
                f"kl={step_metrics['kl']:.4f} | "
                f"len={step_metrics['mean_completion_len']:.1f}"
                + (" (skipped: zero-var)" if step_metrics["skipped"] else "")
            )

        if step in eval_steps:
            summary, records = trainer.evaluate(test_puzzles)
            after_records = records
            eval_history.append({"step": step, **summary})
            print(
                f"[{name}] EVAL step={step} | pass@1={summary['pass_at_1']:.3f} | "
                f"format={summary['format_rate']:.3f} | reward={summary['mean_reward']:.3f} | "
                f"len={summary['mean_completion_len']:.1f}"
            )

    runtime_sec = time.time() - t_start
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if device.type == "cuda"
        else 0.0
    )

    traces = _select_qualitative_traces(before_records, after_records, num_traces=3)

    result = {
        "name": name,
        "config": {
            "model_name": config.model_name,
            "group_size": config.group_size,
            "micro_batch_size": config.micro_batch_size,
            "num_steps": config.num_steps,
            "num_grpo_epochs": config.num_grpo_epochs,
            "lr": config.lr,
            "clip_eps": config.clip_eps,
            "beta_kl": config.beta_kl,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "seed": config.seed,
            "lora_r": r,
            "lora_alpha": lora_alpha,
        },
        "system": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "compute_dtype": str(trainer.compute_dtype),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "peak_vram_gb": round(peak_vram_gb, 3),
            "runtime_sec": round(runtime_sec, 2),
        },
        "train_history": train_history,
        "eval_history": eval_history,
        "qualitative_traces": traces,
    }

    del trainer, model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure-PyTorch GRPO on Countdown Numbers Game.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-train", type=int, default=500)
    parser.add_argument("--num-test", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ablation-steps", type=int, default=40)
    parser.add_argument("--beta-kl", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GRPO on device={device}")

    train_puzzles, test_puzzles = generate_dataset(
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed,
    )
    print(f"Generated {len(train_puzzles)} train and {len(test_puzzles)} held-out test puzzles.")

    main_eval_steps = {0, 25, 40, 50, 75, args.steps}
    main_eval_steps = {s for s in main_eval_steps if s <= args.steps}

    main_config = GRPOConfig(
        model_name=args.model_name,
        group_size=args.group_size,
        micro_batch_size=args.micro_batch_size,
        num_steps=args.steps,
        lr=args.lr,
        beta_kl=args.beta_kl,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )
    main_run = _run_experiment(
        name=f"grpo_kl_{args.beta_kl}",
        config=main_config,
        train_puzzles=train_puzzles,
        test_puzzles=test_puzzles,
        eval_steps=main_eval_steps,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "dataset": {
            "num_train": len(train_puzzles),
            "num_test": len(test_puzzles),
            "seed": args.seed,
        },
        "main_run": main_run,
        "ablation": None,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.skip_ablation and args.ablation_steps > 0:
        ablation_eval_steps = {0, 25, args.ablation_steps}
        ablation_eval_steps = {s for s in ablation_eval_steps if s <= args.ablation_steps}

        no_kl_config = GRPOConfig(
            model_name=args.model_name,
            group_size=args.group_size,
            micro_batch_size=args.micro_batch_size,
            num_steps=args.ablation_steps,
            lr=args.lr,
            beta_kl=0.0,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
        )
        no_kl_run = _run_experiment(
            name="grpo_kl_0.0",
            config=no_kl_config,
            train_puzzles=train_puzzles,
            test_puzzles=test_puzzles,
            eval_steps=ablation_eval_steps,
            device=device,
        )

        # Slice the matching first `ablation_steps` of the main run for apples-to-apples plotting.
        with_kl_slice = {
            "name": main_run["name"],
            "beta_kl": args.beta_kl,
            "train_history": [
                m for m in main_run["train_history"] if m["step"] <= args.ablation_steps
            ],
            "eval_history": [
                m for m in main_run["eval_history"] if m["step"] <= args.ablation_steps
            ],
        }
        payload["ablation"] = {
            "horizon_steps": args.ablation_steps,
            "with_kl": with_kl_slice,
            "without_kl": {
                "name": no_kl_run["name"],
                "beta_kl": 0.0,
                "train_history": no_kl_run["train_history"],
                "eval_history": no_kl_run["eval_history"],
                "system": no_kl_run["system"],
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved experiment results to {output_path}")


if __name__ == "__main__":
    main()
