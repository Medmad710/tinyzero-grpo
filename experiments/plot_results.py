"""Publication-style plots for GRPO Countdown RLVR learning curves and KL ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: list[float], window: int = 10) -> np.ndarray:
    """Compute a centered/causal moving average for noisy per-step RL signals."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


def plot_learning_curve(main_run: dict[str, Any], output_path: Path) -> None:
    """Generate 3-panel learning curve: pass@1, reward & format rate, and reasoning length."""
    train_hist = main_run["train_history"]
    eval_hist = main_run["eval_history"]

    eval_steps = [e["step"] for e in eval_hist]
    eval_pass1 = [e["pass_at_1"] * 100.0 for e in eval_hist]
    eval_fmt = [e["format_rate"] * 100.0 for e in eval_hist]

    train_steps = [t["step"] for t in train_hist]
    train_rewards = [t["mean_reward"] for t in train_hist]
    train_fmt = [t["format_rate"] * 100.0 for t in train_hist]
    train_len = [t["mean_completion_len"] for t in train_hist]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=180)

    # Panel 1: Held-out pass@1 accuracy
    ax = axes[0]
    ax.plot(eval_steps, eval_pass1, marker="o", linewidth=2.2, color="#1f77b4", label="Held-out pass@1")
    for x, y in zip(eval_steps, eval_pass1):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
    ax.set_title("Held-Out pass@1 Accuracy (Greedy)")
    ax.set_xlabel("RL Step")
    ax.set_ylabel("pass@1 (%)")
    ax.set_ylim(0, max(eval_pass1 + [25.0]) * 1.18)
    ax.legend(loc="lower right", frameon=True)

    # Panel 2: Mean Reward & Format Compliance Rate
    ax = axes[1]
    ax.plot(train_steps, train_rewards, alpha=0.25, color="#2ca02c", linewidth=1.0)
    ax.plot(
        train_steps,
        _rolling_mean(train_rewards, window=10),
        color="#2ca02c",
        linewidth=2.2,
        label="Mean Reward (10-step MA)",
    )
    ax.set_title("Training Reward & Format Compliance")
    ax.set_xlabel("RL Step")
    ax.set_ylabel("Mean Reward", color="#2ca02c")
    ax.set_ylim(0, 1.25)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        train_steps,
        _rolling_mean(train_fmt, window=10),
        color="#ff7f0e",
        linewidth=1.8,
        linestyle="--",
        label="Train Format Rate",
    )
    ax2.plot(eval_steps, eval_fmt, marker="s", markersize=4.5, color="#d62728", linewidth=1.6, label="Eval Format Rate")
    ax2.set_ylabel("Format Compliance (%)", color="#d62728")
    ax2.set_ylim(0, 108)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8, frameon=True)

    # Panel 3: Mean Reasoning Length (tokens)
    ax = axes[2]
    ax.plot(train_steps, train_len, alpha=0.25, color="#9467bd", linewidth=1.0)
    ax.plot(
        train_steps,
        _rolling_mean(train_len, window=10),
        color="#9467bd",
        linewidth=2.2,
        label="Train Rollout Len (10-step MA)",
    )
    if eval_hist:
        eval_len = [e["mean_completion_len"] for e in eval_hist]
        ax.plot(eval_steps, eval_len, marker="^", color="#8c564b", linewidth=1.8, label="Eval Greedy Len")
    ax.set_title("Mean Completion Length")
    ax.set_xlabel("RL Step")
    ax.set_ylabel("Tokens")
    ax.legend(loc="best", fontsize=8.5, frameon=True)

    model_label = main_run.get("config", {}).get("model_name", "Qwen2.5-1.5B-Instruct")
    beta_kl = main_run.get("config", {}).get("beta_kl", 0.04)
    group_size = main_run.get("config", {}).get("group_size", 4)
    fig.suptitle(
        f"Pure-PyTorch GRPO on Countdown (3-Number) — {model_label} (G={group_size}, beta_KL={beta_kl})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: dict[str, Any], output_path: Path) -> None:
    """Generate side-by-side ablation comparing KL penalty (beta=0.04 vs beta=0.0)."""
    with_kl = ablation["with_kl"]
    without_kl = ablation["without_kl"]

    steps_w = [t["step"] for t in with_kl["train_history"]]
    kl_w = [t["kl"] for t in with_kl["train_history"]]
    rew_w = [t["mean_reward"] for t in with_kl["train_history"]]
    eval_steps_w = [e["step"] for e in with_kl["eval_history"]]
    pass_w = [e["pass_at_1"] * 100.0 for e in with_kl["eval_history"]]

    steps_wo = [t["step"] for t in without_kl["train_history"]]
    kl_wo = [t["kl"] for t in without_kl["train_history"]]
    rew_wo = [t["mean_reward"] for t in without_kl["train_history"]]
    eval_steps_wo = [e["step"] for e in without_kl["eval_history"]]
    pass_wo = [e["pass_at_1"] * 100.0 for e in without_kl["eval_history"]]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=180)

    # Panel 1: KL divergence trajectory
    ax = axes[0]
    ax.plot(steps_w, kl_w, alpha=0.25, color="#1f77b4", linewidth=1.0)
    ax.plot(
        steps_w,
        _rolling_mean(kl_w, window=8),
        color="#1f77b4",
        linewidth=2.2,
        label=f"beta_KL = {with_kl['beta_kl']} (Regularized)",
    )
    ax.plot(steps_wo, kl_wo, alpha=0.25, color="#d62728", linewidth=1.0)
    ax.plot(
        steps_wo,
        _rolling_mean(kl_wo, window=8),
        color="#d62728",
        linewidth=2.2,
        label="beta_KL = 0.0 (No KL Penalty)",
    )
    ax.set_title("Policy-to-Reference Schulman KL Divergence")
    ax.set_xlabel("RL Step")
    ax.set_ylabel("Per-Token KL Estimate")
    ax.legend(loc="upper left", frameon=True)

    # Panel 2: Reward & Held-out pass@1 comparison
    ax = axes[1]
    ax.plot(
        steps_w,
        _rolling_mean(rew_w, window=8),
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label=f"Train Reward (beta={with_kl['beta_kl']})",
    )
    ax.plot(
        steps_wo,
        _rolling_mean(rew_wo, window=8),
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="Train Reward (beta=0.0)",
    )
    ax.set_title("Reward & Held-Out pass@1 Comparison")
    ax.set_xlabel("RL Step")
    ax.set_ylabel("Mean Reward (8-step MA)")
    ax.set_ylim(0, 1.25)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        eval_steps_w,
        pass_w,
        marker="o",
        color="#1f77b4",
        linewidth=2.2,
        label=f"pass@1 % (beta={with_kl['beta_kl']})",
    )
    ax2.plot(
        eval_steps_wo,
        pass_wo,
        marker="s",
        color="#d62728",
        linewidth=2.2,
        label="pass@1 % (beta=0.0)",
    )
    ax2.set_ylabel("Held-Out pass@1 (%)")
    ax2.set_ylim(0, max(pass_w + pass_wo + [25.0]) * 1.18)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8, frameon=True)

    horizon = ablation.get("horizon_steps", len(steps_wo))
    fig.suptitle(
        f"KL Regularization Ablation ({horizon} RL Steps): Schulman KL Penalty vs Unconstrained Policy",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results_path: Path, assets_dir: Path) -> list[Path]:
    """Read `results.json` and save publication-grade figures into `assets_dir`."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    created: list[Path] = []

    if data.get("main_run"):
        lc_path = assets_dir / "grpo_learning_curve.png"
        plot_learning_curve(data["main_run"], lc_path)
        created.append(lc_path)

    if data.get("ablation"):
        ab_path = assets_dir / "grpo_ablation.png"
        plot_ablation(data["ablation"], ab_path)
        created.append(ab_path)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRPO training and ablation results.")
    parser.add_argument("--input", type=str, default="experiments/results.json")
    parser.add_argument("--assets-dir", type=str, default="assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = generate_all_plots(Path(args.input), Path(args.assets_dir))
    for path in created:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
