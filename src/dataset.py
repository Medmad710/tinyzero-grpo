"""Deterministic Countdown Numbers Game puzzle generator and prompt formatter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Sequence


SYSTEM_PROMPT = (
    "You are a precise mathematical reasoning assistant solving the Countdown Numbers Game.\n"
    "Given target integer T and a list of available integers, form an arithmetic expression that "
    "equals T using EACH given number AT MOST ONCE and in fact using ALL given numbers EXACTLY ONCE.\n"
    "Allowed operations: +, -, *, / (exact division) and parentheses ().\n"
    "Always reason step-by-step inside <think>...</think>, then output ONLY the final arithmetic "
    "expression (no '=' sign, no target number, no extra text) inside <answer>...</answer>.\n\n"
    "Example:\n"
    "Numbers: [4, 7, 10], Target: 38\n"
    "<think>\n"
    "We have 4, 7, 10 and want 38. Note that 4 * 10 = 40, and 40 - 7 = 33 (not 38). "
    "Try 4 * 7 = 28, then 28 + 10 = 38. Uses 4, 7, 10 once each.\n"
    "</think>\n"
    "<answer>(4 * 7) + 10</answer>"
)


@dataclass(frozen=True)
class CountdownPuzzle:
    """A single guaranteed-solvable Countdown puzzle."""

    puzzle_id: int
    numbers: tuple[int, ...]
    target: int
    solution: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the puzzle into a JSON-compatible dictionary."""
        data = asdict(self)
        data["numbers"] = list(self.numbers)
        return data


def _strip_outer_parens(expr: str) -> str:
    """Remove a single balanced pair of outer parentheses if present."""
    if len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
        depth = 0
        for idx, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth == 0 and idx < len(expr) - 1:
                return expr
        return expr[1:-1]
    return expr


def _sample_expression(nums: Sequence[int], rng: random.Random) -> tuple[int, str] | None:
    """Randomly combine `nums` into a valid integer-valued expression tree."""
    nodes: list[tuple[int, str]] = [(n, str(n)) for n in nums]
    rng.shuffle(nodes)

    while len(nodes) > 1:
        # Pick two distinct indices to combine.
        idx_a = rng.randrange(len(nodes))
        val_a, expr_a = nodes.pop(idx_a)
        idx_b = rng.randrange(len(nodes))
        val_b, expr_b = nodes.pop(idx_b)

        candidates: list[tuple[int, str]] = [
            (val_a + val_b, f"({expr_a} + {expr_b})"),
            (val_a * val_b, f"({expr_a} * {expr_b})"),
        ]
        if val_a >= val_b:
            candidates.append((val_a - val_b, f"({expr_a} - {expr_b})"))
        else:
            candidates.append((val_b - val_a, f"({expr_b} - {expr_a})"))

        if val_b != 0 and val_a % val_b == 0:
            candidates.append((val_a // val_b, f"({expr_a} / {expr_b})"))
        elif val_a != 0 and val_b % val_a == 0:
            candidates.append((val_b // val_a, f"({expr_b} / {expr_a})"))

        if not candidates:
            return None

        nodes.append(rng.choice(candidates))

    val, expr = nodes[0]
    return val, _strip_outer_parens(expr)


def generate_dataset(
    num_train: int = 500,
    num_test: int = 60,
    seed: int = 42,
    num_operands: int = 3,
    min_num: int = 1,
    max_num: int = 25,
    min_target: int = 10,
    max_target: int = 100,
) -> tuple[list[CountdownPuzzle], list[CountdownPuzzle]]:
    """Generate deterministic, globally deduplicated, disjoint train and test Countdown splits."""
    if num_operands < 2:
        raise ValueError("num_operands must be >= 2.")
    total_needed = num_train + num_test
    if total_needed <= 0:
        return [], []

    rng = random.Random(seed)
    seen_keys: set[tuple[tuple[int, ...], int]] = set()
    puzzles: list[CountdownPuzzle] = []

    max_attempts = max(total_needed * 200, 10_000)
    attempts = 0
    while len(puzzles) < total_needed and attempts < max_attempts:
        attempts += 1
        raw_nums = tuple(rng.randint(min_num, max_num) for _ in range(num_operands))
        sampled = _sample_expression(raw_nums, rng)
        if sampled is None:
            continue
        target, solution = sampled
        if not (min_target <= target <= max_target):
            continue

        # Deduplicate across sorted numbers + target so train and test share no equivalent puzzle.
        dedup_key = (tuple(sorted(raw_nums)), target)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        puzzles.append(
            CountdownPuzzle(
                puzzle_id=len(puzzles),
                numbers=raw_nums,
                target=target,
                solution=solution,
            )
        )

    if len(puzzles) < total_needed:
        raise RuntimeError(
            f"Generated only {len(puzzles)} unique puzzles out of {total_needed} requested "
            f"after {max_attempts} attempts."
        )

    return puzzles[:num_train], puzzles[num_train:total_needed]


def build_messages(puzzle: CountdownPuzzle) -> list[dict[str, str]]:
    """Build chat messages instructing `<think>...</think><answer>...</answer>` output."""
    nums_str = ", ".join(str(n) for n in puzzle.numbers)
    user_content = (
        f"Available numbers: [{nums_str}]\n"
        f"Target: {puzzle.target}\n"
        "Use every available number exactly once with +, -, *, /, and parentheses to reach the target.\n"
        "Respond with <think>...</think> followed by <answer>...</answer>."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_prompt(puzzle: CountdownPuzzle, tokenizer: Any = None) -> str:
    """Format a Countdown puzzle prompt using the tokenizer's chat template when available."""
    messages = build_messages(puzzle)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{messages[0]['content']}\n\n{messages[1]['content']}\nAssistant:\n"
