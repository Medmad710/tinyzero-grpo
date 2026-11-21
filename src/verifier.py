"""Strict AST-based math verifier and RLVR reward function for Countdown."""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import math
import re
from typing import Sequence


_FORMAT_PATTERN = re.compile(
    r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",
    re.DOTALL,
)
_DISALLOWED_INNER_TAGS = ("<think>", "</think>", "<answer>", "</answer>")
_MAX_EXPR_LEN = 512
_MAX_AST_DEPTH = 64


def extract_think_and_answer(completion: str) -> tuple[str | None, str | None]:
    """Extract `(think_text, answer_text)` if completion strictly matches format, else `(None, None)`."""
    if not isinstance(completion, str):
        return None, None

    match = _FORMAT_PATTERN.match(completion)
    if match is None:
        return None, None

    think_text, answer_text = match.group(1), match.group(2)
    for tag in _DISALLOWED_INNER_TAGS:
        if tag in think_text or tag in answer_text:
            return None, None

    answer_clean = answer_text.strip()
    if not answer_clean:
        return None, None

    return think_text.strip(), answer_clean


def _eval_ast_node(node: ast.AST, used_numbers: list[int], depth: int = 0) -> Fraction:
    """Recursively evaluate a whitelisted arithmetic AST node using exact Fractions."""
    if depth > _MAX_AST_DEPTH:
        raise ValueError("Expression exceeds maximum AST depth.")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, used_numbers, depth + 1)

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, used_numbers, depth + 1)
        right = _eval_ast_node(node.right, used_numbers, depth + 1)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero in Countdown expression.")
            return left / right

        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Constant):
        value = node.value
        # Reject booleans (subclass of int in Python), floats, strings, None, etc.
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            used_numbers.append(value)
            return Fraction(value)
        raise ValueError(f"Unsupported constant: {value!r}")

    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_eval_countdown(
    expr: str,
    numbers: Sequence[int],
    target: int,
    tol: float = 1e-6,
) -> bool:
    """Safely parse and verify an arithmetic expression against Countdown constraints.

    Checks:
    1. Expression parses cleanly as a single Python expression (`ast.parse(..., mode='eval')`).
    2. Contains ONLY non-negative integer literals, parentheses, and binary `+`, `-`, `*`, `/`.
    3. Multiset of integer literals in the expression equals `numbers` with exact multiplicity.
    4. Exact Fraction evaluation equals `target` within `tol` (and safely rejects division by zero).
    """
    if not isinstance(expr, str):
        return False
    cleaned = expr.strip()
    if not cleaned or len(cleaned) > _MAX_EXPR_LEN:
        return False

    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return False

    used_numbers: list[int] = []
    try:
        result = _eval_ast_node(tree, used_numbers)
    except (ValueError, ZeroDivisionError, RecursionError, TypeError, OverflowError):
        return False

    if Counter(used_numbers) != Counter(int(n) for n in numbers):
        return False

    try:
        result_float = float(result)
    except OverflowError:
        return False

    if not math.isfinite(result_float):
        return False

    return abs(result_float - float(target)) < tol


def compute_reward(
    completion: str,
    numbers: Sequence[int],
    target: int,
    format_reward: float = 0.15,
    correct_reward: float = 1.0,
) -> tuple[float, bool, bool]:
    """Compute RLVR reward for a Countdown completion.

    Returns:
        `(total_reward, is_correct, has_valid_format)`
    """
    _, answer_expr = extract_think_and_answer(completion)
    if answer_expr is None:
        return 0.0, False, False

    is_correct = safe_eval_countdown(answer_expr, numbers=numbers, target=target)
    total_reward = format_reward + (correct_reward if is_correct else 0.0)
    return total_reward, is_correct, True
