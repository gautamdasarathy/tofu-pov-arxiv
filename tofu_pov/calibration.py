"""Shared validation-calibration helpers for experiment runners."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Iterable, TypeVar

import numpy as np

T = TypeVar("T", bound=Hashable)

THRESHOLD_GRID = (0.03, 0.05, 0.07, 0.10, 0.15, 0.25, 0.50, 1.00)
QUICK_THRESHOLD_GRID = (0.10, 1.00)
OFUL_LAMBDA_GRID = (0.1, 1.0, 10.0)
QUICK_OFUL_LAMBDA_GRID = (1.0,)
OFUL_BETA_SCALE_GRID = (0.25, 0.5, 1.0, 2.0)
QUICK_OFUL_BETA_SCALE_GRID = (0.5, 1.0)


def select_by_mean(
    results: dict[T, list[float]],
    *,
    tie_break,
) -> T:
    """Select a candidate by mean regret plus a deterministic tie-break."""

    if not results:
        raise ValueError("Cannot select from an empty calibration result set.")
    return min(results, key=lambda candidate: (float(np.mean(results[candidate])), tie_break(candidate)))


def select_fixed_rank(results: dict[int, list[float]]) -> int:
    return select_by_mean(results, tie_break=lambda rank: int(rank))


def select_threshold(results: dict[float, list[float]]) -> float:
    return select_by_mean(results, tie_break=lambda value: float(value))


def oful_tie_break(candidate: tuple[float, float]) -> tuple[float, float, float]:
    lambda_reg, beta_scale = candidate
    distance = abs(math.log(lambda_reg)) + abs(math.log(beta_scale))
    return (distance, float(lambda_reg), float(beta_scale))


def select_oful(results: dict[tuple[float, float], list[float]]) -> tuple[float, float]:
    return select_by_mean(results, tie_break=oful_tie_break)


def threshold_grid(quick: bool) -> tuple[float, ...]:
    return QUICK_THRESHOLD_GRID if quick else THRESHOLD_GRID


def oful_lambda_grid(quick: bool) -> tuple[float, ...]:
    return QUICK_OFUL_LAMBDA_GRID if quick else OFUL_LAMBDA_GRID


def oful_beta_scale_grid(quick: bool) -> tuple[float, ...]:
    return QUICK_OFUL_BETA_SCALE_GRID if quick else OFUL_BETA_SCALE_GRID


def candidate_label(
    *,
    rank: int | str = "",
    rank_threshold_constant: float | str = "",
    lambda_reg: float | str = "",
    beta_scale: float | str = "",
) -> str:
    parts: list[str] = []
    if rank != "":
        parts.append(f"rank={rank}")
    if rank_threshold_constant != "":
        parts.append(f"threshold={float(rank_threshold_constant):g}")
    if lambda_reg != "":
        parts.append(f"lambda={float(lambda_reg):g}")
    if beta_scale != "":
        parts.append(f"beta={float(beta_scale):g}")
    return ",".join(parts) if parts else "default"


def mark_selected(
    rows: Iterable[dict[str, float | int | str]],
    selected_by_family: dict[str, object],
) -> list[dict[str, float | int | str]]:
    marked: list[dict[str, float | int | str]] = []
    for row in rows:
        item = dict(row)
        family = str(item["method_family"])
        selected = selected_by_family.get(family)
        candidate = _row_candidate(item)
        item["selected"] = int(candidate == selected)
        marked.append(item)
    return marked


def _row_candidate(row: dict[str, float | int | str]) -> object:
    family = str(row["method_family"])
    if family == "Zero-imputed OFUL":
        return (float(row["lambda_reg"]), float(row["beta_scale"]))
    if "adaptive-rank" in family:
        return float(row["rank_threshold_constant"])
    if "fixed-rank" in family:
        return int(row["rank"])
    return row.get("candidate_label", "")
