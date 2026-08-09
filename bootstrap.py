from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import derive_seed


class TrajectoryBootstrapError(ValueError):
    pass


def stratum_seed(base_seed, experiment_id: str, state: str, seed_county: str) -> int:
    return derive_seed(base_seed, experiment_id, state, seed_county)


def generate_index_matrix(n_replicates: int, n_per_stratum: int, seed) -> np.ndarray:
    if n_replicates <= 0 or n_per_stratum <= 0:
        raise TrajectoryBootstrapError(f"n_replicates and n_per_stratum must be positive, got {n_replicates}, {n_per_stratum}.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_per_stratum, size=(n_replicates, n_per_stratum))


def reconstruct_numerator(rate: np.ndarray, denom: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    denom = np.asarray(denom, dtype=float)
    if rate.shape != denom.shape:
        raise TrajectoryBootstrapError(f"rate and denom must share shape, got {rate.shape} and {denom.shape}.")
    with np.errstate(invalid="ignore"):
        return np.where(denom > 0, np.round(rate * denom), 0.0)


def reconstruct_weighted_numerator(mean_value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_value = np.asarray(mean_value, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if mean_value.shape != weight.shape:
        raise TrajectoryBootstrapError(f"mean_value and weight must share shape, got {mean_value.shape} and {weight.shape}.")
    with np.errstate(invalid="ignore"):
        return np.where(weight > 0, mean_value * weight, 0.0)


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_replicates_used: int


def bootstrap_pooled_rate(
    numerator_by_stratum: dict,
    denom_by_stratum: dict,
    index_matrix_by_stratum: dict,
    strata: list,
    confidence_level: float,
) -> BootstrapResult:
    if not strata:
        raise TrajectoryBootstrapError("strata must be non-empty.")
    missing = [k for k in strata if k not in index_matrix_by_stratum]
    if missing:
        raise TrajectoryBootstrapError(f"no index matrix for strata: {missing}")
    n_replicates = index_matrix_by_stratum[strata[0]].shape[0]

    total_num = np.zeros(n_replicates)
    total_den = np.zeros(n_replicates)
    point_num = 0.0
    point_den = 0.0
    for key in strata:
        idx = index_matrix_by_stratum[key]
        num20, den20 = numerator_by_stratum[key], denom_by_stratum[key]
        if idx.shape[0] != n_replicates:
            raise TrajectoryBootstrapError(f"stratum {key}: index matrix has {idx.shape[0]} replicates, expected {n_replicates}.")
        total_num += num20[idx].sum(axis=1)
        total_den += den20[idx].sum(axis=1)
        point_num += float(num20.sum())
        point_den += float(den20.sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        distribution = np.where(total_den > 0, total_num / total_den, np.nan)
    point_estimate = point_num / point_den if point_den > 0 else float("nan")

    finite = distribution[~np.isnan(distribution)]
    if finite.size == 0:
        return BootstrapResult(point_estimate=point_estimate, ci_lower=float("nan"), ci_upper=float("nan"), n_replicates_used=0)
    alpha = 1.0 - confidence_level
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(point_estimate=point_estimate, ci_lower=float(lo), ci_upper=float(hi), n_replicates_used=int(finite.size))
