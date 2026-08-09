from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EXPERIMENT_BINOMIAL_NOISE as EXPERIMENT_ID
from config import N_OBSERVATION_DRAWS as DEFAULT_N_DRAWS
from config import PRIMARY_THRESHOLD as DEFAULT_PRIMARY_THRESHOLD
from config import STATE_U_LEVELS as DEFAULT_SEVERITIES
from config import TRUE_COUNT_GROUP_BOUNDS as COUNT_GROUP_BOUNDS
from config import TRUE_COUNT_GROUP_BOUNDS as DEFAULT_COUNT_GROUP_BOUNDS
from config import TRUE_COUNT_GROUP_LABELS as COUNT_GROUP_LABELS
from config import TRUE_COUNT_GROUP_LABELS as DEFAULT_COUNT_GROUP_LABELS
from config import derive_seed
from drn import compute_drn


class BinomialReportingNoiseError(ValueError):
    pass


def context_seed(base_seed, u: float, draw_index: int, state: str = "kansas", trajectory_id: str = "primary") -> int:
    return derive_seed(base_seed, state, trajectory_id, EXPERIMENT_ID, "u", u, "draw", draw_index)


def _validate_counts(I: np.ndarray) -> np.ndarray:
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise BinomialReportingNoiseError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")
    if not np.all(np.isfinite(I)):
        raise BinomialReportingNoiseError("I must be finite.")
    if np.any(I < 0):
        raise BinomialReportingNoiseError("I must be nonnegative.")
    if not np.allclose(I, np.round(I)):
        raise BinomialReportingNoiseError("I must contain integer counts.")
    return np.round(I).astype(np.int64)


def draw_reported_counts(
    I: np.ndarray,
    u: float,
    n_draws: int,
    base_seed,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= u < 1.0):
        raise BinomialReportingNoiseError(f"u must satisfy 0 <= u < 1, got {u}.")
    if n_draws <= 0:
        raise BinomialReportingNoiseError(f"n_draws must be positive, got {n_draws}.")

    I_int = _validate_counts(I)
    n_times, n = I_int.shape

    if u == 0.0:
        return np.ones((1, n)), I_int[np.newaxis, :, :].astype(float)

    alphas = np.empty((n_draws, n))
    I_tilde = np.empty((n_draws, n_times, n))
    for k in range(n_draws):
        seed = context_seed(base_seed, u, k, state, trajectory_id)
        rng = np.random.default_rng(seed)
        alpha = rng.uniform(1.0 - u, 1.0, size=n)
        alphas[k] = alpha
        I_tilde[k] = rng.binomial(I_int, alpha[np.newaxis, :])
    return alphas, I_tilde


@dataclass
class GroupSummary:
    n: int
    availability_rate: float
    misclassification_rate: float


@dataclass
class SeverityResult:
    u: float
    n_draws: int
    n_primary_county_times: int
    availability_rate: float
    zero_report_rate: float
    threshold_misclassification_rate: float
    mean_abs_error: float
    median_abs_error: float
    mean_abs_log_ratio_error: float
    median_abs_log_ratio_error: float
    group_summary: dict


def run_severity(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    u: float,
    base_seed,
    n_draws: int = DEFAULT_N_DRAWS,
    primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> SeverityResult:
    I_int = _validate_counts(I)

    primary_mask = I_int >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BinomialReportingNoiseError(
            f"No county-times reach I >= {primary_threshold}; cannot run the diagnostic."
        )

    true_result = compute_drn(S, I_int, N, B, gamma)
    true_drn_primary = true_result.drn[primary_mask]
    true_counts_primary = I_int[primary_mask].astype(float)

    draws = 1 if u == 0.0 else n_draws
    _, I_tildes = draw_reported_counts(I_int, u, draws, base_seed, state=state, trajectory_id=trajectory_id)
    n_used = I_tildes.shape[0]

    avail_parts, point_parts = [], []
    for k in range(n_used):
        point_result = compute_drn(S, I_tildes[k], N, B, gamma)
        avail_parts.append(point_result.defined_mask[primary_mask])
        point_parts.append(point_result.drn[primary_mask])

    avail_all = np.concatenate(avail_parts)
    point_all = np.concatenate(point_parts)
    true_all = np.tile(true_drn_primary, n_used)
    counts_all = np.tile(true_counts_primary, n_used)

    availability_rate = float(avail_all.mean())
    zero_report_rate = float((~avail_all).mean())

    point_a = point_all[avail_all]
    true_a = true_all[avail_all]

    misclassification = (point_a - 1.0) * (true_a - 1.0) < 0.0
    abs_error = np.abs(point_a - true_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_log_ratio_error = np.abs(np.log(point_a / true_a))

    def _mean(arr):
        return float(np.mean(arr)) if arr.size else float("nan")

    def _median(arr):
        return float(np.median(arr)) if arr.size else float("nan")

    group_summary: dict[str, GroupSummary] = {}
    for (lo, hi), label in zip(COUNT_GROUP_BOUNDS, COUNT_GROUP_LABELS):
        in_group = (counts_all >= lo) & (counts_all < hi)
        n_group = int(in_group.sum())
        if n_group == 0:
            group_summary[label] = GroupSummary(n=0, availability_rate=float("nan"), misclassification_rate=float("nan"))
            continue
        group_avail_rate = float(avail_all[in_group].mean())
        group_avail_and_in = in_group & avail_all
        if group_avail_and_in.any():
            gp = point_all[group_avail_and_in]
            gt = true_all[group_avail_and_in]
            group_misclass_rate = float(((gp - 1.0) * (gt - 1.0) < 0.0).mean())
        else:
            group_misclass_rate = float("nan")
        group_summary[label] = GroupSummary(
            n=n_group, availability_rate=group_avail_rate, misclassification_rate=group_misclass_rate
        )

    return SeverityResult(
        u=u,
        n_draws=n_used,
        n_primary_county_times=n_primary,
        availability_rate=availability_rate,
        zero_report_rate=zero_report_rate,
        threshold_misclassification_rate=_mean(misclassification),
        mean_abs_error=_mean(abs_error),
        median_abs_error=_median(abs_error),
        mean_abs_log_ratio_error=float(np.nanmean(abs_log_ratio_error)) if abs_log_ratio_error.size else float("nan"),
        median_abs_log_ratio_error=float(np.nanmedian(abs_log_ratio_error)) if abs_log_ratio_error.size else float("nan"),
        group_summary=group_summary,
    )


STAT_FIELDS = (
    "n",
    "n_available",
    "availability_rate",
    "zero_report_rate",
    "threshold_misclassification_rate",
    "mean_signed_error",
    "median_signed_error",
    "mean_abs_error",
    "median_abs_error",
    "mean_relative_error",
    "median_relative_error",
)


def summarize_cells(true_arr: np.ndarray, point_arr: np.ndarray, available_arr: np.ndarray) -> dict:
    n = int(true_arr.shape[0])
    if n == 0:
        return {field: (0 if field in ("n", "n_available") else float("nan")) for field in STAT_FIELDS}

    avail = available_arr
    availability_rate = float(avail.mean())
    zero_report_rate = float((~avail).mean())

    point_a = point_arr[avail]
    true_a = true_arr[avail]

    misclassification = (point_a - 1.0) * (true_a - 1.0) < 0.0
    signed_error = point_a - true_a
    abs_error = np.abs(signed_error)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = np.where(true_a != 0, signed_error / true_a, np.nan)

    def _mean(arr):
        return float(np.mean(arr)) if arr.size else float("nan")

    def _median(arr):
        return float(np.median(arr)) if arr.size else float("nan")

    return {
        "n": n,
        "n_available": int(avail.sum()),
        "availability_rate": availability_rate,
        "zero_report_rate": zero_report_rate,
        "threshold_misclassification_rate": _mean(misclassification),
        "mean_signed_error": _mean(signed_error),
        "median_signed_error": _median(signed_error),
        "mean_abs_error": _mean(abs_error),
        "median_abs_error": _median(abs_error),
        "mean_relative_error": float(np.nanmean(relative_error)) if relative_error.size and np.any(~np.isnan(relative_error)) else float("nan"),
        "median_relative_error": float(np.nanmedian(relative_error)) if relative_error.size and np.any(~np.isnan(relative_error)) else float("nan"),
    }


def _strata(true_arr: np.ndarray, I_primary: np.ndarray, count_group_bounds, count_group_labels) -> dict:
    n = true_arr.shape[0]
    count_groups = {"overall": np.ones(n, dtype=bool)}
    for (lo, hi), label in zip(count_group_bounds, count_group_labels):
        count_groups[label] = (I_primary >= lo) & (I_primary < hi)
    regimes = {"overall": np.ones(n, dtype=bool), "growth": true_arr > 1.0, "decline": true_arr < 1.0}
    return {
        (group_label, regime_label): g_mask & r_mask
        for group_label, g_mask in count_groups.items()
        for regime_label, r_mask in regimes.items()
    }


def verify_binomial_sample(I_true: np.ndarray, I_tilde: np.ndarray) -> bool:
    if not np.all(np.isfinite(I_tilde)):
        return False
    if not np.allclose(I_tilde, np.round(I_tilde)):
        return False
    return bool(np.all((I_tilde >= 0) & (I_tilde <= I_true)))


def run_trajectory_severity(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    u: float,
    base_seed,
    n_draws: int = DEFAULT_N_DRAWS,
    primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD,
    state: str = "kansas",
    trajectory_id: str = "primary",
    count_group_bounds=DEFAULT_COUNT_GROUP_BOUNDS,
    count_group_labels=DEFAULT_COUNT_GROUP_LABELS,
) -> tuple[list[dict], list[dict]]:
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise BinomialReportingNoiseError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BinomialReportingNoiseError(
            f"No county-times reach I >= {primary_threshold} for trajectory_id={trajectory_id!r}; "
            "cannot run the diagnostic."
        )
    I_primary = I[primary_mask]

    true_result = compute_drn(S, I, N, B, gamma)
    true_drn_primary = true_result.drn[primary_mask]

    draws = 1 if u == 0.0 else n_draws
    _, I_tildes = draw_reported_counts(I, u, draws, base_seed, state=state, trajectory_id=trajectory_id)
    n_used = I_tildes.shape[0]

    pooled = {"point": [], "available": []}
    draw_rows: list[dict] = []

    for k in range(n_used):
        if not verify_binomial_sample(I, I_tildes[k]):
            raise BinomialReportingNoiseError(f"draw {k} (u={u}) produced an invalid binomial sample; trajectory_id={trajectory_id!r}.")
        seed = context_seed(base_seed, u, k, state, trajectory_id)
        point_result = compute_drn(S, I_tildes[k], N, B, gamma)
        point_drn = point_result.drn[primary_mask]
        available = point_result.defined_mask[primary_mask]

        for (group_label, regime_label), mask in _strata(true_drn_primary, I_primary, count_group_bounds, count_group_labels).items():
            stats = summarize_cells(true_drn_primary[mask], point_drn[mask], available[mask])
            draw_rows.append(
                {
                    "state": state, "trajectory_id": trajectory_id, "u": u, "p": 1.0 - u,
                    "draw_index": k, "seed": seed, "count_group": group_label, "regime": regime_label, **stats,
                }
            )

        pooled["point"].append(point_drn)
        pooled["available"].append(available)

    pooled_point = np.concatenate(pooled["point"])
    pooled_available = np.concatenate(pooled["available"])
    pooled_true = np.tile(true_drn_primary, n_used)
    pooled_I_primary = np.tile(I_primary, n_used)

    trajectory_rows: list[dict] = []
    for (group_label, regime_label), mask in _strata(pooled_true, pooled_I_primary, count_group_bounds, count_group_labels).items():
        stats = summarize_cells(pooled_true[mask], pooled_point[mask], pooled_available[mask])
        trajectory_rows.append(
            {
                "state": state, "trajectory_id": trajectory_id, "u": u, "p": 1.0 - u,
                "n_draws": n_used, "count_group": group_label, "regime": regime_label, **stats,
            }
        )

    return draw_rows, trajectory_rows
