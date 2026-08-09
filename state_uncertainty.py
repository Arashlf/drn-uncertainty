from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EXPERIMENT_STATE_UNCERTAINTY as EXPERIMENT_ID
from config import N_OBSERVATION_DRAWS as DEFAULT_N_DRAWS
from config import PRIMARY_THRESHOLD as DEFAULT_PRIMARY_THRESHOLD
from config import STATE_U_LEVELS as DEFAULT_SEVERITIES
from config import TRUE_COUNT_GROUP_BOUNDS as DEFAULT_COUNT_GROUP_BOUNDS
from config import TRUE_COUNT_GROUP_LABELS as DEFAULT_COUNT_GROUP_LABELS
from config import derive_seed
from drn import compute_drn, observable_certificate


class BoundedStateUncertaintyError(ValueError):
    pass


def context_seed(base_seed, u: float, draw_index: int, state: str = "kansas", trajectory_id: str = "primary") -> int:
    return derive_seed(base_seed, state, trajectory_id, EXPERIMENT_ID, "u", u, "draw", draw_index)


def draw_alpha(
    n_counties: int,
    u: float,
    n_draws: int,
    base_seed,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> np.ndarray:
    if not (0.0 <= u < 1.0):
        raise BoundedStateUncertaintyError(f"u must satisfy 0 <= u < 1, got {u}.")
    if n_draws <= 0:
        raise BoundedStateUncertaintyError(f"n_draws must be positive, got {n_draws}.")
    if n_counties <= 0:
        raise BoundedStateUncertaintyError(f"n_counties must be positive, got {n_counties}.")

    if u == 0.0:
        return np.ones((n_draws, n_counties))

    alpha = np.empty((n_draws, n_counties))
    for k in range(n_draws):
        seed = context_seed(base_seed, u, k, state, trajectory_id)
        alpha[k] = np.random.default_rng(seed).uniform(1.0 - u, 1.0, size=n_counties)
    return alpha


@dataclass
class DrawComparison:
    true_drn: np.ndarray
    point_drn: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    available: np.ndarray
    decision: np.ndarray


def compare_draw(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    alpha: np.ndarray,
    a: float,
    primary_mask: np.ndarray,
) -> DrawComparison:
    I_tilde = alpha[np.newaxis, :] * I

    true_result = compute_drn(S, I, N, B, gamma)
    point_result = compute_drn(S, I_tilde, N, B, gamma)
    cert = observable_certificate(S / N, I_tilde / N, B, gamma, a)

    return DrawComparison(
        true_drn=true_result.drn[primary_mask],
        point_drn=point_result.drn[primary_mask],
        lower=cert.lower[primary_mask],
        upper=cert.upper[primary_mask],
        available=cert.defined_mask[primary_mask],
        decision=cert.decision[primary_mask],
    )


@dataclass
class SeverityResult:
    u: float
    a: float
    n_draws: int
    n_primary_county_times: int
    fraction_zero: float
    fraction_low_signal: float
    fraction_primary: float
    fraction_primary_unavailable: float
    availability_rate: float
    truth_containment_rate: float
    false_certification_rate: float
    certified_growth_rate: float
    certified_decline_rate: float
    certification_rate: float
    indeterminate_rate: float
    point_misclassification_rate: float
    mean_abs_interval_width: float
    median_abs_interval_width: float
    mean_relative_interval_width: float
    median_relative_interval_width: float


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
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise BoundedStateUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")
    n_times, n = I.shape

    zero_mask = I == 0
    low_mask = (I > 0) & (I < primary_threshold)
    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BoundedStateUncertaintyError(
            f"No county-times reach I >= {primary_threshold}; cannot run the experiment."
        )

    a = 1.0 - u
    draws = 1 if u == 0.0 else n_draws
    alphas = draw_alpha(n, u, draws, base_seed, state=state, trajectory_id=trajectory_id)

    true_parts, point_parts, lower_parts, upper_parts = [], [], [], []
    available_parts, decision_parts = [], []
    unavailable_primary_fractions = []

    for k in range(alphas.shape[0]):
        cmp = compare_draw(S, I, N, B, gamma, alphas[k], a, primary_mask)
        true_parts.append(cmp.true_drn)
        point_parts.append(cmp.point_drn)
        lower_parts.append(cmp.lower)
        upper_parts.append(cmp.upper)
        available_parts.append(cmp.available)
        decision_parts.append(cmp.decision)
        unavailable_primary_fractions.append(1.0 - float(cmp.available.mean()))

    true_all = np.concatenate(true_parts)
    point_all = np.concatenate(point_parts)
    lower_all = np.concatenate(lower_parts)
    upper_all = np.concatenate(upper_parts)
    avail = np.concatenate(available_parts)
    decision_all = np.concatenate(decision_parts)
    n_all = true_all.shape[0]

    true_a = true_all[avail]
    point_a = point_all[avail]
    lower_a = lower_all[avail]
    upper_a = upper_all[avail]
    decision_a = decision_all[avail]

    containment = (lower_a <= true_a + 1e-9) & (true_a <= upper_a + 1e-9)

    certified_growth = decision_a == "certified_growth"
    certified_decline = decision_a == "certified_decline"
    indeterminate = decision_a == "indeterminate"

    false_certification = (certified_growth & (true_a <= 1.0)) | (certified_decline & (true_a >= 1.0))
    misclassification = (point_a - 1.0) * (true_a - 1.0) < 0.0

    abs_width = upper_a - lower_a
    finite_width = np.isfinite(abs_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(finite_width & (true_a != 0), abs_width / true_a, np.nan)

    def _mean(mask_arr):
        return float(mask_arr.mean()) if mask_arr.size else float("nan")

    return SeverityResult(
        u=u,
        a=a,
        n_draws=alphas.shape[0],
        n_primary_county_times=n_primary,
        fraction_zero=float(zero_mask.sum() / I.size),
        fraction_low_signal=float(low_mask.sum() / I.size),
        fraction_primary=float(primary_mask.sum() / I.size),
        fraction_primary_unavailable=float(np.mean(unavailable_primary_fractions)),
        availability_rate=float(avail.sum() / n_all) if n_all else float("nan"),
        truth_containment_rate=_mean(containment),
        false_certification_rate=_mean(false_certification),
        certified_growth_rate=_mean(certified_growth),
        certified_decline_rate=_mean(certified_decline),
        certification_rate=_mean(certified_growth | certified_decline),
        indeterminate_rate=_mean(indeterminate),
        point_misclassification_rate=_mean(misclassification),
        mean_abs_interval_width=float(np.mean(abs_width[finite_width])) if finite_width.any() else float("nan"),
        median_abs_interval_width=float(np.median(abs_width[finite_width])) if finite_width.any() else float("nan"),
        mean_relative_interval_width=float(np.nanmean(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
        median_relative_interval_width=float(np.nanmedian(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
    )


STAT_FIELDS = (
    "n",
    "n_available",
    "availability_rate",
    "containment_rate",
    "false_certification_rate",
    "certified_growth_rate",
    "certified_decline_rate",
    "certification_rate",
    "indeterminate_rate",
    "point_misclassification_rate",
    "mean_abs_interval_width",
    "median_abs_interval_width",
    "mean_relative_interval_width",
    "median_relative_interval_width",
)


def summarize_cells(
    true_arr: np.ndarray,
    point_arr: np.ndarray,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    available_arr: np.ndarray,
    decision_arr: np.ndarray,
) -> dict:
    n = int(true_arr.shape[0])
    if n == 0:
        return {field: (0 if field in ("n", "n_available") else float("nan")) for field in STAT_FIELDS}

    avail = available_arr
    true_a = true_arr[avail]
    point_a = point_arr[avail]
    lower_a = lower_arr[avail]
    upper_a = upper_arr[avail]
    decision_a = decision_arr[avail]

    containment = (lower_a <= true_a + 1e-9) & (true_a <= upper_a + 1e-9)
    certified_growth = decision_a == "certified_growth"
    certified_decline = decision_a == "certified_decline"
    indeterminate = decision_a == "indeterminate"
    false_certification = (certified_growth & (true_a <= 1.0)) | (certified_decline & (true_a >= 1.0))
    misclassification = (point_a - 1.0) * (true_a - 1.0) < 0.0

    abs_width = upper_a - lower_a
    finite_width = np.isfinite(abs_width)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(finite_width & (true_a != 0), abs_width / true_a, np.nan)

    def _mean(mask_arr):
        return float(mask_arr.mean()) if mask_arr.size else float("nan")

    return {
        "n": n,
        "n_available": int(avail.sum()),
        "availability_rate": float(avail.mean()),
        "containment_rate": _mean(containment),
        "false_certification_rate": _mean(false_certification),
        "certified_growth_rate": _mean(certified_growth),
        "certified_decline_rate": _mean(certified_decline),
        "certification_rate": _mean(certified_growth | certified_decline),
        "indeterminate_rate": _mean(indeterminate),
        "point_misclassification_rate": _mean(misclassification),
        "mean_abs_interval_width": float(np.mean(abs_width[finite_width])) if finite_width.any() else float("nan"),
        "median_abs_interval_width": float(np.median(abs_width[finite_width])) if finite_width.any() else float("nan"),
        "mean_relative_interval_width": float(np.nanmean(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
        "median_relative_interval_width": float(np.nanmedian(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
    }


def _count_groups(I_primary: np.ndarray, count_group_bounds, count_group_labels) -> dict:
    groups = {"overall": np.ones(I_primary.shape[0], dtype=bool)}
    for (lo, hi), label in zip(count_group_bounds, count_group_labels):
        groups[label] = (I_primary >= lo) & (I_primary < hi)
    return groups


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
        raise BoundedStateUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BoundedStateUncertaintyError(
            f"No county-times reach I >= {primary_threshold} for trajectory_id={trajectory_id!r}; "
            "cannot run the experiment."
        )
    I_primary = I[primary_mask]

    a = 1.0 - u
    draws = 1 if u == 0.0 else n_draws
    alphas = draw_alpha(I.shape[1], u, draws, base_seed, state=state, trajectory_id=trajectory_id)

    pooled = {"true": [], "point": [], "lower": [], "upper": [], "available": [], "decision": []}
    draw_rows: list[dict] = []

    for k in range(alphas.shape[0]):
        seed = context_seed(base_seed, u, k, state, trajectory_id)
        cmp = compare_draw(S, I, N, B, gamma, alphas[k], a, primary_mask)

        for group_label, group_mask in _count_groups(I_primary, count_group_bounds, count_group_labels).items():
            stats = summarize_cells(
                cmp.true_drn[group_mask],
                cmp.point_drn[group_mask],
                cmp.lower[group_mask],
                cmp.upper[group_mask],
                cmp.available[group_mask],
                cmp.decision[group_mask],
            )
            draw_rows.append(
                {
                    "state": state,
                    "trajectory_id": trajectory_id,
                    "u": u,
                    "a": a,
                    "draw_index": k,
                    "seed": seed,
                    "count_group": group_label,
                    **stats,
                }
            )

        pooled["true"].append(cmp.true_drn)
        pooled["point"].append(cmp.point_drn)
        pooled["lower"].append(cmp.lower)
        pooled["upper"].append(cmp.upper)
        pooled["available"].append(cmp.available)
        pooled["decision"].append(cmp.decision)

    pooled_true = np.concatenate(pooled["true"])
    pooled_point = np.concatenate(pooled["point"])
    pooled_lower = np.concatenate(pooled["lower"])
    pooled_upper = np.concatenate(pooled["upper"])
    pooled_available = np.concatenate(pooled["available"])
    pooled_decision = np.concatenate(pooled["decision"])
    pooled_I_primary = np.tile(I_primary, alphas.shape[0])

    trajectory_rows: list[dict] = []
    for group_label, group_mask in _count_groups(pooled_I_primary, count_group_bounds, count_group_labels).items():
        stats = summarize_cells(
            pooled_true[group_mask],
            pooled_point[group_mask],
            pooled_lower[group_mask],
            pooled_upper[group_mask],
            pooled_available[group_mask],
            pooled_decision[group_mask],
        )
        trajectory_rows.append(
            {
                "state": state,
                "trajectory_id": trajectory_id,
                "u": u,
                "a": a,
                "n_draws": int(alphas.shape[0]),
                "count_group": group_label,
                **stats,
            }
        )

    return draw_rows, trajectory_rows
