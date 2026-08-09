from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import COMBINED_B as DEFAULT_B
from config import COMBINED_U as DEFAULT_U
from config import EXPERIMENT_COMBINED_UNCERTAINTY as EXPERIMENT_ID
from config import N_OBSERVATION_DRAWS as DEFAULT_N_DRAWS
from config import PRIMARY_THRESHOLD as DEFAULT_PRIMARY_THRESHOLD
from config import TRUE_COUNT_GROUP_BOUNDS as DEFAULT_COUNT_GROUP_BOUNDS
from config import TRUE_COUNT_GROUP_LABELS as DEFAULT_COUNT_GROUP_LABELS
from config import derive_seed
from drn import compute_drn, compute_weights


class CombinedUncertaintyError(ValueError):
    pass


def context_seed(
    base_seed,
    stream: str,
    severity: float,
    draw_index: int,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> int:
    return derive_seed(base_seed, state, trajectory_id, EXPERIMENT_ID, stream, severity, "draw", draw_index)


def draw_realizations(n_counties: int, u: float, b: float, n_draws: int, base_seed, state: str = "kansas", trajectory_id: str = "primary"):
    if not (0.0 <= u < 1.0):
        raise CombinedUncertaintyError(f"u must satisfy 0 <= u < 1, got {u}.")
    if not (0.0 <= b < 1.0):
        raise CombinedUncertaintyError(f"b must satisfy 0 <= b < 1, got {b}.")
    if n_draws <= 0:
        raise CombinedUncertaintyError(f"n_draws must be positive, got {n_draws}.")
    if n_counties <= 0:
        raise CombinedUncertaintyError(f"n_counties must be positive, got {n_counties}.")

    if u == 0.0:
        alphas = np.ones((n_draws, n_counties))
    else:
        alphas = np.empty((n_draws, n_counties))
        for k in range(n_draws):
            seed = context_seed(base_seed, "alpha", u, k, state, trajectory_id)
            alphas[k] = np.random.default_rng(seed).uniform(1.0 - u, 1.0, size=n_counties)

    if b == 0.0:
        xis = np.zeros((n_draws, n_counties, n_counties))
    else:
        xis = np.empty((n_draws, n_counties, n_counties))
        for k in range(n_draws):
            seed = context_seed(base_seed, "xi", b, k, state, trajectory_id)
            xis[k] = np.random.default_rng(seed).uniform(-b, b, size=(n_counties, n_counties))

    return alphas, xis


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
    B0: np.ndarray,
    gamma: float,
    alpha: np.ndarray,
    xi: np.ndarray,
    u: float,
    b: float,
    primary_mask: np.ndarray,
) -> DrawComparison:
    I_tilde = alpha[np.newaxis, :] * I
    B_hat = B0 * (1.0 + xi)
    x_tilde = I_tilde / N

    true_result = compute_drn(S, I, N, B0, gamma)
    point_result = compute_drn(S, I_tilde, N, B_hat, gamma)
    hat_drn = point_result.drn

    weights = compute_weights(x_tilde, B_hat)
    defined_mask = x_tilde > 0

    q_hat_safe = np.where(weights.defined_mask, weights.q_i, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.where(defined_mask, (hat_drn / (1.0 + b)) * (1.0 - q_hat_safe * u), np.nan)
        upper = np.where(
            defined_mask, (hat_drn / (1.0 - b)) * (1.0 + q_hat_safe * u / (1.0 - u)), np.nan
        )

    growth = defined_mask & (lower > 1.0)
    decline = defined_mask & (upper < 1.0)
    indeterminate = defined_mask & ~growth & ~decline

    decision = np.full(hat_drn.shape, "undefined", dtype="<U20")
    decision[growth] = "certified_growth"
    decision[decline] = "certified_decline"
    decision[indeterminate] = "indeterminate"

    return DrawComparison(
        true_drn=true_result.drn[primary_mask],
        point_drn=hat_drn[primary_mask],
        lower=lower[primary_mask],
        upper=upper[primary_mask],
        available=defined_mask[primary_mask],
        decision=decision[primary_mask],
    )


@dataclass
class CombinedResult:
    u: float
    b: float
    n_draws: int
    n_primary_county_times: int
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


def run_combined(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B0: np.ndarray,
    gamma: float,
    u: float = DEFAULT_U,
    b: float = DEFAULT_B,
    base_seed=None,
    n_draws: int = DEFAULT_N_DRAWS,
    primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> CombinedResult:
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise CombinedUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")
    n_times, n = I.shape

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise CombinedUncertaintyError(
            f"No county-times reach I >= {primary_threshold}; cannot run the experiment."
        )

    draws = 1 if (u == 0.0 and b == 0.0) else n_draws
    alphas, xis = draw_realizations(n, u, b, draws, base_seed, state=state, trajectory_id=trajectory_id)

    true_parts, point_parts, lower_parts, upper_parts = [], [], [], []
    available_parts, decision_parts = [], []

    for k in range(alphas.shape[0]):
        cmp = compare_draw(S, I, N, B0, gamma, alphas[k], xis[k], u, b, primary_mask)
        true_parts.append(cmp.true_drn)
        point_parts.append(cmp.point_drn)
        lower_parts.append(cmp.lower)
        upper_parts.append(cmp.upper)
        available_parts.append(cmp.available)
        decision_parts.append(cmp.decision)

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
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(true_a != 0, abs_width / true_a, np.nan)

    def _mean(mask_arr):
        return float(mask_arr.mean()) if mask_arr.size else float("nan")

    return CombinedResult(
        u=u,
        b=b,
        n_draws=alphas.shape[0],
        n_primary_county_times=n_primary,
        availability_rate=float(avail.sum() / n_all) if n_all else float("nan"),
        truth_containment_rate=_mean(containment),
        false_certification_rate=_mean(false_certification),
        certified_growth_rate=_mean(certified_growth),
        certified_decline_rate=_mean(certified_decline),
        certification_rate=_mean(certified_growth | certified_decline),
        indeterminate_rate=_mean(indeterminate),
        point_misclassification_rate=_mean(misclassification),
        mean_abs_interval_width=float(np.mean(abs_width)) if abs_width.size else float("nan"),
        median_abs_interval_width=float(np.median(abs_width)) if abs_width.size else float("nan"),
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
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(true_a != 0, abs_width / true_a, np.nan)

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
        "mean_abs_interval_width": float(np.mean(abs_width)) if abs_width.size else float("nan"),
        "median_abs_interval_width": float(np.median(abs_width)) if abs_width.size else float("nan"),
        "mean_relative_interval_width": float(np.nanmean(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
        "median_relative_interval_width": float(np.nanmedian(rel_width)) if np.any(~np.isnan(rel_width)) else float("nan"),
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


def verify_alpha_bounds(alpha: np.ndarray, u: float) -> bool:
    if u == 0.0:
        return bool(np.all(alpha == 1.0))
    return bool(np.all((alpha >= 1.0 - u - 1e-12) & (alpha <= 1.0 + 1e-12)))


def run_trajectory_combined(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B0: np.ndarray,
    gamma: float,
    u: float,
    b: float,
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
        raise CombinedUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise CombinedUncertaintyError(
            f"No county-times reach I >= {primary_threshold} for trajectory_id={trajectory_id!r}; "
            "cannot run the experiment."
        )
    I_primary = I[primary_mask]

    draws = 1 if (u == 0.0 and b == 0.0) else n_draws
    alphas, xis = draw_realizations(I.shape[1], u, b, draws, base_seed, state=state, trajectory_id=trajectory_id)

    pooled = {"true": [], "point": [], "lower": [], "upper": [], "available": [], "decision": []}
    draw_rows: list[dict] = []

    for k in range(alphas.shape[0]):
        if not verify_alpha_bounds(alphas[k], u):
            raise CombinedUncertaintyError(f"draw {k}: alpha outside [1-u, 1]; trajectory_id={trajectory_id!r}.")
        seed_alpha = context_seed(base_seed, "alpha", u, k, state, trajectory_id)
        seed_xi = context_seed(base_seed, "xi", b, k, state, trajectory_id)
        cmp = compare_draw(S, I, N, B0, gamma, alphas[k], xis[k], u, b, primary_mask)

        for (group_label, regime_label), mask in _strata(cmp.true_drn, I_primary, count_group_bounds, count_group_labels).items():
            stats = summarize_cells(cmp.true_drn[mask], cmp.point_drn[mask], cmp.lower[mask], cmp.upper[mask], cmp.available[mask], cmp.decision[mask])
            draw_rows.append(
                {
                    "state": state, "trajectory_id": trajectory_id, "u": u, "b": b,
                    "draw_index": k, "seed_alpha": seed_alpha, "seed_xi": seed_xi,
                    "count_group": group_label, "regime": regime_label, **stats,
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
    for (group_label, regime_label), mask in _strata(pooled_true, pooled_I_primary, count_group_bounds, count_group_labels).items():
        stats = summarize_cells(pooled_true[mask], pooled_point[mask], pooled_lower[mask], pooled_upper[mask], pooled_available[mask], pooled_decision[mask])
        trajectory_rows.append(
            {
                "state": state, "trajectory_id": trajectory_id, "u": u, "b": b,
                "n_draws": int(alphas.shape[0]), "count_group": group_label, "regime": regime_label, **stats,
            }
        )

    return draw_rows, trajectory_rows
