from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import COUPLING_B_LEVELS as DEFAULT_SEVERITIES
from config import EXPERIMENT_COUPLING_UNCERTAINTY as EXPERIMENT_ID
from config import N_OBSERVATION_DRAWS as DEFAULT_N_DRAWS
from config import PRIMARY_THRESHOLD as DEFAULT_PRIMARY_THRESHOLD
from config import TRUE_COUNT_GROUP_BOUNDS as DEFAULT_COUNT_GROUP_BOUNDS
from config import TRUE_COUNT_GROUP_LABELS as DEFAULT_COUNT_GROUP_LABELS
from config import derive_seed
from drn import compute_drn


class BoundedCouplingUncertaintyError(ValueError):
    pass


def context_seed(base_seed, b: float, draw_index: int, state: str = "kansas", trajectory_id: str = "primary") -> int:
    return derive_seed(base_seed, state, trajectory_id, EXPERIMENT_ID, "b", b, "draw", draw_index)


def draw_B_hat(
    B0: np.ndarray,
    b: float,
    n_draws: int,
    base_seed,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> np.ndarray:
    B0 = np.asarray(B0, dtype=float)
    if B0.ndim != 2 or B0.shape[0] != B0.shape[1]:
        raise BoundedCouplingUncertaintyError(f"B0 must be square, got shape {B0.shape}.")
    if not (0.0 <= b < 1.0):
        raise BoundedCouplingUncertaintyError(f"b must satisfy 0 <= b < 1, got {b}.")
    if n_draws <= 0:
        raise BoundedCouplingUncertaintyError(f"n_draws must be positive, got {n_draws}.")

    n = B0.shape[0]
    if b == 0.0:
        return np.broadcast_to(B0, (n_draws, n, n)).copy()

    B_hats = np.empty((n_draws, n, n))
    for k in range(n_draws):
        seed = context_seed(base_seed, b, k, state, trajectory_id)
        xi = np.random.default_rng(seed).uniform(-b, b, size=(n, n))
        B_hats[k] = B0 * (1.0 + xi)
    return B_hats


@dataclass
class DrawComparison:
    true_drn: np.ndarray
    point_drn: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    decision: np.ndarray


def compare_draw(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B0: np.ndarray,
    B_hat: np.ndarray,
    gamma: float,
    b: float,
    primary_mask: np.ndarray,
) -> DrawComparison:
    true_result = compute_drn(S, I, N, B0, gamma)
    point_result = compute_drn(S, I, N, B_hat, gamma)
    hat_drn = point_result.drn

    lower = hat_drn / (1.0 + b)
    upper = hat_drn / (1.0 - b)

    defined = ~np.isnan(hat_drn)
    growth = defined & (hat_drn > 1.0 + b)
    decline = defined & (hat_drn < 1.0 - b)
    indeterminate = defined & ~growth & ~decline

    decision = np.full(hat_drn.shape, "undefined", dtype="<U20")
    decision[growth] = "certified_growth"
    decision[decline] = "certified_decline"
    decision[indeterminate] = "indeterminate"

    return DrawComparison(
        true_drn=true_result.drn[primary_mask],
        point_drn=hat_drn[primary_mask],
        lower=lower[primary_mask],
        upper=upper[primary_mask],
        decision=decision[primary_mask],
    )


@dataclass
class SeverityResult:
    b: float
    n_draws: int
    n_primary_county_times: int
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
    B0: np.ndarray,
    gamma: float,
    b: float,
    base_seed,
    n_draws: int = DEFAULT_N_DRAWS,
    primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD,
    state: str = "kansas",
    trajectory_id: str = "primary",
) -> SeverityResult:
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise BoundedCouplingUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BoundedCouplingUncertaintyError(
            f"No county-times reach I >= {primary_threshold}; cannot run the experiment."
        )

    draws = 1 if b == 0.0 else n_draws
    B_hats = draw_B_hat(B0, b, draws, base_seed, state=state, trajectory_id=trajectory_id)

    true_parts, point_parts, lower_parts, upper_parts, decision_parts = [], [], [], [], []
    for k in range(B_hats.shape[0]):
        cmp = compare_draw(S, I, N, B0, B_hats[k], gamma, b, primary_mask)
        true_parts.append(cmp.true_drn)
        point_parts.append(cmp.point_drn)
        lower_parts.append(cmp.lower)
        upper_parts.append(cmp.upper)
        decision_parts.append(cmp.decision)

    true_all = np.concatenate(true_parts)
    point_all = np.concatenate(point_parts)
    lower_all = np.concatenate(lower_parts)
    upper_all = np.concatenate(upper_parts)
    decision_all = np.concatenate(decision_parts)

    defined = (~np.isnan(true_all)) & (~np.isnan(point_all))
    true_d = true_all[defined]
    point_d = point_all[defined]
    lower_d = lower_all[defined]
    upper_d = upper_all[defined]
    decision_d = decision_all[defined]

    containment = (lower_d <= true_d + 1e-9) & (true_d <= upper_d + 1e-9)

    certified_growth = decision_d == "certified_growth"
    certified_decline = decision_d == "certified_decline"
    indeterminate = decision_d == "indeterminate"

    false_certification = (certified_growth & (true_d <= 1.0)) | (certified_decline & (true_d >= 1.0))
    misclassification = (point_d - 1.0) * (true_d - 1.0) < 0.0

    abs_width = upper_d - lower_d
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(true_d != 0, abs_width / true_d, np.nan)

    def _mean(mask_arr):
        return float(mask_arr.mean()) if mask_arr.size else float("nan")

    return SeverityResult(
        b=b,
        n_draws=B_hats.shape[0],
        n_primary_county_times=n_primary,
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
    decision_arr: np.ndarray,
) -> dict:
    defined = (~np.isnan(true_arr)) & (~np.isnan(point_arr))
    n = int(defined.shape[0])
    if n == 0:
        return {field: (0 if field in ("n", "n_available") else float("nan")) for field in STAT_FIELDS}

    true_d = true_arr[defined]
    point_d = point_arr[defined]
    lower_d = lower_arr[defined]
    upper_d = upper_arr[defined]
    decision_d = decision_arr[defined]

    containment = (lower_d <= true_d + 1e-9) & (true_d <= upper_d + 1e-9)
    certified_growth = decision_d == "certified_growth"
    certified_decline = decision_d == "certified_decline"
    indeterminate = decision_d == "indeterminate"
    false_certification = (certified_growth & (true_d <= 1.0)) | (certified_decline & (true_d >= 1.0))
    misclassification = (point_d - 1.0) * (true_d - 1.0) < 0.0

    abs_width = upper_d - lower_d
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_width = np.where(true_d != 0, abs_width / true_d, np.nan)

    def _mean(mask_arr):
        return float(mask_arr.mean()) if mask_arr.size else float("nan")

    return {
        "n": n,
        "n_available": int(defined.sum()),
        "availability_rate": float(defined.mean()),
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


def verify_structural_preservation(B0: np.ndarray, B_hat: np.ndarray, b: float) -> bool:
    zero_mask = B0 == 0.0
    if not np.all(B_hat[zero_mask] == 0.0):
        return False
    nz = ~zero_mask
    lower = (1.0 - b) * B0[nz]
    upper = (1.0 + b) * B0[nz]
    return bool(np.all((B_hat[nz] >= lower - 1e-12) & (B_hat[nz] <= upper + 1e-12)))


def run_trajectory_severity(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B0: np.ndarray,
    gamma: float,
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
        raise BoundedCouplingUncertaintyError(f"I must be 2D (n_times, n_counties), got ndim={I.ndim}.")

    primary_mask = I >= primary_threshold
    n_primary = int(primary_mask.sum())
    if n_primary == 0:
        raise BoundedCouplingUncertaintyError(
            f"No county-times reach I >= {primary_threshold} for trajectory_id={trajectory_id!r}; "
            "cannot run the experiment."
        )
    I_primary = I[primary_mask]

    draws = 1 if b == 0.0 else n_draws
    B_hats = draw_B_hat(B0, b, draws, base_seed, state=state, trajectory_id=trajectory_id)

    pooled = {"true": [], "point": [], "lower": [], "upper": [], "decision": []}
    draw_rows: list[dict] = []

    for k in range(B_hats.shape[0]):
        seed = context_seed(base_seed, b, k, state, trajectory_id)
        if not verify_structural_preservation(B0, B_hats[k], b):
            raise BoundedCouplingUncertaintyError(
                f"draw {k} (b={b}) does not preserve B0's structural zeros or stay within "
                f"[(1-b)B0, (1+b)B0]; trajectory_id={trajectory_id!r}."
            )
        cmp = compare_draw(S, I, N, B0, B_hats[k], gamma, b, primary_mask)

        for (group_label, regime_label), mask in _strata(cmp.true_drn, I_primary, count_group_bounds, count_group_labels).items():
            stats = summarize_cells(cmp.true_drn[mask], cmp.point_drn[mask], cmp.lower[mask], cmp.upper[mask], cmp.decision[mask])
            draw_rows.append(
                {
                    "state": state,
                    "trajectory_id": trajectory_id,
                    "b": b,
                    "draw_index": k,
                    "seed": seed,
                    "count_group": group_label,
                    "regime": regime_label,
                    **stats,
                }
            )

        pooled["true"].append(cmp.true_drn)
        pooled["point"].append(cmp.point_drn)
        pooled["lower"].append(cmp.lower)
        pooled["upper"].append(cmp.upper)
        pooled["decision"].append(cmp.decision)

    pooled_true = np.concatenate(pooled["true"])
    pooled_point = np.concatenate(pooled["point"])
    pooled_lower = np.concatenate(pooled["lower"])
    pooled_upper = np.concatenate(pooled["upper"])
    pooled_decision = np.concatenate(pooled["decision"])
    pooled_I_primary = np.tile(I_primary, B_hats.shape[0])

    trajectory_rows: list[dict] = []
    for (group_label, regime_label), mask in _strata(pooled_true, pooled_I_primary, count_group_bounds, count_group_labels).items():
        stats = summarize_cells(pooled_true[mask], pooled_point[mask], pooled_lower[mask], pooled_upper[mask], pooled_decision[mask])
        trajectory_rows.append(
            {
                "state": state,
                "trajectory_id": trajectory_id,
                "b": b,
                "n_draws": int(B_hats.shape[0]),
                "count_group": group_label,
                "regime": regime_label,
                **stats,
            }
        )

    return draw_rows, trajectory_rows


UINT64_FIELDS = frozenset({"seed"})


def rows_to_columns(rows: list[dict], keys: list[str]) -> dict:
    columns = {}
    for k in keys:
        if k in UINT64_FIELDS:
            columns[k] = np.array([int(r[k]) for r in rows], dtype=np.uint64)
        else:
            columns[k] = np.array([r[k] for r in rows])
    return columns


def columns_to_rows(columns: dict, keys: list[str], n: int) -> list[dict]:
    out = []
    for i in range(n):
        row = {}
        for k in keys:
            v = columns[k][i]
            if isinstance(v, np.generic):
                v = v.item()
            elif isinstance(v, np.str_):
                v = str(v)
            row[k] = v
        out.append(row)
    return out
