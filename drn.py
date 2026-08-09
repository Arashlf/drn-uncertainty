from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import AVAILABILITY_THRESHOLDS as DEFAULT_THRESHOLDS
from config import PRIMARY_THRESHOLD as DEFAULT_PRIMARY_THRESHOLD


class DRNError(ValueError):
    pass


class IntervalDRNError(ValueError):
    pass


class UnderreportingError(ValueError):
    pass


@dataclass
class DRNResult:
    drn: np.ndarray
    defined_mask: np.ndarray
    primary_mask: np.ndarray
    primary_threshold: int


def _validate_inputs(
    S: np.ndarray, I: np.ndarray, N: np.ndarray, B: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    N = np.asarray(N, dtype=float)
    B = np.asarray(B, dtype=float)

    if S.ndim != 2:
        raise DRNError(f"S must be 2D (n_times, n_counties), got ndim={S.ndim}.")
    if S.shape != I.shape:
        raise DRNError(f"S and I must have the same shape, got {S.shape} and {I.shape}.")
    n = S.shape[1]
    if N.shape != (n,):
        raise DRNError(f"N must have shape ({n},), got {N.shape}.")
    if B.shape != (n, n):
        raise DRNError(f"B must have shape ({n}, {n}), got {B.shape}.")
    if not (np.isfinite(gamma) and gamma > 0):
        raise DRNError(f"gamma must be a finite positive scalar, got {gamma}.")
    if not (
        np.all(np.isfinite(S))
        and np.all(np.isfinite(I))
        and np.all(np.isfinite(N))
        and np.all(np.isfinite(B))
    ):
        raise DRNError("S, I, N, and B must all be finite.")
    if np.any(S < 0) or np.any(I < 0):
        raise DRNError("S and I must be nonnegative.")
    if np.any(N <= 0):
        raise DRNError("N must be strictly positive.")
    if np.any(B < 0):
        raise DRNError("B must be nonnegative.")
    return S, I, N, B


def compute_force(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) @ np.asarray(B, dtype=float).T


def compute_drn(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD,
) -> DRNResult:
    S, I, N, B = _validate_inputs(S, I, N, B, gamma)
    if primary_threshold <= 0:
        raise DRNError(f"primary_threshold must be positive, got {primary_threshold}.")

    s = S / N
    x = I / N
    force = compute_force(x, B)

    defined_mask = I > 0
    primary_mask = I >= primary_threshold

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = s * force / (gamma * x)
    drn = np.where(defined_mask, raw, np.nan)

    return DRNResult(
        drn=drn, defined_mask=defined_mask, primary_mask=primary_mask, primary_threshold=primary_threshold
    )


@dataclass
class AvailabilityStats:
    threshold: int
    fraction_county_times: float
    counties_ever_satisfying: int
    n_counties: int
    usable_timepoints_per_county: np.ndarray


def availability_at_threshold(I: np.ndarray, threshold: int) -> AvailabilityStats:
    I = np.asarray(I)
    mask = I >= threshold
    n_counties = I.shape[1]
    usable_timepoints_per_county = mask.sum(axis=0)
    counties_ever_satisfying = int(np.sum(usable_timepoints_per_county > 0))
    fraction_county_times = float(mask.sum() / mask.size)
    return AvailabilityStats(
        threshold=threshold,
        fraction_county_times=fraction_county_times,
        counties_ever_satisfying=counties_ever_satisfying,
        n_counties=n_counties,
        usable_timepoints_per_county=usable_timepoints_per_county,
    )


@dataclass
class PrimaryAvailabilityStats:
    threshold: int
    fraction_zero: float
    fraction_low_signal: float
    fraction_primary: float
    counties_ever_infected: int
    counties_ever_primary: int
    counties_never_infected: int
    counties_infected_never_primary: int
    peak_infection_per_county: np.ndarray
    first_infection_time_per_county: np.ndarray
    peak_infection_time_per_county: np.ndarray
    usable_timepoints_per_county: np.ndarray
    usable_duration_days_per_county: np.ndarray


def primary_availability(I: np.ndarray, t: np.ndarray, threshold: int = DEFAULT_PRIMARY_THRESHOLD) -> PrimaryAvailabilityStats:
    I = np.asarray(I)
    t = np.asarray(t)
    n_times, n_counties = I.shape

    zero_mask = I == 0
    low_signal_mask = (I > 0) & (I < threshold)
    primary_mask = I >= threshold

    fraction_zero = float(zero_mask.sum() / I.size)
    fraction_low_signal = float(low_signal_mask.sum() / I.size)
    fraction_primary = float(primary_mask.sum() / I.size)

    ever_infected = (I > 0).any(axis=0)
    ever_primary = primary_mask.any(axis=0)

    peak_infection_per_county = I.max(axis=0)
    first_infection_time_per_county = np.full(n_counties, np.nan)
    peak_infection_time_per_county = np.full(n_counties, np.nan)
    for i in range(n_counties):
        infected_idx = np.flatnonzero(I[:, i] > 0)
        if infected_idx.size:
            first_infection_time_per_county[i] = t[infected_idx[0]]
        peak_infection_time_per_county[i] = t[int(np.argmax(I[:, i]))]

    usable_timepoints_per_county = primary_mask.sum(axis=0)
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    usable_duration_days_per_county = usable_timepoints_per_county * dt

    return PrimaryAvailabilityStats(
        threshold=threshold,
        fraction_zero=fraction_zero,
        fraction_low_signal=fraction_low_signal,
        fraction_primary=fraction_primary,
        counties_ever_infected=int(ever_infected.sum()),
        counties_ever_primary=int(ever_primary.sum()),
        counties_never_infected=int((~ever_infected).sum()),
        counties_infected_never_primary=int((ever_infected & ~ever_primary).sum()),
        peak_infection_per_county=peak_infection_per_county,
        first_infection_time_per_county=first_infection_time_per_county,
        peak_infection_time_per_county=peak_infection_time_per_county,
        usable_timepoints_per_county=usable_timepoints_per_county,
        usable_duration_days_per_county=usable_duration_days_per_county,
    )


@dataclass
class IntervalDRNResult:
    lower: np.ndarray
    upper: np.ndarray
    defined_mask: np.ndarray
    unbounded_upper_mask: np.ndarray
    decision: np.ndarray


def _broadcast_gamma_interval(gamma, n: int, name: str) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 0:
        return np.full(n, float(gamma))
    if gamma.shape != (n,):
        raise IntervalDRNError(f"{name} must be a scalar or have shape ({n},), got {gamma.shape}.")
    return gamma


def _validate_interval(
    s_lower: np.ndarray,
    s_upper: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    B_lower: np.ndarray,
    B_upper: np.ndarray,
    gamma_lower,
    gamma_upper,
):
    s_lower = np.asarray(s_lower, dtype=float)
    s_upper = np.asarray(s_upper, dtype=float)
    x_lower = np.asarray(x_lower, dtype=float)
    x_upper = np.asarray(x_upper, dtype=float)
    B_lower = np.asarray(B_lower, dtype=float)
    B_upper = np.asarray(B_upper, dtype=float)

    if s_lower.ndim != 2:
        raise IntervalDRNError(f"s_lower must be 2D (n_times, n_counties), got ndim={s_lower.ndim}.")
    if not (s_lower.shape == s_upper.shape == x_lower.shape == x_upper.shape):
        raise IntervalDRNError(
            "s_lower, s_upper, x_lower, x_upper must share the same shape; got "
            f"{s_lower.shape}, {s_upper.shape}, {x_lower.shape}, {x_upper.shape}."
        )
    n = s_lower.shape[1]
    if B_lower.shape != (n, n) or B_upper.shape != (n, n):
        raise IntervalDRNError(
            f"B_lower and B_upper must have shape ({n}, {n}), got {B_lower.shape} and {B_upper.shape}."
        )

    gamma_lower = _broadcast_gamma_interval(gamma_lower, n, "gamma_lower")
    gamma_upper = _broadcast_gamma_interval(gamma_upper, n, "gamma_upper")

    all_arrays = (s_lower, s_upper, x_lower, x_upper, B_lower, B_upper, gamma_lower, gamma_upper)
    if not all(np.all(np.isfinite(a)) for a in all_arrays):
        raise IntervalDRNError("All box endpoints must be finite.")
    if (
        np.any(s_lower > s_upper)
        or np.any(x_lower > x_upper)
        or np.any(B_lower > B_upper)
        or np.any(gamma_lower > gamma_upper)
    ):
        raise IntervalDRNError("Lower endpoints must not exceed upper endpoints.")
    if np.any(s_lower < 0) or np.any(s_upper > 1):
        raise IntervalDRNError("s must be nonnegative and no greater than one.")
    if np.any(x_lower < 0):
        raise IntervalDRNError("x must be nonnegative.")
    if np.any(B_lower < 0):
        raise IntervalDRNError("B must be nonnegative.")
    if np.any(gamma_lower <= 0):
        raise IntervalDRNError("gamma_lower must be strictly positive.")

    return s_lower, s_upper, x_lower, x_upper, B_lower, B_upper, gamma_lower, gamma_upper


def compute_interval_drn(
    s_lower: np.ndarray,
    s_upper: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    B_lower: np.ndarray,
    B_upper: np.ndarray,
    gamma_lower,
    gamma_upper,
) -> IntervalDRNResult:
    s_lower, s_upper, x_lower, x_upper, B_lower, B_upper, gamma_lower, gamma_upper = _validate_interval(
        s_lower, s_upper, x_lower, x_upper, B_lower, B_upper, gamma_lower, gamma_upper
    )
    n = s_lower.shape[1]
    diag_idx = np.arange(n)

    B_lower_diag = B_lower[diag_idx, diag_idx]
    B_upper_diag = B_upper[diag_idx, diag_idx]

    B_lower_off = B_lower.copy()
    B_lower_off[diag_idx, diag_idx] = 0.0
    B_upper_off = B_upper.copy()
    B_upper_off[diag_idx, diag_idx] = 0.0

    ext_lower_numerator = x_lower @ B_lower_off.T
    ext_upper_numerator = x_upper @ B_upper_off.T

    defined_mask = x_upper > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        ext_lower_term = ext_lower_numerator / x_upper
    lower = np.where(
        defined_mask,
        (s_lower / gamma_upper) * (B_lower_diag[np.newaxis, :] + ext_lower_term),
        np.nan,
    )

    x_lower_is_zero = x_lower == 0
    ext_has_positive_upper = ext_upper_numerator > 0
    unbounded_upper_mask = defined_mask & x_lower_is_zero & ext_has_positive_upper

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_denominator = np.where(x_lower_is_zero, 1.0, x_lower)
        ext_upper_term = ext_upper_numerator / safe_denominator
    ext_upper_term = np.where(unbounded_upper_mask, np.inf, ext_upper_term)

    upper = np.where(
        defined_mask,
        (s_upper / gamma_lower) * (B_upper_diag[np.newaxis, :] + ext_upper_term),
        np.nan,
    )

    decision = np.full(s_lower.shape, "undefined", dtype="<U20")
    growth = defined_mask & (lower > 1.0)
    decline = defined_mask & (upper < 1.0)
    indeterminate = defined_mask & ~growth & ~decline
    decision[growth] = "certified_growth"
    decision[decline] = "certified_decline"
    decision[indeterminate] = "indeterminate"

    return IntervalDRNResult(
        lower=lower,
        upper=upper,
        defined_mask=defined_mask,
        unbounded_upper_mask=unbounded_upper_mask,
        decision=decision,
    )


def _validate_x_B(x: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    B = np.asarray(B, dtype=float)
    if x.ndim != 2:
        raise UnderreportingError(f"x must be 2D (n_times, n_counties), got ndim={x.ndim}.")
    n = x.shape[1]
    if B.shape != (n, n):
        raise UnderreportingError(f"B must have shape ({n}, {n}), got {B.shape}.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(B)):
        raise UnderreportingError("x and B must be finite.")
    if np.any(x < 0):
        raise UnderreportingError("x must be nonnegative.")
    if np.any(B < 0):
        raise UnderreportingError("B must be nonnegative.")
    return x, B


def _validate_a(a: float) -> None:
    if not (np.isfinite(a) and 0 < a <= 1):
        raise UnderreportingError(f"a must satisfy 0 < a <= 1, got {a}.")


def _broadcast_gamma_underreporting(gamma, n: int) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim == 0:
        gamma = np.full(n, float(gamma))
    elif gamma.shape != (n,):
        raise UnderreportingError(f"gamma must be a scalar or have shape ({n},), got {gamma.shape}.")
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise UnderreportingError("gamma must be finite and strictly positive.")
    return gamma


@dataclass
class WeightsResult:
    p_ii: np.ndarray
    q_i: np.ndarray
    defined_mask: np.ndarray


def compute_weights(x: np.ndarray, B: np.ndarray) -> WeightsResult:
    x, B = _validate_x_B(x, B)
    n = x.shape[1]

    force = x @ B.T
    self_numerator = x * np.diag(B)[np.newaxis, :]

    defined_mask = force > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ii = np.where(defined_mask, self_numerator / force, np.nan)
        q_i = np.where(defined_mask, 1.0 - p_ii, np.nan)

    return WeightsResult(p_ii=p_ii, q_i=q_i, defined_mask=defined_mask)


def full_weights(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    x, B = _validate_x_B(x, B)
    force = x @ B.T
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (x[:, np.newaxis, :] * B[np.newaxis, :, :]) / force[:, :, np.newaxis]
    return np.where(force[:, :, np.newaxis] > 0, p, np.nan)


def ratio_hat_over_true(
    x: np.ndarray, x_tilde: np.ndarray, B: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x, B = _validate_x_B(x, B)
    x_tilde, _ = _validate_x_B(x_tilde, B)
    if x.shape != x_tilde.shape:
        raise UnderreportingError(f"x and x_tilde must share shape, got {x.shape} and {x_tilde.shape}.")
    if np.any(x_tilde > x + 1e-9):
        raise UnderreportingError("x_tilde must not exceed x (underreporting only reduces counts).")

    force_true = x @ B.T
    force_obs = x_tilde @ B.T

    defined = (x > 0) & (x_tilde > 0) & (force_true > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hat_shape = force_obs / x_tilde
        true_shape = force_true / x
        ratio = np.where(defined, hat_shape / true_shape, np.nan)

    return ratio, defined


def sensitivity_bounds(q: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    _validate_a(a)
    q = np.asarray(q, dtype=float)
    if np.any((~np.isnan(q)) & ((q < 0) | (q > 1))):
        raise UnderreportingError("q must lie in [0, 1] wherever defined.")

    lower = 1.0 - q * (1.0 - a)
    upper = 1.0 + q * (1.0 / a - 1.0)
    return lower, upper


@dataclass
class ObservableCertificateResult:
    hat_drn: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    q_hat: np.ndarray
    defined_mask: np.ndarray
    decision: np.ndarray


def observable_certificate(
    s: np.ndarray, x_tilde: np.ndarray, B: np.ndarray, gamma, a: float
) -> ObservableCertificateResult:
    _validate_a(a)
    s = np.asarray(s, dtype=float)
    x_tilde, B = _validate_x_B(x_tilde, B)
    if s.shape != x_tilde.shape:
        raise UnderreportingError(f"s must have the same shape as x_tilde, got {s.shape} vs {x_tilde.shape}.")
    if np.any(s < 0) or np.any(s > 1):
        raise UnderreportingError("s must be nonnegative and no greater than one.")

    gamma_arr = _broadcast_gamma_underreporting(gamma, x_tilde.shape[1])

    weights = compute_weights(x_tilde, B)
    force_obs = x_tilde @ B.T

    defined_mask = x_tilde > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        hat_drn = np.where(
            defined_mask, s * force_obs / (gamma_arr[np.newaxis, :] * x_tilde), np.nan
        )

    q_hat_safe = np.where(weights.defined_mask, weights.q_i, 0.0)
    lower_factor, upper_factor = sensitivity_bounds(q_hat_safe, a)

    lower = np.where(defined_mask, hat_drn * lower_factor, np.nan)
    upper = np.where(defined_mask, hat_drn * upper_factor, np.nan)

    decision = np.full(x_tilde.shape, "undefined", dtype="<U20")
    growth = defined_mask & (lower > 1.0)
    decline = defined_mask & (upper < 1.0)
    indeterminate = defined_mask & ~growth & ~decline
    decision[growth] = "certified_growth"
    decision[decline] = "certified_decline"
    decision[indeterminate] = "indeterminate"

    return ObservableCertificateResult(
        hat_drn=hat_drn,
        lower=lower,
        upper=upper,
        q_hat=weights.q_i,
        defined_mask=defined_mask,
        decision=decision,
    )


@dataclass
class ReferenceDRNResult:
    drn: np.ndarray
    defined_mask: np.ndarray
    primary_mask: np.ndarray
    availability_thresholds: np.ndarray
    availability_masks: np.ndarray
    p_ii: np.ndarray
    q: np.ndarray
    q_hat: np.ndarray
    q_defined_mask: np.ndarray


def compute_reference_drn(
    S: np.ndarray,
    I: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    primary_threshold: int,
    availability_thresholds,
) -> ReferenceDRNResult:
    drn_result = compute_drn(S, I, N, B, gamma, primary_threshold=primary_threshold)

    x = np.asarray(I, dtype=float) / np.asarray(N, dtype=float)
    weights = compute_weights(x, B)

    thresholds = np.asarray(sorted(set(int(t) for t in availability_thresholds)), dtype=int)
    I_arr = np.asarray(I)
    availability_masks = np.stack([I_arr >= th for th in thresholds], axis=0)

    return ReferenceDRNResult(
        drn=drn_result.drn,
        defined_mask=drn_result.defined_mask,
        primary_mask=drn_result.primary_mask,
        availability_thresholds=thresholds,
        availability_masks=availability_masks,
        p_ii=weights.p_ii,
        q=weights.q_i,
        q_hat=weights.q_i,
        q_defined_mask=weights.defined_mask,
    )


@dataclass
class ReferenceDRNValidation:
    ok: bool
    reason: str


def validate_reference_drn(result: ReferenceDRNResult, I: np.ndarray, primary_threshold: int) -> ReferenceDRNValidation:
    I = np.asarray(I)
    n_times, n = I.shape

    for name, arr in (
        ("drn", result.drn),
        ("defined_mask", result.defined_mask),
        ("primary_mask", result.primary_mask),
        ("p_ii", result.p_ii),
        ("q", result.q),
        ("q_hat", result.q_hat),
        ("q_defined_mask", result.q_defined_mask),
    ):
        if arr.shape != (n_times, n):
            return ReferenceDRNValidation(False, f"{name} shape {arr.shape} != I shape {(n_times, n)}")

    if result.availability_masks.shape != (len(result.availability_thresholds), n_times, n):
        return ReferenceDRNValidation(
            False,
            f"availability_masks shape {result.availability_masks.shape} does not match "
            f"(n_thresholds={len(result.availability_thresholds)}, n_times={n_times}, n_counties={n})",
        )

    defined = result.defined_mask
    if not np.all(np.isfinite(result.drn[defined])):
        return ReferenceDRNValidation(False, "drn contains a non-finite value on a defined (I > 0) cell")
    if not np.all(np.isnan(result.drn[~defined])):
        return ReferenceDRNValidation(False, "drn is not NaN on every undefined (I == 0) cell")
    if np.any(result.drn[defined] < 0):
        return ReferenceDRNValidation(False, "drn contains a negative value on a defined cell")

    if not np.array_equal(defined, I > 0):
        return ReferenceDRNValidation(False, "defined_mask does not exactly equal I > 0")
    if not np.array_equal(result.primary_mask, I >= primary_threshold):
        return ReferenceDRNValidation(False, f"primary_mask does not exactly equal I >= {primary_threshold}")

    for k, th in enumerate(result.availability_thresholds):
        if not np.array_equal(result.availability_masks[k], I >= th):
            return ReferenceDRNValidation(False, f"availability_masks[{int(th)}] does not exactly equal I >= {int(th)}")

    q_defined = result.q_defined_mask
    for name, arr in (("p_ii", result.p_ii), ("q", result.q), ("q_hat", result.q_hat)):
        if not np.all(np.isfinite(arr[q_defined])):
            return ReferenceDRNValidation(False, f"{name} contains a non-finite value on a defined (Bx > 0) cell")
        if not np.all(np.isnan(arr[~q_defined])):
            return ReferenceDRNValidation(False, f"{name} is not NaN on every undefined (Bx == 0) cell")
        if np.any(arr[q_defined] < 0) or np.any(arr[q_defined] > 1):
            return ReferenceDRNValidation(False, f"{name} is outside [0, 1] on a defined cell")

    return ReferenceDRNValidation(True, "ok")
