from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from config import DT_DAYS, EXPERIMENT_TRUTH_TRAJECTORY, HORIZON_DAYS, N_SEED_INFECTED, derive_seed

DEFAULT_T_SPAN = (0.0, 180.0)
DEFAULT_DT_OUTPUT = 0.25
DEFAULT_N_SEED_INFECTED_DETERMINISTIC = 10

DEFAULT_DT = DT_DAYS
DEFAULT_HORIZON = HORIZON_DAYS
DEFAULT_N_SEED_INFECTED_STOCHASTIC = N_SEED_INFECTED

SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12


class DeterministicSIRError(ValueError):
    pass


class StochasticSIRError(ValueError):
    pass


def find_seed_index(counties: list[str], county_name: str) -> int:
    try:
        return counties.index(county_name)
    except ValueError:
        raise ValueError(
            f"Seed county '{county_name}' not found among {len(counties)} counties."
        ) from None


def initial_conditions(
    N: np.ndarray, seed_index: int, n_seed_infected: float = DEFAULT_N_SEED_INFECTED_DETERMINISTIC
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.asarray(N, dtype=float)
    n = N.shape[0]

    if not (0 <= seed_index < n):
        raise DeterministicSIRError(f"seed_index {seed_index} out of range for {n} counties.")
    if not (n_seed_infected > 0):
        raise DeterministicSIRError(f"n_seed_infected must be positive, got {n_seed_infected}.")
    if n_seed_infected > N[seed_index]:
        raise DeterministicSIRError(
            f"n_seed_infected={n_seed_infected} exceeds population "
            f"{N[seed_index]} of seed county index {seed_index}."
        )

    x0 = np.zeros(n)
    x0[seed_index] = n_seed_infected / N[seed_index]
    s0 = 1.0 - x0
    r0 = np.zeros(n)
    return s0, x0, r0


def sir_derivatives(t: float, y: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    n = B.shape[0]
    s, x = y[:n], y[n : 2 * n]
    force = B @ x
    ds = -s * force
    dx = s * force - gamma * x
    dr = gamma * x
    return np.concatenate([ds, dx, dr])


@dataclass
class SIRResult:
    t: np.ndarray
    S: np.ndarray
    X: np.ndarray
    R: np.ndarray
    counties: list[str] | None
    seed_index: int
    gamma: float
    n_seed_infected: float


def simulate_deterministic(
    B: np.ndarray,
    N: np.ndarray,
    gamma: float,
    seed_index: int,
    n_seed_infected: float = DEFAULT_N_SEED_INFECTED_DETERMINISTIC,
    t_span: tuple[float, float] = DEFAULT_T_SPAN,
    dt_output: float = DEFAULT_DT_OUTPUT,
    counties: list[str] | None = None,
) -> SIRResult:
    B = np.asarray(B, dtype=float)
    N = np.asarray(N, dtype=float)
    n = B.shape[0]

    if B.shape != (n, n):
        raise DeterministicSIRError(f"B must be square, got shape {B.shape}.")
    if N.shape != (n,):
        raise DeterministicSIRError(f"N must have shape ({n},), got {N.shape}.")
    if not (np.isfinite(gamma) and gamma > 0):
        raise DeterministicSIRError(f"gamma must be finite and positive, got {gamma}.")
    t0, t1 = t_span
    if not (t1 > t0):
        raise DeterministicSIRError(f"t_span must have end > start, got {t_span}.")
    if not (dt_output > 0):
        raise DeterministicSIRError(f"dt_output must be positive, got {dt_output}.")

    s0, x0, r0 = initial_conditions(N, seed_index, n_seed_infected)
    y0 = np.concatenate([s0, x0, r0])

    n_steps = round((t1 - t0) / dt_output)
    t_eval = t0 + dt_output * np.arange(n_steps + 1)

    sol = solve_ivp(
        sir_derivatives,
        t_span=(t0, t1),
        y0=y0,
        t_eval=t_eval,
        args=(B, gamma),
        method="RK45",
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
    )
    if not sol.success:
        raise DeterministicSIRError(f"solve_ivp failed: {sol.message}")

    S = sol.y[0:n, :].T
    X = sol.y[n : 2 * n, :].T
    R = sol.y[2 * n : 3 * n, :].T

    return SIRResult(
        t=sol.t,
        S=S,
        X=X,
        R=R,
        counties=counties,
        seed_index=seed_index,
        gamma=gamma,
        n_seed_infected=n_seed_infected,
    )


def _validate_population_counts(N: np.ndarray) -> np.ndarray:
    N = np.asarray(N, dtype=float)
    if not np.all(np.isfinite(N)):
        raise StochasticSIRError("N contains non-finite values.")
    if np.any(N <= 0):
        raise StochasticSIRError("N must contain strictly positive populations.")
    if not np.allclose(N, np.round(N)):
        raise StochasticSIRError("N must contain integer county populations.")
    return np.round(N).astype(np.int64)


def initial_state_counts(
    N: np.ndarray, seed_index: int, n_seed_infected: int = DEFAULT_N_SEED_INFECTED_STOCHASTIC
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = _validate_population_counts(N)
    n = N.shape[0]

    if not (0 <= seed_index < n):
        raise StochasticSIRError(f"seed_index {seed_index} out of range for {n} counties.")
    if n_seed_infected <= 0 or n_seed_infected != int(n_seed_infected):
        raise StochasticSIRError(
            f"n_seed_infected must be a positive integer, got {n_seed_infected}."
        )
    n_seed_infected = int(n_seed_infected)
    if n_seed_infected > N[seed_index]:
        raise StochasticSIRError(
            f"n_seed_infected={n_seed_infected} exceeds population "
            f"{N[seed_index]} of seed county index {seed_index}."
        )

    I0 = np.zeros(n, dtype=np.int64)
    I0[seed_index] = n_seed_infected
    S0 = N - I0
    R0 = np.zeros(n, dtype=np.int64)
    return S0, I0, R0


def step(
    S: np.ndarray,
    I: np.ndarray,
    R: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    gamma: float,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = I / N
    force = B @ x
    p_infection = 1.0 - np.exp(-force * dt)
    p_recovery = 1.0 - np.exp(-gamma * dt)

    if np.any(p_infection < 0) or np.any(p_infection > 1) or not (0.0 <= p_recovery <= 1.0):
        raise StochasticSIRError(
            "Computed transition probability outside [0, 1]; check B, gamma, dt."
        )

    new_infections = rng.binomial(S, p_infection)
    new_recoveries = rng.binomial(I, p_recovery)

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    return S_new, I_new, R_new


@dataclass
class StochasticSIRResult:
    t: np.ndarray
    S: np.ndarray
    I: np.ndarray
    R: np.ndarray
    counties: list[str] | None
    seed_index: int
    seed: int
    gamma: float
    dt: float
    n_seed_infected: int
    stopped_at: int


def simulate_stochastic(
    B: np.ndarray,
    N: np.ndarray,
    gamma: float,
    seed_index: int,
    seed: int,
    n_seed_infected: int = DEFAULT_N_SEED_INFECTED_STOCHASTIC,
    dt: float = DEFAULT_DT,
    horizon: float = DEFAULT_HORIZON,
    counties: list[str] | None = None,
) -> StochasticSIRResult:
    B = np.asarray(B, dtype=float)
    n = B.shape[0]

    if B.shape != (n, n):
        raise StochasticSIRError(f"B must be square, got shape {B.shape}.")
    if not np.all(np.isfinite(B)):
        raise StochasticSIRError("B contains non-finite values.")
    if np.any(B < 0):
        raise StochasticSIRError("B contains negative entries.")

    N = _validate_population_counts(N)
    if N.shape != (n,):
        raise StochasticSIRError(f"N must have shape ({n},), got {N.shape}.")
    if not (np.isfinite(gamma) and gamma > 0):
        raise StochasticSIRError(f"gamma must be finite and positive, got {gamma}.")
    if not (dt > 0):
        raise StochasticSIRError(f"dt must be positive, got {dt}.")
    if not (horizon > 0):
        raise StochasticSIRError(f"horizon must be positive, got {horizon}.")
    if seed is None:
        raise StochasticSIRError("seed must be provided explicitly.")

    rng = np.random.default_rng(seed)

    n_steps = round(horizon / dt)
    t = dt * np.arange(n_steps + 1)

    S = np.empty((n_steps + 1, n), dtype=np.int64)
    I = np.empty((n_steps + 1, n), dtype=np.int64)
    R = np.empty((n_steps + 1, n), dtype=np.int64)
    S[0], I[0], R[0] = initial_state_counts(N, seed_index, n_seed_infected)

    stopped_at = n_steps
    for k in range(n_steps):
        if I[k].sum() == 0:
            S[k:] = S[k]
            I[k:] = I[k]
            R[k:] = R[k]
            stopped_at = k
            break
        S[k + 1], I[k + 1], R[k + 1] = step(S[k], I[k], R[k], N, B, gamma, dt, rng)

    return StochasticSIRResult(
        t=t,
        S=S,
        I=I,
        R=R,
        counties=counties,
        seed_index=seed_index,
        seed=seed,
        gamma=gamma,
        dt=dt,
        n_seed_infected=n_seed_infected,
        stopped_at=stopped_at,
    )


@dataclass
class EnsembleResult:
    t: np.ndarray
    S_frac: np.ndarray
    I_frac: np.ndarray
    R_frac: np.ndarray


def run_stochastic_ensemble(
    B: np.ndarray,
    N: np.ndarray,
    gamma: float,
    seed_index: int,
    n_seed_infected: int,
    dt: float,
    horizon: float,
    n_realizations: int,
    base_seed: int,
) -> EnsembleResult:
    total_N = float(np.sum(N))
    child_seeds = np.random.SeedSequence(base_seed).spawn(n_realizations)

    n_steps = round(horizon / dt)
    S_frac = np.empty((n_realizations, n_steps + 1))
    I_frac = np.empty((n_realizations, n_steps + 1))
    R_frac = np.empty((n_realizations, n_steps + 1))
    t = None

    for k, child_seed in enumerate(child_seeds):
        result = simulate_stochastic(
            B,
            N,
            gamma,
            seed_index=seed_index,
            seed=child_seed,
            n_seed_infected=n_seed_infected,
            dt=dt,
            horizon=horizon,
        )
        if t is None:
            t = result.t
        S_frac[k] = result.S.sum(axis=1) / total_N
        I_frac[k] = result.I.sum(axis=1) / total_N
        R_frac[k] = result.R.sum(axis=1) / total_N

    return EnsembleResult(t=t, S_frac=S_frac, I_frac=I_frac, R_frac=R_frac)


def deterministic_total_fractions(
    B: np.ndarray,
    N: np.ndarray,
    gamma: float,
    seed_index: int,
    n_seed_infected: int,
    dt: float,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_N = float(np.sum(N))
    result = simulate_deterministic(
        B,
        N,
        gamma,
        seed_index=seed_index,
        n_seed_infected=n_seed_infected,
        t_span=(0.0, horizon),
        dt_output=dt,
    )
    S_total = (result.S @ N) / total_N
    I_total = (result.X @ N) / total_N
    R_total = (result.R @ N) / total_N
    return result.t, S_total, I_total, R_total


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def peak_value(y: np.ndarray) -> float:
    return float(np.max(y))


def peak_time(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.asarray(t)[np.argmax(y)])


@dataclass
class ComparisonMetrics:
    rmse: float
    max_abs_diff: float
    peak_diff: float
    peak_time_diff: float
    final_recovered_diff: float


def compare_to_deterministic(
    t: np.ndarray,
    I_det: np.ndarray,
    I_ens_mean: np.ndarray,
    R_det: np.ndarray,
    R_ens_mean: np.ndarray,
) -> ComparisonMetrics:
    return ComparisonMetrics(
        rmse=rmse(I_ens_mean, I_det),
        max_abs_diff=max_abs_diff(I_ens_mean, I_det),
        peak_diff=abs(peak_value(I_ens_mean) - peak_value(I_det)),
        peak_time_diff=abs(peak_time(t, I_ens_mean) - peak_time(t, I_det)),
        final_recovered_diff=abs(float(R_ens_mean[-1]) - float(R_det[-1])),
    )


def trajectory_seed(
    base_seed, state: str, seed_county: str, realization: int, trajectory_id: str = "truth"
) -> int:
    return derive_seed(
        base_seed, state, trajectory_id, EXPERIMENT_TRUTH_TRAJECTORY, seed_county, "realization", realization
    )


@dataclass
class TrajectoryValidation:
    ok: bool
    reason: str


def validate_trajectory(
    t: np.ndarray, S: np.ndarray, I: np.ndarray, R: np.ndarray, N: np.ndarray
) -> TrajectoryValidation:
    t = np.asarray(t)
    S = np.asarray(S)
    I = np.asarray(I)
    R = np.asarray(R)
    N = np.asarray(N)

    if S.shape != I.shape or S.shape != R.shape:
        return TrajectoryValidation(False, f"shape mismatch: S={S.shape}, I={I.shape}, R={R.shape}")
    if S.ndim != 2:
        return TrajectoryValidation(False, f"expected 2D (n_times, n_counties), got ndim={S.ndim}")
    n_times, n = S.shape
    if t.shape != (n_times,):
        return TrajectoryValidation(False, f"t shape {t.shape} does not match n_times={n_times}")
    if N.shape != (n,):
        return TrajectoryValidation(False, f"N shape {N.shape} does not match n_counties={n}")

    if not (
        np.all(np.isfinite(t))
        and np.all(np.isfinite(S))
        and np.all(np.isfinite(I))
        and np.all(np.isfinite(R))
        and np.all(np.isfinite(N))
    ):
        return TrajectoryValidation(False, "non-finite values present")

    for name, arr in (("S", S), ("I", I), ("R", R)):
        if not np.allclose(arr, np.round(arr)):
            return TrajectoryValidation(False, f"{name} contains non-integer values")
        if np.any(arr < 0):
            return TrajectoryValidation(False, f"{name} contains negative values")

    totals = S + I + R
    if not np.all(totals == N[np.newaxis, :]):
        max_dev = float(np.max(np.abs(totals - N[np.newaxis, :])))
        return TrajectoryValidation(False, f"population not conserved (max|S+I+R-N|={max_dev})")

    if np.any(np.diff(S, axis=0) > 0):
        return TrajectoryValidation(False, "S is not monotonically non-increasing")
    if np.any(np.diff(R, axis=0) < 0):
        return TrajectoryValidation(False, "R is not monotonically non-decreasing")

    return TrajectoryValidation(True, "ok")


def validate_resume_metadata(saved: dict, expected: dict) -> TrajectoryValidation:
    for key, expected_value in expected.items():
        saved_value = saved.get(key)
        if isinstance(expected_value, np.ndarray):
            mismatch = saved_value is None or not np.array_equal(saved_value, expected_value)
        else:
            mismatch = saved_value != expected_value
        if mismatch:
            return TrajectoryValidation(False, f"{key} mismatch (saved={saved_value!r}, expected={expected_value!r})")
    return TrajectoryValidation(True, "ok")
