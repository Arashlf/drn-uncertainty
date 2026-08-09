from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED_QUANTILES as DEFAULT_QUANTILES

ROW_SUM_ATOL = 1e-8

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


class MobilityDataError(ValueError):
    pass


class SeedSelectionError(ValueError):
    pass


def data_paths(state: str, data_root: Path | str | None = None) -> tuple[Path, Path]:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    state_dir = root / state
    edges_path = state_dir / f"commuting_edges_{state}.csv"
    pop_path = state_dir / f"population_{state}.csv"
    return edges_path, pop_path


def load_state_inputs(
    state: str, data_root: Path | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges_path, pop_path = data_paths(state, data_root)
    if not edges_path.exists():
        raise MobilityDataError(f"Commuting edges file not found: {edges_path}")
    if not pop_path.exists():
        raise MobilityDataError(f"Population file not found: {pop_path}")

    edges_df = pd.read_csv(edges_path)
    pop_df = pd.read_csv(pop_path)

    required_edge_cols = {"source", "target", "weight"}
    if not required_edge_cols.issubset(edges_df.columns):
        raise MobilityDataError(
            f"{edges_path} must have columns {sorted(required_edge_cols)}, "
            f"got {list(edges_df.columns)}"
        )
    required_pop_cols = {"county", "population"}
    if not required_pop_cols.issubset(pop_df.columns):
        raise MobilityDataError(
            f"{pop_path} must have columns {sorted(required_pop_cols)}, "
            f"got {list(pop_df.columns)}"
        )

    return edges_df, pop_df


def align_counties(
    edges_df: pd.DataFrame, pop_df: pd.DataFrame
) -> tuple[list[str], np.ndarray, np.ndarray]:
    counties = pop_df["county"].tolist()

    duplicates = pop_df["county"][pop_df["county"].duplicated()].unique().tolist()
    if duplicates:
        raise MobilityDataError(f"Duplicate counties in population data: {duplicates}")

    N = pop_df["population"].to_numpy(dtype=float)
    if not np.all(np.isfinite(N)):
        raise MobilityDataError("Population data contains non-finite values.")
    if not np.all(N > 0):
        bad = [c for c, n in zip(counties, N) if not (n > 0)]
        raise MobilityDataError(f"Non-positive population for counties: {bad}")

    index = {county: i for i, county in enumerate(counties)}

    unknown_sources = sorted(set(edges_df["source"]) - set(index))
    unknown_targets = sorted(set(edges_df["target"]) - set(index))
    unknown = sorted(set(unknown_sources) | set(unknown_targets))
    if unknown:
        raise MobilityDataError(
            f"Commuting edges reference counties absent from population data: {unknown}"
        )

    weights = edges_df["weight"].to_numpy(dtype=float)
    if not np.all(np.isfinite(weights)):
        raise MobilityDataError("Commuting weights contain non-finite values.")
    if np.any(weights < 0):
        raise MobilityDataError("Commuting weights must be nonnegative.")

    dup_mask = edges_df.duplicated(subset=["source", "target"], keep=False)
    if dup_mask.any():
        dup_pairs = (
            edges_df.loc[dup_mask, ["source", "target"]].drop_duplicates().to_records(index=False)
        )
        raise MobilityDataError(
            f"Duplicate (source, target) commuting edges would silently overwrite "
            f"each other; aggregate them first. Duplicated pairs: {list(dup_pairs)}"
        )

    n = len(counties)
    W = np.zeros((n, n), dtype=float)
    src_idx = edges_df["source"].map(index).to_numpy()
    tgt_idx = edges_df["target"].map(index).to_numpy()
    W[src_idx, tgt_idx] = weights

    return counties, W, N


def presence_matrix(
    W: np.ndarray,
    N: np.ndarray,
    theta: float = 1 / 3,
    county_labels: list[str] | None = None,
) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    N = np.asarray(N, dtype=float)
    n = W.shape[0]

    if W.shape != (n, n):
        raise MobilityDataError(f"W must be square, got shape {W.shape}.")
    if N.shape != (n,):
        raise MobilityDataError(f"N must have shape ({n},), got {N.shape}.")
    if not (0 < theta <= 1):
        raise MobilityDataError(f"theta must be in (0, 1], got {theta}.")
    if not np.all(np.isfinite(W)):
        raise MobilityDataError("W contains non-finite values.")
    if not np.all(np.isfinite(N)):
        raise MobilityDataError("N contains non-finite values.")
    if np.any(W < 0):
        raise MobilityDataError("W contains negative entries.")
    if not np.all(N > 0):
        raise MobilityDataError("N contains non-positive entries.")

    labels = county_labels if county_labels is not None else list(range(n))

    P = theta * W / N[:, None]
    np.fill_diagonal(P, 0.0)

    off_diag_row_sums = P.sum(axis=1)
    over = np.where(off_diag_row_sums > 1 + ROW_SUM_ATOL)[0]
    if over.size:
        offenders = [(labels[i], float(off_diag_row_sums[i])) for i in over]
        raise MobilityDataError(
            "Off-diagonal presence row sum exceeds 1 before diagonal is set "
            f"(theta={theta}); inspect commuting/population inputs, do not "
            f"renormalize. Offending counties (label, row_sum): {offenders}"
        )

    np.fill_diagonal(P, 1.0 - off_diag_row_sums)

    _check_finite_nonneg(P, "P")
    _check_row_sums(P, 1.0, "P")
    return P


def population_present(N: np.ndarray, P: np.ndarray) -> np.ndarray:
    N = np.asarray(N, dtype=float)
    P = np.asarray(P, dtype=float)
    D = N @ P
    if not np.all(np.isfinite(D)):
        raise MobilityDataError("D contains non-finite values.")
    if not np.all(D > 0):
        raise MobilityDataError(f"D must be strictly positive everywhere, got {D}.")
    return D


def mixing_matrix(P: np.ndarray, N: np.ndarray, D: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)

    if not np.all(D > 0):
        raise MobilityDataError("D must be strictly positive to build C (division by D).")

    C = (P / D[np.newaxis, :]) @ P.T * N[np.newaxis, :]

    _check_finite_nonneg(C, "C")
    _check_row_sums(C, 1.0, "C")
    return C


def transmission_matrix(C: np.ndarray, beta0: float) -> np.ndarray:
    if not (np.isfinite(beta0) and beta0 > 0):
        raise MobilityDataError(f"beta0 must be a finite positive number, got {beta0}.")

    C = np.asarray(C, dtype=float)
    B = beta0 * C

    _check_finite_nonneg(B, "B")
    _check_row_sums(B, beta0, "B")
    return B


@dataclass
class MobilityCoupling:
    counties: list[str]
    W: np.ndarray
    N: np.ndarray
    P: np.ndarray
    D: np.ndarray
    C: np.ndarray
    B: np.ndarray
    theta: float
    beta0: float


def build_transmission_matrix(
    edges_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    theta: float = 1 / 3,
    beta0: float | None = None,
) -> MobilityCoupling:
    if beta0 is None:
        raise MobilityDataError("beta0 must be provided explicitly (no implicit default).")

    counties, W, N = align_counties(edges_df, pop_df)
    P = presence_matrix(W, N, theta=theta, county_labels=counties)
    D = population_present(N, P)
    C = mixing_matrix(P, N, D)
    B = transmission_matrix(C, beta0)

    return MobilityCoupling(
        counties=counties, W=W, N=N, P=P, D=D, C=C, B=B, theta=theta, beta0=beta0
    )


def _check_finite_nonneg(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise MobilityDataError(f"{name} contains non-finite values.")
    if np.any(arr < 0):
        raise MobilityDataError(f"{name} contains negative entries.")


def _check_row_sums(arr: np.ndarray, target: float, name: str, atol: float = ROW_SUM_ATOL) -> None:
    row_sums = arr.sum(axis=1)
    bad = np.where(np.abs(row_sums - target) > atol)[0]
    if bad.size:
        raise MobilityDataError(
            f"{name} rows must sum to {target}; rows {bad.tolist()} sum to "
            f"{row_sums[bad].tolist()}."
        )


def mobility_rate(W: np.ndarray, N: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    N = np.asarray(N, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise SeedSelectionError(f"W must be square, got shape {W.shape}.")
    n = W.shape[0]
    if N.shape != (n,):
        raise SeedSelectionError(f"N must have shape ({n},), got {N.shape}.")
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(N)):
        raise SeedSelectionError("W and N must be finite.")
    if np.any(W < 0):
        raise SeedSelectionError("W must be nonnegative.")
    if np.any(N <= 0):
        raise SeedSelectionError("N must be strictly positive.")

    off_county_sum = W.sum(axis=1) - np.diag(W)
    return off_county_sum / N


@dataclass
class SeedCounty:
    quantile: float
    county_index: int
    county: str
    m: float
    population: float


def select_seed_counties(
    m: np.ndarray,
    counties: list[str],
    N: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> list[SeedCounty]:
    m = np.asarray(m, dtype=float)
    n = m.shape[0]
    if len(counties) != n:
        raise SeedSelectionError(f"counties must have length {n}, got {len(counties)}.")
    N = np.asarray(N, dtype=float)
    if N.shape != (n,):
        raise SeedSelectionError(f"N must have shape ({n},), got {N.shape}.")
    if not np.all(np.isfinite(m)):
        raise SeedSelectionError("m must be finite.")
    if len(quantiles) == 0:
        raise SeedSelectionError("quantiles must be non-empty.")
    if n < len(quantiles):
        raise SeedSelectionError(
            f"Need at least {len(quantiles)} counties to select {len(quantiles)} unique "
            f"seed counties, got {n}."
        )

    used: set[int] = set()
    selected: list[SeedCounty] = []
    for q in quantiles:
        target = np.quantile(m, q)
        distances = np.abs(m - target)
        order = np.lexsort((np.arange(n), distances))
        chosen = None
        for idx in order:
            if int(idx) not in used:
                chosen = int(idx)
                break
        if chosen is None:
            raise SeedSelectionError(f"No unused county available for quantile {q}.")
        used.add(chosen)
        selected.append(
            SeedCounty(
                quantile=q,
                county_index=chosen,
                county=counties[chosen],
                m=float(m[chosen]),
                population=float(N[chosen]),
            )
        )

    if len({s.county_index for s in selected}) != len(quantiles):
        raise SeedSelectionError("Seed county selection did not yield unique counties.")

    return selected
