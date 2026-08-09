from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _PACKAGE_DIR / "config.yaml"
RESULTS_DIR = _PACKAGE_DIR / "results"

_REQUIRED_STUDY_DESIGN_KEYS = (
    "primary_threshold",
    "availability_thresholds",
    "true_count_groups",
    "seed_quantiles",
    "state_u_levels",
    "coupling_b_levels",
    "combined_scenario",
    "n_observation_draws",
    "n_truth_realizations",
    "bootstrap_replicates",
    "confidence_level",
)


def _validate_number_list(values: Any, name: str, path: Path, *, upper_bound: float | None = None) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: study_design.{name} must be a non-empty list.")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{path}: study_design.{name} entries must be numbers, got {v!r}.")
        if v < 0:
            raise ValueError(f"{path}: study_design.{name} entries must be nonnegative, got {v}.")
        if upper_bound is not None and v >= upper_bound:
            raise ValueError(f"{path}: study_design.{name} entries must be < {upper_bound}, got {v}.")


def _validate_study_design(sd: Any, path: Path) -> None:
    if not isinstance(sd, dict):
        raise ValueError(f"{path} must contain a 'study_design' mapping.")

    missing = [k for k in _REQUIRED_STUDY_DESIGN_KEYS if k not in sd]
    if missing:
        raise ValueError(f"{path}: study_design section missing keys: {missing}")

    if not isinstance(sd["primary_threshold"], int) or isinstance(sd["primary_threshold"], bool) or sd["primary_threshold"] <= 0:
        raise ValueError(f"{path}: study_design.primary_threshold must be a positive integer.")

    _validate_number_list(sd["availability_thresholds"], "availability_thresholds", path)

    groups = sd["true_count_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{path}: study_design.true_count_groups must be a non-empty list.")
    for g in groups:
        if not isinstance(g, dict) or not {"label", "low", "high"} <= g.keys():
            raise ValueError(f"{path}: each study_design.true_count_groups entry needs label/low/high.")
        if not isinstance(g["label"], str) or not g["label"]:
            raise ValueError(f"{path}: true_count_groups entries need a non-empty string label.")
        if g["high"] is not None and g["low"] >= g["high"]:
            raise ValueError(f"{path}: true_count_groups entry {g['label']!r} must have low < high.")

    for q in sd["seed_quantiles"]:
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not (0.0 <= q <= 1.0):
            raise ValueError(f"{path}: study_design.seed_quantiles entries must lie in [0, 1], got {q!r}.")
    if not sd["seed_quantiles"]:
        raise ValueError(f"{path}: study_design.seed_quantiles must be non-empty.")

    _validate_number_list(sd["state_u_levels"], "state_u_levels", path, upper_bound=1.0)
    _validate_number_list(sd["coupling_b_levels"], "coupling_b_levels", path, upper_bound=1.0)

    combined = sd["combined_scenario"]
    if not isinstance(combined, dict) or not {"u", "b"} <= combined.keys():
        raise ValueError(f"{path}: study_design.combined_scenario must have 'u' and 'b'.")
    if not (0.0 <= combined["u"] < 1.0) or not (0.0 <= combined["b"] < 1.0):
        raise ValueError(f"{path}: study_design.combined_scenario values must lie in [0, 1).")

    if (
        not isinstance(sd["n_observation_draws"], int)
        or isinstance(sd["n_observation_draws"], bool)
        or sd["n_observation_draws"] <= 0
    ):
        raise ValueError(f"{path}: study_design.n_observation_draws must be a positive integer.")

    if (
        not isinstance(sd["n_truth_realizations"], int)
        or isinstance(sd["n_truth_realizations"], bool)
        or sd["n_truth_realizations"] <= 0
    ):
        raise ValueError(f"{path}: study_design.n_truth_realizations must be a positive integer.")

    if (
        not isinstance(sd["bootstrap_replicates"], int)
        or isinstance(sd["bootstrap_replicates"], bool)
        or sd["bootstrap_replicates"] <= 0
    ):
        raise ValueError(f"{path}: study_design.bootstrap_replicates must be a positive integer.")

    cl = sd["confidence_level"]
    if isinstance(cl, bool) or not isinstance(cl, (int, float)) or not (0.0 < cl < 1.0):
        raise ValueError(f"{path}: study_design.confidence_level must lie in (0, 1).")


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "epidemiology" not in config:
        raise ValueError(f"{path} must contain an 'epidemiology' section.")
    if not isinstance(config.get("states"), list) or not config["states"]:
        raise ValueError(f"{path} must contain a non-empty 'states' list.")
    sim = config.get("simulation")
    if not isinstance(sim, dict) or "seed" not in sim:
        raise ValueError(f"{path} must contain a 'simulation.seed' value.")
    _required_sim_keys = ("horizon_days", "dt_days", "n_seed_infected")
    missing_sim = [k for k in _required_sim_keys if k not in sim]
    if missing_sim:
        raise ValueError(f"{path}: simulation section missing keys: {missing_sim}")
    if isinstance(sim["horizon_days"], bool) or not isinstance(sim["horizon_days"], (int, float)) or sim["horizon_days"] <= 0:
        raise ValueError(f"{path}: simulation.horizon_days must be a positive number.")
    if isinstance(sim["dt_days"], bool) or not isinstance(sim["dt_days"], (int, float)) or sim["dt_days"] <= 0:
        raise ValueError(f"{path}: simulation.dt_days must be a positive number.")
    if (
        not isinstance(sim["n_seed_infected"], int)
        or isinstance(sim["n_seed_infected"], bool)
        or sim["n_seed_infected"] <= 0
    ):
        raise ValueError(f"{path}: simulation.n_seed_infected must be a positive integer.")
    _validate_study_design(config.get("study_design"), path)

    return config


_config = load_config()
_epi = _config["epidemiology"]
_sd = _config["study_design"]

RAW_CONFIG: dict[str, Any] = _config

GAMMA: float = _epi["gamma"]
R_STAR: float = _epi["r_star"]
THETA: float = _epi["theta"]
BETA0: float = R_STAR * GAMMA
STATES: list[str] = _config["states"]
SEED: int = _config["simulation"]["seed"]
HORIZON_DAYS: float = _config["simulation"]["horizon_days"]
DT_DAYS: float = _config["simulation"]["dt_days"]
N_SEED_INFECTED: int = _config["simulation"]["n_seed_infected"]

PRIMARY_THRESHOLD: int = _sd["primary_threshold"]
AVAILABILITY_THRESHOLDS: tuple = tuple(_sd["availability_thresholds"])
TRUE_COUNT_GROUP_BOUNDS: tuple = tuple(
    (g["low"], g["high"] if g["high"] is not None else float("inf")) for g in _sd["true_count_groups"]
)
TRUE_COUNT_GROUP_LABELS: tuple = tuple(g["label"] for g in _sd["true_count_groups"])
SEED_QUANTILES: tuple = tuple(_sd["seed_quantiles"])
STATE_U_LEVELS: tuple = tuple(_sd["state_u_levels"])
COUPLING_B_LEVELS: tuple = tuple(_sd["coupling_b_levels"])
COMBINED_U: float = _sd["combined_scenario"]["u"]
COMBINED_B: float = _sd["combined_scenario"]["b"]
N_OBSERVATION_DRAWS: int = _sd["n_observation_draws"]
N_TRUTH_REALIZATIONS: int = _sd["n_truth_realizations"]
BOOTSTRAP_REPLICATES: int = _sd["bootstrap_replicates"]
CONFIDENCE_LEVEL: float = _sd["confidence_level"]


def results_root() -> Path:
    return RESULTS_DIR


ANALYSIS_MANIFEST_FILES: dict[str, tuple[str, ...]] = {
    "state_underreporting": (
        "full_study_bounded_state_underreporting_draws.csv",
        "full_study_bounded_state_underreporting_manifest.csv",
        "full_study_bounded_state_underreporting_trajectories.csv",
    ),
    "coupling_uncertainty": (
        "full_study_bounded_coupling_uncertainty_draws.csv",
        "full_study_bounded_coupling_uncertainty_manifest.csv",
        "full_study_bounded_coupling_uncertainty_trajectories.csv",
    ),
    "combined_uncertainty": (
        "full_study_combined_uncertainty_draws.csv",
        "full_study_combined_uncertainty_manifest.csv",
        "full_study_combined_uncertainty_trajectories.csv",
    ),
    "binomial_reporting_noise": (
        "full_study_binomial_reporting_noise_draws.csv",
        "full_study_binomial_reporting_noise_manifest.csv",
        "full_study_binomial_reporting_noise_trajectories.csv",
    ),
    "robustness_radius": (
        "full_study_robustness_radius_trajectory_summary.csv",
        "full_study_robustness_radius_joint_strata.csv",
    ),
    "bootstrap": (
        "bootstrap_confidence_intervals.csv",
        "bootstrap_manifest.json",
    ),
}
STATE_MANIFEST_FILES: tuple[str, ...] = (
    "truth_trajectory_manifest.csv",
    "nominal_drn_manifest.csv",
    "robustness_radius_manifest.csv",
)
_FILE_TO_ANALYSIS: dict[str, str] = {
    fname: name for name, fnames in ANALYSIS_MANIFEST_FILES.items() for fname in fnames
}


def states_dir() -> Path:
    return results_root() / "states"


def state_dir(state: str, category: str) -> Path:
    return states_dir() / state / category


def state_manifests_dir() -> Path:
    return states_dir() / "manifests"


def state_manifest_path(filename: str) -> Path:
    if filename not in STATE_MANIFEST_FILES:
        raise ValueError(f"{filename!r} is not one of the cross-state manifests {STATE_MANIFEST_FILES}.")
    return state_manifests_dir() / filename


def analyses_dir(name: str) -> Path:
    return results_root() / "analyses" / name


def analysis_manifest_path(filename: str) -> Path:
    name = _FILE_TO_ANALYSIS.get(filename)
    if name is None:
        raise ValueError(f"{filename!r} is not a recognized analysis output filename.")
    return analyses_dir(name) / filename


def reports_dir(subdir: str = "") -> Path:
    base = results_root() / "reports"
    return base / subdir if subdir else base


def figures_dir(subdir: str = "") -> Path:
    base = results_root() / "figures"
    return base / subdir if subdir else base


def cache_dir(name: str) -> Path:
    return results_root() / "cache" / name


def archive_dir() -> Path:
    return RESULTS_DIR / "archive"


EXPERIMENT_STATE_UNCERTAINTY = "bounded_state_uncertainty"
EXPERIMENT_COUPLING_UNCERTAINTY = "bounded_coupling_uncertainty"
EXPERIMENT_COMBINED_UNCERTAINTY = "combined_uncertainty"
EXPERIMENT_BINOMIAL_NOISE = "binomial_reporting_noise"
EXPERIMENT_ROBUSTNESS_RADIUS = "robustness_radius"
EXPERIMENT_TRUTH_TRAJECTORY = "truth_trajectory_generation"
EXPERIMENT_TRAJECTORY_BOOTSTRAP = "trajectory_bootstrap"

def derive_seed(*parts) -> int:
    if not parts:
        raise ValueError("derive_seed requires at least one label.")
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
