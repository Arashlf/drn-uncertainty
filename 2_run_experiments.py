from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import bootstrap
import combined_uncertainty as cmb
import config as cfg
import coupling_uncertainty as bcu
import progress
import reporting_noise as brn
import state_uncertainty as bsu
from sir import validate_resume_metadata


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "")


def _b_hash(B: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(B, dtype=float).tobytes()).hexdigest()


def _load_mobility_full(state: str) -> np.ndarray:
    path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
    with np.load(path, allow_pickle=True) as data:
        return {
            "counties": list(data["counties"]), "N": data["N"], "B": data["B"],
            "theta": float(data["theta"]), "beta0": float(data["beta0"]),
        }


def _load_nominal_manifest() -> list[dict]:
    path = cfg.state_manifest_path("nominal_drn_manifest.csv")
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row["validation_status"] == "valid"]


def _load_trajectory_for_experiment(nominal_row: dict) -> dict:
    results_root = cfg.results_root()
    with np.load(results_root / nominal_row["output_path"], allow_pickle=True) as data:
        source_path = str(data["source_path"])

    with np.load(results_root / source_path, allow_pickle=True) as data:
        S, I, N = data["S"], data["I"], data["N"]

    return {"S": S, "I": I, "N": N, "source_path": source_path}


def _state_uncertainty_full_study() -> None:
    n_draws = cfg.N_OBSERVATION_DRAWS
    draw_csv_path = cfg.analysis_manifest_path("full_study_bounded_state_underreporting_draws.csv")
    trajectory_csv_path = cfg.analysis_manifest_path("full_study_bounded_state_underreporting_trajectories.csv")
    manifest_path = cfg.analysis_manifest_path("full_study_bounded_state_underreporting_manifest.csv")

    nominal_rows = _load_nominal_manifest()

    mobility_cache: dict[str, dict] = {}
    draw_rows_all: list[dict] = []
    trajectory_rows_all: list[dict] = []
    manifest_rows: list[dict] = []

    for nominal_row in nominal_rows:
        state = nominal_row["state"]
        county = nominal_row["seed_county"]
        realization = int(nominal_row["realization"])
        trajectory_id = f"{_slug(county)}_realization_{realization:02d}"

        if state not in mobility_cache:
            m = _load_mobility_full(state)
            mobility_cache[state] = {"B": m["B"]}
        mobility = mobility_cache[state]
        traj = _load_trajectory_for_experiment(nominal_row)

        total_draws = 0
        for u in cfg.STATE_U_LEVELS:
            draw_rows, trajectory_rows = bsu.run_trajectory_severity(
                traj["S"], traj["I"], traj["N"], mobility["B"], cfg.GAMMA, u, cfg.SEED,
                n_draws=n_draws, primary_threshold=cfg.PRIMARY_THRESHOLD,
                state=state, trajectory_id=trajectory_id,
                count_group_bounds=cfg.TRUE_COUNT_GROUP_BOUNDS, count_group_labels=cfg.TRUE_COUNT_GROUP_LABELS,
            )
            for row in draw_rows:
                draw_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, **row})
            for row in trajectory_rows:
                trajectory_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, **row})

            total_draws += trajectory_rows[0]["n_draws"]

        manifest_rows.append({
            "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county,
            "realization": realization, "trajectory_id": trajectory_id,
            "source_path": traj["source_path"], "nominal_drn_path": nominal_row["output_path"],
            "n_severities": len(cfg.STATE_U_LEVELS), "total_draws": total_draws,
        })

    draw_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "u", "a", "draw_index", "seed", "count_group"] + list(bsu.STAT_FIELDS)
    trajectory_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "u", "a", "n_draws", "count_group"] + list(bsu.STAT_FIELDS)
    manifest_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "source_path", "nominal_drn_path", "n_severities", "total_draws"]

    draw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=draw_fields)
        writer.writeheader()
        writer.writerows(draw_rows_all)

    with open(trajectory_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows_all)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def stage_state_uncertainty() -> None:
    _state_uncertainty_full_study()


def _coupling_config_spec(n_draws: int) -> dict:
    return {
        "base_seed": cfg.SEED,
        "severities": ",".join(str(s) for s in sorted(cfg.COUPLING_B_LEVELS)),
        "n_observation_draws": n_draws,
        "count_group_spec": repr(list(zip(cfg.TRUE_COUNT_GROUP_BOUNDS, cfg.TRUE_COUNT_GROUP_LABELS))),
    }


def _coupling_run_or_resume(state: str, county: str, trajectory_id: str, traj: dict, mobility: dict, nominal_row: dict, n_draws: int, draw_row_keys: list, traj_row_keys: list) -> tuple[list[dict], list[dict]]:
    out_dir = cfg.cache_dir("coupling_uncertainty") / state / _slug(county)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"realization_{int(nominal_row['realization']):02d}.npz"

    expected_meta = {
        "state": state, "seed_county": county, "seed_quantile": float(nominal_row["seed_quantile"]),
        "realization": int(nominal_row["realization"]), "trajectory_id": trajectory_id,
        "nominal_drn_path": nominal_row["output_path"], "source_path": traj["source_path"],
        "B_hash": mobility["B_hash"], "gamma_used": cfg.GAMMA, "primary_threshold_used": cfg.PRIMARY_THRESHOLD,
        **_coupling_config_spec(n_draws),
    }

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as data:
            saved_meta = {k: (data[k].item() if isinstance(data[k], np.ndarray) and data[k].ndim == 0 else str(data[k])) for k in expected_meta}
            for k in ("seed_quantile",):
                saved_meta[k] = float(saved_meta[k])
            for k in ("realization", "n_observation_draws"):
                saved_meta[k] = int(saved_meta[k])
            for k in ("base_seed",):
                saved_meta[k] = int(saved_meta[k])
            check_result = validate_resume_metadata(saved_meta, expected_meta)
            if not check_result.ok:
                raise ValueError(f"existing bounded-coupling-uncertainty cache's metadata does not match the current trajectory/mobility/config ({check_result.reason}); remove {cache_path} to recompute.")
            n_draw = int(data["n_draw_rows"])
            n_traj = int(data["n_traj_rows"])
            draw_rows = bcu.columns_to_rows({k: data[f"draw_{k}"] for k in draw_row_keys}, draw_row_keys, n_draw)
            trajectory_rows = bcu.columns_to_rows({k: data[f"traj_{k}"] for k in traj_row_keys}, traj_row_keys, n_traj)
        return draw_rows, trajectory_rows

    draw_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for b in cfg.COUPLING_B_LEVELS:
        d_rows, t_rows = bcu.run_trajectory_severity(
            traj["S"], traj["I"], traj["N"], mobility["B"], cfg.GAMMA, b, cfg.SEED,
            n_draws=n_draws, primary_threshold=cfg.PRIMARY_THRESHOLD,
            state=state, trajectory_id=trajectory_id,
            count_group_bounds=cfg.TRUE_COUNT_GROUP_BOUNDS, count_group_labels=cfg.TRUE_COUNT_GROUP_LABELS,
        )
        draw_rows.extend(d_rows)
        trajectory_rows.extend(t_rows)

    save_kwargs = {f"draw_{k}": v for k, v in bcu.rows_to_columns(draw_rows, draw_row_keys).items()}
    save_kwargs.update({f"traj_{k}": v for k, v in bcu.rows_to_columns(trajectory_rows, traj_row_keys).items()})
    save_kwargs["n_draw_rows"] = len(draw_rows)
    save_kwargs["n_traj_rows"] = len(trajectory_rows)
    np.savez_compressed(cache_path, **save_kwargs, **expected_meta)
    return draw_rows, trajectory_rows


def _coupling_uncertainty_full_study() -> None:
    n_draws = cfg.N_OBSERVATION_DRAWS
    draw_row_keys = ["b", "draw_index", "seed", "count_group", "regime"] + list(bcu.STAT_FIELDS)
    traj_row_keys = ["b", "n_draws", "count_group", "regime"] + list(bcu.STAT_FIELDS)
    draw_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + draw_row_keys
    trajectory_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + traj_row_keys
    manifest_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "source_path", "nominal_drn_path", "n_severities", "total_draws"]

    draw_csv_path = cfg.analysis_manifest_path("full_study_bounded_coupling_uncertainty_draws.csv")
    trajectory_csv_path = cfg.analysis_manifest_path("full_study_bounded_coupling_uncertainty_trajectories.csv")
    manifest_path = cfg.analysis_manifest_path("full_study_bounded_coupling_uncertainty_manifest.csv")

    nominal_rows = _load_nominal_manifest()

    mobility_cache: dict[str, dict] = {}
    draw_rows_all: list[dict] = []
    trajectory_rows_all: list[dict] = []
    manifest_rows: list[dict] = []

    for nominal_row in nominal_rows:
        state = nominal_row["state"]
        county = nominal_row["seed_county"]
        realization = int(nominal_row["realization"])
        trajectory_id = f"{_slug(county)}_realization_{realization:02d}"

        if state not in mobility_cache:
            m = _load_mobility_full(state)
            mobility_cache[state] = {"B": m["B"], "B_hash": _b_hash(m["B"])}
        mobility = mobility_cache[state]

        traj = _load_trajectory_for_experiment(nominal_row)
        draw_rows, trajectory_rows = _coupling_run_or_resume(state, county, trajectory_id, traj, mobility, nominal_row, n_draws, draw_row_keys, traj_row_keys)

        total_draws = 0
        for row in draw_rows:
            draw_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})
        for row in trajectory_rows:
            trajectory_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})

        for b in cfg.COUPLING_B_LEVELS:
            overall = next(r for r in trajectory_rows if r["b"] == b and r["count_group"] == "overall" and r["regime"] == "overall")
            total_draws += overall["n_draws"]

        manifest_rows.append({
            "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county,
            "realization": realization, "trajectory_id": trajectory_id, "source_path": traj["source_path"],
            "nominal_drn_path": nominal_row["output_path"], "n_severities": len(cfg.COUPLING_B_LEVELS),
            "total_draws": total_draws,
        })

    draw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=draw_fields)
        writer.writeheader()
        writer.writerows(draw_rows_all)

    with open(trajectory_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows_all)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def stage_coupling_uncertainty() -> None:
    _coupling_uncertainty_full_study()


def _combined_uncertainty_full_study() -> None:
    n_draws = cfg.N_OBSERVATION_DRAWS
    draw_row_keys = ["u", "b", "draw_index", "seed_alpha", "seed_xi", "count_group", "regime"] + list(cmb.STAT_FIELDS)
    traj_row_keys = ["u", "b", "n_draws", "count_group", "regime"] + list(cmb.STAT_FIELDS)
    draw_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + draw_row_keys
    trajectory_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + traj_row_keys
    manifest_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "source_path", "nominal_drn_path", "n_draws"]

    draw_csv_path = cfg.analysis_manifest_path("full_study_combined_uncertainty_draws.csv")
    trajectory_csv_path = cfg.analysis_manifest_path("full_study_combined_uncertainty_trajectories.csv")
    manifest_path = cfg.analysis_manifest_path("full_study_combined_uncertainty_manifest.csv")

    nominal_rows = _load_nominal_manifest()

    mobility_cache: dict[str, dict] = {}
    draw_rows_all: list[dict] = []
    trajectory_rows_all: list[dict] = []
    manifest_rows: list[dict] = []

    for nominal_row in nominal_rows:
        state = nominal_row["state"]
        county = nominal_row["seed_county"]
        realization = int(nominal_row["realization"])
        trajectory_id = f"{_slug(county)}_realization_{realization:02d}"

        if state not in mobility_cache:
            m = _load_mobility_full(state)
            mobility_cache[state] = {"B": m["B"]}
        mobility = mobility_cache[state]

        traj = _load_trajectory_for_experiment(nominal_row)
        draw_rows, trajectory_rows = cmb.run_trajectory_combined(
            traj["S"], traj["I"], traj["N"], mobility["B"], cfg.GAMMA, cfg.COMBINED_U, cfg.COMBINED_B, cfg.SEED,
            n_draws=n_draws, primary_threshold=cfg.PRIMARY_THRESHOLD,
            state=state, trajectory_id=trajectory_id,
            count_group_bounds=cfg.TRUE_COUNT_GROUP_BOUNDS, count_group_labels=cfg.TRUE_COUNT_GROUP_LABELS,
        )

        for row in draw_rows:
            draw_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})
        for row in trajectory_rows:
            trajectory_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})

        overall = next(r for r in trajectory_rows if r["count_group"] == "overall" and r["regime"] == "overall")

        manifest_rows.append({
            "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county,
            "realization": realization, "trajectory_id": trajectory_id, "source_path": traj["source_path"],
            "nominal_drn_path": nominal_row["output_path"], "n_draws": overall["n_draws"],
        })

    draw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=draw_fields)
        writer.writeheader()
        writer.writerows(draw_rows_all)

    with open(trajectory_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows_all)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def stage_combined_uncertainty() -> None:
    _combined_uncertainty_full_study()


def _reporting_noise_full_study() -> None:
    n_draws = cfg.N_OBSERVATION_DRAWS
    draw_row_keys = ["u", "p", "draw_index", "seed", "count_group", "regime"] + list(brn.STAT_FIELDS)
    traj_row_keys = ["u", "p", "n_draws", "count_group", "regime"] + list(brn.STAT_FIELDS)
    draw_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + draw_row_keys
    trajectory_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id"] + traj_row_keys
    manifest_fields = ["state", "seed_quantile", "seed_county", "realization", "trajectory_id", "source_path", "nominal_drn_path", "n_severities", "total_draws"]

    draw_csv_path = cfg.analysis_manifest_path("full_study_binomial_reporting_noise_draws.csv")
    trajectory_csv_path = cfg.analysis_manifest_path("full_study_binomial_reporting_noise_trajectories.csv")
    manifest_path = cfg.analysis_manifest_path("full_study_binomial_reporting_noise_manifest.csv")

    nominal_rows = _load_nominal_manifest()

    mobility_cache: dict[str, dict] = {}
    draw_rows_all: list[dict] = []
    trajectory_rows_all: list[dict] = []
    manifest_rows: list[dict] = []

    for nominal_row in nominal_rows:
        state = nominal_row["state"]
        county = nominal_row["seed_county"]
        realization = int(nominal_row["realization"])
        trajectory_id = f"{_slug(county)}_realization_{realization:02d}"

        if state not in mobility_cache:
            m = _load_mobility_full(state)
            mobility_cache[state] = {"B": m["B"]}
        mobility = mobility_cache[state]

        traj = _load_trajectory_for_experiment(nominal_row)
        draw_rows: list[dict] = []
        trajectory_rows: list[dict] = []
        for u in cfg.STATE_U_LEVELS:
            d_rows, t_rows = brn.run_trajectory_severity(
                traj["S"], traj["I"], traj["N"], mobility["B"], cfg.GAMMA, u, cfg.SEED,
                n_draws=n_draws, primary_threshold=cfg.PRIMARY_THRESHOLD,
                state=state, trajectory_id=trajectory_id,
                count_group_bounds=cfg.TRUE_COUNT_GROUP_BOUNDS, count_group_labels=cfg.TRUE_COUNT_GROUP_LABELS,
            )
            draw_rows.extend(d_rows)
            trajectory_rows.extend(t_rows)

        total_draws = 0
        for row in draw_rows:
            draw_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})
        for row in trajectory_rows:
            trajectory_rows_all.append({"state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization, "trajectory_id": trajectory_id, **row})

        for u in cfg.STATE_U_LEVELS:
            overall = next(r for r in trajectory_rows if r["u"] == u and r["count_group"] == "overall" and r["regime"] == "overall")
            total_draws += overall["n_draws"]

        manifest_rows.append({
            "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county,
            "realization": realization, "trajectory_id": trajectory_id, "source_path": traj["source_path"],
            "nominal_drn_path": nominal_row["output_path"], "n_severities": len(cfg.STATE_U_LEVELS),
            "total_draws": total_draws,
        })

    draw_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(draw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=draw_fields)
        writer.writeheader()
        writer.writerows(draw_rows_all)

    with open(trajectory_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows_all)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)


def stage_reporting_noise() -> None:
    _reporting_noise_full_study()


def _robustness_radius_full_study() -> None:
    count_groups = ("overall",) + tuple(cfg.TRUE_COUNT_GROUP_LABELS)
    regimes = ("overall", "growth", "decline")
    n_quantile_bins = 5

    trajectory_summary_fields = [
        "state", "seed_quantile", "seed_county", "realization", "count_group", "regime",
        "n", "mean", "median", "iqr", "p5", "p95", "frac_ge_0.2", "frac_ge_0.4", "frac_ge_0.6",
    ]
    joint_strata_fields = [
        "abs_r_minus_1_bin", "abs_r_minus_1_lo", "abs_r_minus_1_hi", "q_bin", "q_lo", "q_hi",
        "n", "mean_u_star", "median_u_star", "frac_ge_0.2", "frac_ge_0.4", "frac_ge_0.6",
    ]

    manifest_path = cfg.state_manifest_path("robustness_radius_manifest.csv")
    trajectory_summary_path = cfg.analysis_manifest_path("full_study_robustness_radius_trajectory_summary.csv")
    joint_strata_path = cfg.analysis_manifest_path("full_study_robustness_radius_joint_strata.csv")
    results_root = cfg.results_root()

    def _regime_mask(hat: np.ndarray, regime: str) -> np.ndarray:
        if regime == "overall":
            return np.ones(hat.shape, dtype=bool)
        if regime == "growth":
            return hat > 1.0
        return hat < 1.0

    def _count_group_mask(I_vals: np.ndarray, group: str) -> np.ndarray:
        if group == "overall":
            return np.ones(I_vals.shape, dtype=bool)
        for (lo, hi), label in zip(cfg.TRUE_COUNT_GROUP_BOUNDS, cfg.TRUE_COUNT_GROUP_LABELS):
            if label == group:
                return (I_vals >= lo) & (I_vals < hi)
        return np.zeros(I_vals.shape, dtype=bool)

    def _stats(u: np.ndarray) -> dict:
        n = int(u.shape[0])
        if n == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan"), "iqr": float("nan"), "p5": float("nan"), "p95": float("nan"), "frac_ge_0.2": float("nan"), "frac_ge_0.4": float("nan"), "frac_ge_0.6": float("nan")}
        p5, p25, median, p75, p95 = np.percentile(u, [5, 25, 50, 75, 95])
        return {
            "n": n, "mean": float(u.mean()), "median": float(median), "iqr": float(p75 - p25),
            "p5": float(p5), "p95": float(p95),
            "frac_ge_0.2": float(np.mean(u >= 0.2)), "frac_ge_0.4": float(np.mean(u >= 0.4)), "frac_ge_0.6": float(np.mean(u >= 0.6)),
        }

    with open(manifest_path, newline="") as f:
        manifest_rows = [row for row in csv.DictReader(f) if row["validation_status"] == "valid"]

    trajectories: list[dict] = []
    for row in manifest_rows:
        with np.load(results_root / row["output_path"], allow_pickle=True) as data:
            primary_mask = data["primary_mask"]
            u_star = data["u_star"][primary_mask]
            hat_drn = data["hat_drn"][primary_mask]
            q_hat = data["q_hat"][primary_mask]
            I_primary = data["I_primary"]
        trajectories.append({
            "state": row["state"], "seed_quantile": row["seed_quantile"], "seed_county": row["seed_county"],
            "realization": row["realization"], "u": u_star, "hat": hat_drn, "q": q_hat, "I": I_primary,
        })

    traj_summary_rows = []
    for t in trajectories:
        for group in count_groups:
            g_mask = _count_group_mask(t["I"], group)
            for regime in regimes:
                r_mask = _regime_mask(t["hat"], regime)
                mask = g_mask & r_mask
                s = _stats(t["u"][mask])
                traj_summary_rows.append({"state": t["state"], "seed_quantile": t["seed_quantile"], "seed_county": t["seed_county"], "realization": t["realization"], "count_group": group, "regime": regime, **s})
    trajectory_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_summary_fields)
        writer.writeheader()
        writer.writerows(traj_summary_rows)

    u_all = np.concatenate([t["u"] for t in trajectories])
    abs_r_all = np.concatenate([np.abs(t["hat"] - 1.0) for t in trajectories])
    q_all = np.concatenate([t["q"] for t in trajectories])
    finite_q = np.isfinite(q_all)
    abs_r_all, q_all, u_for_joint = abs_r_all[finite_q], q_all[finite_q], u_all[finite_q]

    abs_r_edges = np.quantile(abs_r_all, np.linspace(0, 1, n_quantile_bins + 1))
    q_edges = np.quantile(q_all, np.linspace(0, 1, n_quantile_bins + 1))
    abs_r_edges[-1] += 1e-9
    q_edges[-1] += 1e-9

    joint_rows = []
    for i in range(n_quantile_bins):
        r_lo, r_hi = abs_r_edges[i], abs_r_edges[i + 1]
        r_mask = (abs_r_all >= r_lo) & (abs_r_all < r_hi)
        for j in range(n_quantile_bins):
            q_lo, q_hi = q_edges[j], q_edges[j + 1]
            cell_mask = r_mask & (q_all >= q_lo) & (q_all < q_hi)
            s = _stats(u_for_joint[cell_mask])
            joint_rows.append({
                "abs_r_minus_1_bin": f"Q{i + 1}", "abs_r_minus_1_lo": float(r_lo), "abs_r_minus_1_hi": float(r_hi),
                "q_bin": f"Q{j + 1}", "q_lo": float(q_lo), "q_hi": float(q_hi),
                "n": s["n"], "mean_u_star": s["mean"], "median_u_star": s["median"],
                "frac_ge_0.2": s["frac_ge_0.2"], "frac_ge_0.4": s["frac_ge_0.4"], "frac_ge_0.6": s["frac_ge_0.6"],
            })
    with open(joint_strata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=joint_strata_fields)
        writer.writeheader()
        writer.writerows(joint_rows)


def stage_robustness_radius_summary() -> None:
    _robustness_radius_full_study()


_SUMMARY_PERCENTILES = (20, 40, 60, 80, 90, 95)
_SUMMARY_FIELDS = ["quantity", "group", "n", "mean", "median", "min", "max"] + [f"p{p}" for p in _SUMMARY_PERCENTILES]


def _bounded_summary_stats(values: np.ndarray) -> dict:
    n = int(values.shape[0])
    if n == 0:
        row = {"n": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
        row.update({f"p{p}": float("nan") for p in _SUMMARY_PERCENTILES})
        return row
    row = {
        "n": n, "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "max": float(values.max()),
    }
    percentile_values = np.percentile(values, _SUMMARY_PERCENTILES)
    row.update({f"p{p}": float(v) for p, v in zip(_SUMMARY_PERCENTILES, percentile_values)})
    return row


def _external_contribution_structural_summary() -> None:
    states = list(cfg.STATES)
    results_root = cfg.results_root()
    out_dir = cfg.analyses_dir("robustness_radius")
    csv_path = out_dir / "external_contribution_and_structural_summary.csv"

    manifest_path = cfg.state_manifest_path("robustness_radius_manifest.csv")
    with open(manifest_path, newline="") as f:
        manifest_rows = [row for row in csv.DictReader(f) if row["validation_status"] == "valid"]

    q_by_state: dict[str, list[np.ndarray]] = {s: [] for s in states}
    for row in manifest_rows:
        with np.load(results_root / row["output_path"], allow_pickle=True) as data:
            primary_mask = data["primary_mask"]
            q_hat = data["q_hat"]
        q_by_state.setdefault(row["state"], []).append(q_hat[primary_mask])
    q_by_state = {s: np.concatenate(v) if v else np.array([]) for s, v in q_by_state.items()}
    q_pooled = np.concatenate([v for v in q_by_state.values() if v.size]) if any(v.size for v in q_by_state.values()) else np.array([])

    c_by_state: dict[str, np.ndarray] = {}
    for state in states:
        mobility_path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(mobility_path, allow_pickle=True) as data:
            C = data["C"]
        c_by_state[state] = 1.0 - np.diagonal(C)
    c_pooled = np.concatenate(list(c_by_state.values())) if c_by_state else np.array([])

    rows = []
    rows.append({"quantity": "q_hat", "group": "pooled", **_bounded_summary_stats(q_pooled)})
    for state in states:
        rows.append({"quantity": "q_hat", "group": state, **_bounded_summary_stats(q_by_state.get(state, np.array([])))})
    rows.append({"quantity": "one_minus_c_ii", "group": "pooled", **_bounded_summary_stats(c_pooled)})
    for state in states:
        rows.append({"quantity": "one_minus_c_ii", "group": state, **_bounded_summary_stats(c_by_state.get(state, np.array([])))})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def stage_external_contribution_structural_summary() -> None:
    _external_contribution_structural_summary()


_BOOTSTRAP_ANALYSIS_SPECS = [
    {
        "key": "state_underreporting", "csv": "full_study_bounded_state_underreporting_trajectories.csv",
        "severity_col": "u", "has_regime": False,
        "rate_metrics": {"certification_rate": "n_available", "indeterminate_rate": "n_available", "point_misclassification_rate": "n_available"},
        "mean_metrics": {"mean_relative_interval_width": "n_available"},
    },
    {
        "key": "coupling_uncertainty", "csv": "full_study_bounded_coupling_uncertainty_trajectories.csv",
        "severity_col": "b", "has_regime": True,
        "rate_metrics": {"certification_rate": "n_available", "indeterminate_rate": "n_available", "point_misclassification_rate": "n_available"},
        "mean_metrics": {"mean_relative_interval_width": "n_available"},
    },
    {
        "key": "combined_uncertainty", "csv": "full_study_combined_uncertainty_trajectories.csv",
        "severity_col": None, "has_regime": True,
        "rate_metrics": {"certification_rate": "n_available", "indeterminate_rate": "n_available", "point_misclassification_rate": "n_available"},
        "mean_metrics": {"mean_relative_interval_width": "n_available"},
    },
    {
        "key": "binomial_reporting_noise", "csv": "full_study_binomial_reporting_noise_trajectories.csv",
        "severity_col": "u", "has_regime": True,
        "rate_metrics": {"availability_rate": "n", "zero_report_rate": "n", "threshold_misclassification_rate": "n_available"},
        "mean_metrics": {"mean_signed_error": "n_available", "mean_abs_error": "n_available", "mean_relative_error": "n_available"},
    },
    {
        "key": "robustness_radius", "csv": "full_study_robustness_radius_trajectory_summary.csv",
        "severity_col": None, "has_regime": True,
        "rate_metrics": {"frac_ge_0.2": "n", "frac_ge_0.4": "n", "frac_ge_0.6": "n"},
        "mean_metrics": {"mean": "n"},
    },
]


def stage_trajectory_bootstrap() -> None:
    def read_csv_rows(path: Path) -> list[dict]:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def load_indexed(path: Path, severity_col, has_regime: bool) -> dict:
        indexed: dict = {}
        for row in read_csv_rows(path):
            severity = float(row[severity_col]) if severity_col else None
            key = (row["state"], row["seed_county"])
            realization = int(row["realization"])
            cg = row["count_group"]
            regime = row["regime"] if has_regime else "overall"
            indexed.setdefault(severity, {}).setdefault(key, {}).setdefault(realization, {})[(cg, regime)] = row
        return indexed

    def stratum_arrays(indexed_severity: dict, strata: list, realizations: list, count_group: str, regime: str, metric_col: str, denom_col: str):
        denom_by, value_by = {}, {}
        for key in strata:
            denom = np.empty(len(realizations))
            value = np.empty(len(realizations))
            stratum_data = indexed_severity[key]
            for i, real in enumerate(realizations):
                row = stratum_data[real][(count_group, regime)]
                denom[i] = float(row[denom_col])
                value[i] = float(row[metric_col])
            denom_by[key] = denom
            value_by[key] = value
        return denom_by, value_by

    ci_fields = ["analysis", "severity", "metric", "view", "point_estimate", "ci_lower", "ci_upper", "n_replicates_used"]
    truth_manifest_path = cfg.state_manifest_path("truth_trajectory_manifest.csv")
    ci_csv_path = cfg.analysis_manifest_path("bootstrap_confidence_intervals.csv")

    n_truth_realizations = cfg.N_TRUTH_REALIZATIONS
    truth_rows = read_csv_rows(truth_manifest_path)
    strata = sorted({(r["state"], r["seed_county"]) for r in truth_rows})
    realizations = list(range(n_truth_realizations))
    strata_by_state: dict[str, list] = defaultdict(list)
    for key in strata:
        strata_by_state[key[0]].append(key)

    stratum_seeds = {key: bootstrap.stratum_seed(cfg.SEED, cfg.EXPERIMENT_TRAJECTORY_BOOTSTRAP, key[0], key[1]) for key in strata}
    index_matrix_by_stratum = {key: bootstrap.generate_index_matrix(cfg.BOOTSTRAP_REPLICATES, n_truth_realizations, stratum_seeds[key]) for key in strata}

    ci_rows: list[dict] = []

    for spec in _BOOTSTRAP_ANALYSIS_SPECS:
        path = cfg.analysis_manifest_path(spec["csv"])
        indexed = load_indexed(path, spec["severity_col"], spec["has_regime"])
        count_groups = ("overall",) + tuple(cfg.TRUE_COUNT_GROUP_LABELS)
        regimes = ("overall", "growth", "decline") if spec["has_regime"] else ("overall",)
        severities = sorted(indexed.keys(), key=lambda s: (s is None, s))

        for severity in severities:
            indexed_severity = indexed[severity]

            for metric_type, metrics in (("rate", spec["rate_metrics"]), ("mean", spec["mean_metrics"])):
                for metric_col, weight_col in metrics.items():
                    arrays_cache = {}
                    for cg in count_groups:
                        for regime in regimes:
                            denom_by, value_by = stratum_arrays(indexed_severity, strata, realizations, cg, regime, metric_col, weight_col)
                            if metric_type == "rate":
                                num_by = {k: bootstrap.reconstruct_numerator(value_by[k], denom_by[k]) for k in strata}
                            else:
                                num_by = {k: bootstrap.reconstruct_weighted_numerator(value_by[k], denom_by[k]) for k in strata}
                            arrays_cache[(cg, regime)] = (num_by, denom_by)

                    views = [("overall", "overall", "overall", None)]
                    for state in cfg.STATES:
                        views.append((f"state={state}", "overall", "overall", state))
                    for cg in cfg.TRUE_COUNT_GROUP_LABELS:
                        views.append((f"count_group={cg}", cg, "overall", None))
                    if spec["has_regime"]:
                        for regime in ("growth", "decline"):
                            views.append((f"regime={regime}", "overall", regime, None))

                    for view_name, cg, regime, state_filter in views:
                        num_by, denom_by = arrays_cache[(cg, regime)]
                        strata_subset = strata_by_state[state_filter] if state_filter else strata
                        result = bootstrap.bootstrap_pooled_rate(num_by, denom_by, index_matrix_by_stratum, strata_subset, cfg.CONFIDENCE_LEVEL)

                        ci_rows.append({
                            "analysis": spec["key"], "severity": severity, "metric": metric_col, "view": view_name,
                            "point_estimate": result.point_estimate, "ci_lower": result.ci_lower,
                            "ci_upper": result.ci_upper, "n_replicates_used": result.n_replicates_used,
                        })

    ci_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ci_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ci_fields)
        writer.writeheader()
        writer.writerows(ci_rows)


STAGES = {
    "state_uncertainty": stage_state_uncertainty,
    "coupling_uncertainty": stage_coupling_uncertainty,
    "combined_uncertainty": stage_combined_uncertainty,
    "reporting_noise": stage_reporting_noise,
    "robustness_radius_summary": stage_robustness_radius_summary,
    "external_contribution_structural_summary": stage_external_contribution_structural_summary,
    "trajectory_bootstrap": stage_trajectory_bootstrap,
}


STAGE_ICONS = {
    "state_uncertainty": "🧮",
    "coupling_uncertainty": "🔗",
    "combined_uncertainty": "🧩",
    "reporting_noise": "📡",
    "robustness_radius_summary": "🛡️",
    "external_contribution_structural_summary": "📐",
    "trajectory_bootstrap": "🎯",
}


def main() -> None:
    def _run_stage(ctx: dict) -> None:
        STAGES[ctx["item"]]()

    progress.run_pipeline(
        "2_run_experiments.py", list(STAGES), [("🚀", "run", _run_stage)],
        item_label="stages",
        item_format=lambda name: f"{STAGE_ICONS.get(name, '🔧')} {name}",
    )


if __name__ == "__main__":
    main()
