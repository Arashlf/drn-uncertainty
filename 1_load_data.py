from __future__ import annotations

import csv
import hashlib

import numpy as np

import config as cfg
import drn
import mobility
import progress
import robustness
import sir


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace(",", "")


def stage_mobility() -> None:
    states = cfg.STATES

    def _load(ctx: dict) -> None:
        ctx["edges_df"], ctx["pop_df"] = mobility.load_state_inputs(ctx["item"])

    def _build(ctx: dict) -> None:
        ctx["result"] = mobility.build_transmission_matrix(ctx["edges_df"], ctx["pop_df"], theta=cfg.THETA, beta0=cfg.BETA0)

    def _save(ctx: dict) -> None:
        state = ctx["item"]
        result = ctx["result"]

        out_dir = cfg.state_dir(state, "mobility")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mobility_{state}.npz"
        np.savez(
            out_path,
            counties=np.array(result.counties, dtype=object),
            W=result.W,
            N=result.N,
            P=result.P,
            D=result.D,
            C=result.C,
            B=result.B,
            theta=result.theta,
            beta0=result.beta0,
        )
        progress.deferred_write(f"    {state}: {len(result.counties)} counties -> {out_path}")

    progress.run_pipeline(
        "Mobility transmission matrix",
        states,
        [("📥", "load", _load), ("🧮", "build", _build), ("💾", "save", _save)],
        item_label="states",
    )


def stage_seed_selection() -> None:
    states = cfg.STATES
    quantiles = cfg.SEED_QUANTILES

    def _load_mobility(state: str) -> dict:
        path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] if k not in ("counties", "theta", "beta0") else data[k] for k in ("counties", "N", "W", "P", "D", "C", "B", "theta", "beta0")}

    def _process_state(ctx: dict) -> None:
        state = ctx["item"]
        data = _load_mobility(state)
        counties = list(data["counties"])
        N, W = data["N"], data["W"]

        m = mobility.mobility_rate(W, N)
        seeds = mobility.select_seed_counties(m, counties, N, quantiles=quantiles)

        out_dir = cfg.state_dir(state, "mobility")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"seed_counties_{state}.csv"
        with open(csv_path, "w") as f:
            f.write("quantile,county,county_index,m_i,population\n")
            for s in seeds:
                f.write(f"{s.quantile},{s.county},{s.county_index},{s.m},{int(s.population)}\n")
        progress.deferred_write(f"    {state}: {len(seeds)} seed counties selected -> {csv_path}")

    progress.run_pipeline(
        "Seed-county selection", states, [("🌱", "select", _process_state)],
        item_label="states",
    )


def stage_deterministic_reference() -> None:
    states = cfg.STATES

    def _load(ctx: dict) -> None:
        state = ctx["item"]
        path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(path, allow_pickle=True) as data:
            ctx["counties"] = list(data["counties"])
            ctx["N"] = data["N"]
            ctx["B"] = data["B"]

    def _simulate(ctx: dict) -> None:
        N = ctx["N"]
        seed_index = int(np.argmax(N))
        ctx["seed_index"] = seed_index
        ctx["result"] = sir.simulate_deterministic(
            ctx["B"], N, cfg.GAMMA, seed_index=seed_index,
            n_seed_infected=sir.DEFAULT_N_SEED_INFECTED_DETERMINISTIC,
            t_span=sir.DEFAULT_T_SPAN, dt_output=sir.DEFAULT_DT_OUTPUT,
            counties=ctx["counties"],
        )

    def _save(ctx: dict) -> None:
        state = ctx["item"]
        result = ctx["result"]
        counties = ctx["counties"]
        seed_index = ctx["seed_index"]

        out_dir = cfg.state_dir(state, "validation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"deterministic_sir_{state}.npz"
        np.savez(
            out_path, t=result.t, S=result.S, X=result.X, R=result.R,
            counties=np.array(counties, dtype=object), seed_index=seed_index,
            seed_county=counties[seed_index], gamma=result.gamma,
            n_seed_infected=result.n_seed_infected,
            t_span=np.array(sir.DEFAULT_T_SPAN), dt_output=sir.DEFAULT_DT_OUTPUT,
        )
        progress.deferred_write(f"    {state}: seed={counties[seed_index]} -> {out_path}")

    progress.run_pipeline(
        "Deterministic SIR reference",
        states,
        [("📥", "load", _load), ("🧮", "simulate", _simulate), ("💾", "save", _save)],
        item_label="states",
    )


def stage_stochastic_reference() -> None:
    states = cfg.STATES

    def _load(ctx: dict) -> None:
        state = ctx["item"]
        path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(path, allow_pickle=True) as data:
            ctx["counties"] = list(data["counties"])
            ctx["N"] = data["N"]
            ctx["B"] = data["B"]

    def _simulate(ctx: dict) -> None:
        N = ctx["N"]
        seed_index = int(np.argmax(N))
        ctx["seed_index"] = seed_index
        ctx["result"] = sir.simulate_stochastic(
            ctx["B"], N, cfg.GAMMA, seed_index=seed_index, seed=cfg.SEED,
            n_seed_infected=sir.DEFAULT_N_SEED_INFECTED_STOCHASTIC,
            dt=sir.DEFAULT_DT, horizon=sir.DEFAULT_HORIZON, counties=ctx["counties"],
        )

    def _save(ctx: dict) -> None:
        state = ctx["item"]
        result = ctx["result"]
        counties = ctx["counties"]
        seed_index = ctx["seed_index"]

        out_dir = cfg.state_dir(state, "validation")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stochastic_sir_{state}.npz"
        np.savez(
            out_path, t=result.t, S=result.S, I=result.I, R=result.R,
            counties=np.array(counties, dtype=object), seed_index=seed_index,
            seed_county=counties[seed_index], seed=result.seed, gamma=result.gamma,
            dt=result.dt, n_seed_infected=result.n_seed_infected, stopped_at=result.stopped_at,
        )
        progress.deferred_write(f"    {state}: seed_county={counties[seed_index]} -> {out_path}")

    progress.run_pipeline(
        "Stochastic SIR simulation",
        states,
        [("📥", "load", _load), ("🎲", "simulate", _simulate), ("💾", "save", _save)],
        item_label="states",
    )


def stage_nominal_drn_reference() -> None:
    states = cfg.STATES

    def _load(ctx: dict) -> None:
        state = ctx["item"]
        mobility_path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        stochastic_path = cfg.state_dir(state, "validation") / f"stochastic_sir_{state}.npz"

        with np.load(mobility_path, allow_pickle=True) as data:
            ctx["counties"] = list(data["counties"])
            ctx["N"] = data["N"]
            ctx["B"] = data["B"]
        with np.load(stochastic_path, allow_pickle=True) as data:
            ctx["t"], ctx["S"], ctx["I"] = data["t"], data["S"], data["I"]

    def _compute(ctx: dict) -> None:
        ctx["drn_result"] = drn.compute_drn(ctx["S"], ctx["I"], ctx["N"], ctx["B"], gamma=cfg.GAMMA, primary_threshold=drn.DEFAULT_PRIMARY_THRESHOLD)

    def _save(ctx: dict) -> None:
        state = ctx["item"]
        counties = ctx["counties"]
        drn_result = ctx["drn_result"]

        out_dir = cfg.state_dir(state, "validation")
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"nominal_drn_{state}.npz"
        np.savez(
            npz_path, t=ctx["t"], drn=drn_result.drn, defined_mask=drn_result.defined_mask,
            primary_mask=drn_result.primary_mask, counties=np.array(counties, dtype=object),
            gamma=cfg.GAMMA, primary_threshold=drn_result.primary_threshold,
        )
        progress.deferred_write(f"    {state}: {len(counties)} counties -> {npz_path}")

    progress.run_pipeline(
        "Nominal DRN + availability (validation reference)",
        states,
        [("📥", "load", _load), ("🧮", "compute", _compute), ("💾", "save", _save)],
        item_label="states",
    )


_TRUTH_MANIFEST_FIELDS = ["state", "seed_quantile", "seed_county", "realization", "rng_seed", "output_path", "validation_status"]


def _load_mobility_for_truth(state: str) -> dict:
    path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
    with np.load(path, allow_pickle=True) as data:
        return {"counties": list(data["counties"]), "N": data["N"], "B": data["B"], "theta": float(data["theta"]), "beta0": float(data["beta0"])}


def _load_seed_counties(state: str) -> list[dict]:
    path = cfg.state_dir(state, "mobility") / f"seed_counties_{state}.csv"
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"quantile": float(row["quantile"]), "county": row["county"], "county_index": int(row["county_index"])})
    return rows


def _generate_one_truth_trajectory(state: str, mobility_data: dict, seed_entry: dict, realization: int) -> dict:
    county = seed_entry["county"]
    county_idx = seed_entry["county_index"]
    N, B, counties = mobility_data["N"], mobility_data["B"], mobility_data["counties"]
    results_root = cfg.results_root()

    out_dir = cfg.state_dir(state, "truth") / _slug(county)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"realization_{realization:02d}.npz"

    seed = sir.trajectory_seed(cfg.SEED, state, county, realization)
    n_steps = round(cfg.HORIZON_DAYS / cfg.DT_DAYS)
    expected_t = cfg.DT_DAYS * np.arange(n_steps + 1)
    expected_meta = {
        "seed": seed, "seed_county": county, "seed_quantile": seed_entry["quantile"],
        "seed_county_index": county_idx, "gamma": cfg.GAMMA, "dt": cfg.DT_DAYS,
        "horizon": cfg.HORIZON_DAYS, "n_seed_infected": cfg.N_SEED_INFECTED,
        "theta": mobility_data["theta"], "beta0": mobility_data["beta0"], "t": expected_t,
    }

    if out_path.exists():
        with np.load(out_path, allow_pickle=True) as data:
            t, S, I, R = data["t"], data["S"], data["I"], data["R"]
            saved_meta = {
                "seed": int(data["seed"]), "seed_county": str(data["seed_county"]),
                "seed_quantile": float(data["seed_quantile"]), "seed_county_index": int(data["seed_county_index"]),
                "gamma": float(data["gamma"]), "dt": float(data["dt"]), "horizon": float(data["horizon"]),
                "n_seed_infected": int(data["n_seed_infected"]), "theta": float(data["theta"]),
                "beta0": float(data["beta0"]), "t": t,
            }
        check = sir.validate_resume_metadata(saved_meta, expected_meta)
        if not check.ok:
            raise ValueError(f"existing file's generation metadata does not match the current config ({check.reason}); remove {out_path} to regenerate.")
        status = "resumed"
    else:
        result = sir.simulate_stochastic(
            B, N, cfg.GAMMA, seed_index=county_idx, seed=seed, n_seed_infected=cfg.N_SEED_INFECTED,
            dt=cfg.DT_DAYS, horizon=cfg.HORIZON_DAYS, counties=counties,
        )
        t, S, I, R = result.t, result.S, result.I, result.R
        np.savez(
            out_path, t=t, S=S, I=I, R=R, N=N, counties=np.array(counties, dtype=object),
            state=state, seed_county=county, seed_county_index=county_idx,
            seed_quantile=seed_entry["quantile"], realization=realization, seed=seed, gamma=cfg.GAMMA,
            dt=cfg.DT_DAYS, horizon=cfg.HORIZON_DAYS, n_seed_infected=cfg.N_SEED_INFECTED,
            stopped_at=result.stopped_at, theta=mobility_data["theta"], beta0=mobility_data["beta0"],
        )
        status = "generated"

    validation = sir.validate_trajectory(t, S, I, R, N)
    validation_status = "valid" if validation.ok else f"invalid: {validation.reason}"

    return {
        "status": status, "ok": validation.ok,
        "row": {
            "state": state, "seed_quantile": seed_entry["quantile"], "seed_county": county,
            "realization": realization, "rng_seed": seed,
            "output_path": str(out_path.relative_to(results_root)), "validation_status": validation_status,
        },
    }


def stage_truth_trajectory_generation() -> None:
    states, n_realizations = cfg.STATES, cfg.N_TRUTH_REALIZATIONS
    manifest_path = cfg.state_manifest_path("truth_trajectory_manifest.csv")

    manifest_rows: list[dict] = []
    n_errors = 0

    def _generate_state(ctx: dict) -> None:
        nonlocal n_errors
        state = ctx["item"]
        mobility_data = _load_mobility_for_truth(state)
        seed_counties = _load_seed_counties(state)
        n_valid = 0

        for i, seed_entry in enumerate(seed_counties):
            county = seed_entry["county"]
            for realization in range(n_realizations):
                progress.step_info["seed county"] = f"{county} ({i + 1}/{len(seed_counties)})"
                progress.step_info["realization"] = f"{realization + 1}/{n_realizations}"

                try:
                    outcome = _generate_one_truth_trajectory(state, mobility_data, seed_entry, realization)
                except Exception as exc:
                    n_errors += 1
                    progress.deferred_write(f"    ERROR {state}/{county}/realization {realization}: {exc}")
                    manifest_rows.append({
                        "state": state, "seed_quantile": seed_entry["quantile"], "seed_county": county,
                        "realization": realization, "rng_seed": "", "output_path": "", "validation_status": f"error: {exc}",
                    })
                    continue

                n_valid += int(outcome["ok"])
                manifest_rows.append(outcome["row"])

        progress.deferred_write(f"    {state}: {len(seed_counties) * n_realizations} trajectories, {n_valid} valid")

    progress.run_pipeline(
        "Truth-trajectory generation", states, [("🎲", "generate", _generate_state)],
        item_label="states",
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TRUTH_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if n_errors:
        raise RuntimeError(f"truth_trajectory_generation: {n_errors} trajectories failed; see manifest for details.")


_NOMINAL_DRN_MANIFEST_FIELDS = ["state", "seed_quantile", "seed_county", "realization", "source_seed", "source_path", "output_path", "validation_status"]


def _b_hash(B: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(B, dtype=float).tobytes()).hexdigest()


def _generate_one_nominal_drn(state: str, mobility_data: dict, truth_row: dict) -> dict:
    county = truth_row["seed_county"]
    realization = int(truth_row["realization"])
    source_path = truth_row["output_path"]
    results_root = cfg.results_root()

    with np.load(results_root / source_path, allow_pickle=True) as data:
        t = data["t"]
        S, I, N = data["S"], data["I"], data["N"]
        counties = list(data["counties"])
        source_meta = {
            "source_seed": int(data["seed"]), "source_horizon": float(data["horizon"]), "source_dt": float(data["dt"]),
            "source_n_seed_infected": int(data["n_seed_infected"]), "source_gamma": float(data["gamma"]),
            "source_theta": float(data["theta"]), "source_beta0": float(data["beta0"]),
            "seed_county_index": int(data["seed_county_index"]),
        }

    out_dir = cfg.state_dir(state, "nominal_drn") / _slug(county)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"realization_{realization:02d}.npz"

    availability_thresholds = np.asarray(sorted(cfg.AVAILABILITY_THRESHOLDS), dtype=int)
    expected_meta = {
        "state": state, "seed_county": county, "seed_quantile": float(truth_row["seed_quantile"]),
        "realization": realization, "source_path": source_path, "B_hash": mobility_data["B_hash"],
        "gamma_used": cfg.GAMMA, "primary_threshold_used": cfg.PRIMARY_THRESHOLD,
        "availability_thresholds_used": availability_thresholds, "n_times": t.shape[0], "n_counties": I.shape[1],
        "t": t, **source_meta,
    }

    if out_path.exists():
        with np.load(out_path, allow_pickle=True) as data:
            saved_meta = {
                "state": str(data["state"]), "seed_county": str(data["seed_county"]), "seed_quantile": float(data["seed_quantile"]),
                "realization": int(data["realization"]), "source_path": str(data["source_path"]), "B_hash": str(data["B_hash"]),
                "gamma_used": float(data["gamma_used"]), "primary_threshold_used": int(data["primary_threshold_used"]),
                "availability_thresholds_used": data["availability_thresholds_used"], "n_times": int(data["n_times"]),
                "n_counties": int(data["n_counties"]), "t": data["t"], "source_seed": int(data["source_seed"]),
                "source_horizon": float(data["source_horizon"]), "source_dt": float(data["source_dt"]),
                "source_n_seed_infected": int(data["source_n_seed_infected"]), "source_gamma": float(data["source_gamma"]),
                "source_theta": float(data["source_theta"]), "source_beta0": float(data["source_beta0"]),
                "seed_county_index": int(data["seed_county_index"]),
            }
            check = sir.validate_resume_metadata(saved_meta, expected_meta)
            if not check.ok:
                raise ValueError(f"existing nominal-DRN file's metadata does not match current trajectory/mobility/config ({check.reason}); remove {out_path} to recompute.")
            result = drn.ReferenceDRNResult(
                drn=data["drn"], defined_mask=data["defined_mask"], primary_mask=data["primary_mask"],
                availability_thresholds=data["availability_thresholds"], availability_masks=data["availability_masks"],
                p_ii=data["p_ii"], q=data["q"], q_hat=data["q_hat"], q_defined_mask=data["q_defined_mask"],
            )
        status = "resumed"
    else:
        result = drn.compute_reference_drn(S, I, N, mobility_data["B"], cfg.GAMMA, cfg.PRIMARY_THRESHOLD, availability_thresholds)
        np.savez_compressed(
            out_path, counties=np.array(counties, dtype=object), drn=result.drn, defined_mask=result.defined_mask,
            primary_mask=result.primary_mask, availability_thresholds=result.availability_thresholds,
            availability_masks=result.availability_masks, p_ii=result.p_ii, q=result.q, q_hat=result.q_hat,
            q_defined_mask=result.q_defined_mask, **expected_meta,
        )
        status = "generated"

    validation = drn.validate_reference_drn(result, I, cfg.PRIMARY_THRESHOLD)
    validation_status = "valid" if validation.ok else f"invalid: {validation.reason}"

    return {
        "status": status, "ok": validation.ok,
        "row": {
            "state": state, "seed_quantile": truth_row["seed_quantile"], "seed_county": county, "realization": realization,
            "source_seed": source_meta["source_seed"], "source_path": source_path,
            "output_path": str(out_path.relative_to(results_root)), "validation_status": validation_status,
        },
    }


def stage_nominal_drn_generation() -> None:
    truth_manifest_path = cfg.state_manifest_path("truth_trajectory_manifest.csv")
    manifest_path = cfg.state_manifest_path("nominal_drn_manifest.csv")

    with open(truth_manifest_path, newline="") as f:
        truth_rows = [row for row in csv.DictReader(f) if row["validation_status"] == "valid"]

    mobility_cache: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    n_errors = 0
    rows_by_state: dict[str, list[dict]] = {}
    for row in truth_rows:
        rows_by_state.setdefault(row["state"], []).append(row)

    def _generate_state(ctx: dict) -> None:
        nonlocal n_errors
        state = ctx["item"]
        rows = rows_by_state[state]
        path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(path, allow_pickle=True) as data:
            B = data["B"]
            mobility_cache[state] = {"B": B, "counties": list(data["counties"]), "B_hash": _b_hash(B)}
        n_valid = 0

        for i, row in enumerate(rows):
            county = row["seed_county"]
            progress.step_info["trajectory"] = f"{i + 1}/{len(rows)}"
            try:
                outcome = _generate_one_nominal_drn(state, mobility_cache[state], row)
            except Exception as exc:
                n_errors += 1
                progress.deferred_write(f"    ERROR {state}/{county}/realization {row['realization']}: {exc}")
                manifest_rows.append({
                    "state": state, "seed_quantile": row["seed_quantile"], "seed_county": county, "realization": row["realization"],
                    "source_seed": "", "source_path": row["output_path"], "output_path": "", "validation_status": f"error: {exc}",
                })
                continue

            n_valid += int(outcome["ok"])
            manifest_rows.append(outcome["row"])

        progress.deferred_write(f"    {state}: {len(rows)} processed, {n_valid} valid")

    progress.run_pipeline(
        "Nominal DRN generation", sorted(rows_by_state), [("🧮", "compute", _generate_state)],
        item_label="states",
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_NOMINAL_DRN_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if n_errors:
        raise RuntimeError(f"nominal_drn_generation: {n_errors} trajectories failed; see manifest for details.")


_RADIUS_MANIFEST_FIELDS = ["state", "seed_quantile", "seed_county", "realization", "nominal_drn_path", "output_path", "validation_status"]


def _generate_one_radius(nominal_row: dict) -> dict:
    state = nominal_row["state"]
    county = nominal_row["seed_county"]
    realization = int(nominal_row["realization"])
    nominal_path = nominal_row["output_path"]
    results_root = cfg.results_root()

    with np.load(results_root / nominal_path, allow_pickle=True) as data:
        t = data["t"]
        counties = list(data["counties"])
        hat_drn = data["drn"]
        q_hat = data["q_hat"]
        primary_mask = data["primary_mask"]
        source_meta = {
            "source_path": str(data["source_path"]), "B_hash": str(data["B_hash"]),
            "gamma_used": float(data["gamma_used"]), "primary_threshold_used": int(data["primary_threshold_used"]),
        }

    with np.load(results_root / source_meta["source_path"], allow_pickle=True) as data:
        I = data["I"]
    I_primary = I[primary_mask]

    out_dir = cfg.state_dir(state, "robustness_radius") / _slug(county)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"realization_{realization:02d}.npz"

    expected_meta = {
        "state": state, "seed_county": county, "seed_quantile": float(nominal_row["seed_quantile"]),
        "realization": realization, "nominal_drn_path": nominal_path, "n_times": t.shape[0],
        "n_counties": hat_drn.shape[1], "t": t, **source_meta,
    }

    if out_path.exists():
        with np.load(out_path, allow_pickle=True) as data:
            saved_meta = {
                "state": str(data["state"]), "seed_county": str(data["seed_county"]), "seed_quantile": float(data["seed_quantile"]),
                "realization": int(data["realization"]), "nominal_drn_path": str(data["nominal_drn_path"]),
                "n_times": int(data["n_times"]), "n_counties": int(data["n_counties"]), "t": data["t"],
                "source_path": str(data["source_path"]), "B_hash": str(data["B_hash"]),
                "gamma_used": float(data["gamma_used"]), "primary_threshold_used": int(data["primary_threshold_used"]),
            }
            check = sir.validate_resume_metadata(saved_meta, expected_meta)
            if not check.ok:
                raise ValueError(f"existing robustness-radius file's metadata does not match current nominal-DRN/config ({check.reason}); remove {out_path} to recompute.")
            u_star = data["u_star"]
        status = "resumed"
    else:
        u_star = robustness.robustness_radius(hat_drn, q_hat)
        np.savez_compressed(
            out_path, counties=np.array(counties, dtype=object), u_star=u_star, hat_drn=hat_drn, q_hat=q_hat,
            primary_mask=primary_mask, I_primary=I_primary, **expected_meta,
        )
        status = "generated"

    validation = robustness.validate_radius(u_star, hat_drn, q_hat, primary_mask)
    validation_status = "valid" if validation.ok else f"invalid: {validation.reason}"

    return {
        "status": status, "ok": validation.ok,
        "row": {
            "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": county, "realization": realization,
            "nominal_drn_path": nominal_path, "output_path": str(out_path.relative_to(results_root)), "validation_status": validation_status,
        },
    }


def stage_robustness_radius_generation() -> None:
    nominal_manifest_path = cfg.state_manifest_path("nominal_drn_manifest.csv")
    manifest_path = cfg.state_manifest_path("robustness_radius_manifest.csv")

    with open(nominal_manifest_path, newline="") as f:
        nominal_rows = [row for row in csv.DictReader(f) if row["validation_status"] == "valid"]

    manifest_rows: list[dict] = []
    n_errors = 0
    rows_by_state: dict[str, list[dict]] = {}
    for nominal_row in nominal_rows:
        rows_by_state.setdefault(nominal_row["state"], []).append(nominal_row)

    def _generate_state(ctx: dict) -> None:
        nonlocal n_errors
        state = ctx["item"]
        rows = rows_by_state[state]
        n_valid = 0
        for i, nominal_row in enumerate(rows):
            progress.step_info["trajectory"] = f"{i + 1}/{len(rows)}"
            try:
                outcome = _generate_one_radius(nominal_row)
            except Exception as exc:
                n_errors += 1
                progress.deferred_write(f"    ERROR {state}/{nominal_row['seed_county']}/realization {nominal_row['realization']}: {exc}")
                manifest_rows.append({
                    "state": state, "seed_quantile": nominal_row["seed_quantile"], "seed_county": nominal_row["seed_county"],
                    "realization": nominal_row["realization"], "nominal_drn_path": nominal_row["output_path"],
                    "output_path": "", "validation_status": f"error: {exc}",
                })
                continue

            n_valid += int(outcome["ok"])
            manifest_rows.append(outcome["row"])

        progress.deferred_write(f"    {state}: {len(rows)} processed, {n_valid} valid")

    progress.run_pipeline(
        "Robustness-radius generation", sorted(rows_by_state), [("🧮", "compute", _generate_state)],
        item_label="states",
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RADIUS_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if n_errors:
        raise RuntimeError(f"robustness_radius_generation: {n_errors} trajectories failed; see manifest for details.")


STAGES = {
    "mobility": stage_mobility,
    "seed_selection": stage_seed_selection,
    "deterministic_reference": stage_deterministic_reference,
    "stochastic_reference": stage_stochastic_reference,
    "nominal_drn_reference": stage_nominal_drn_reference,
    "truth_trajectory_generation": stage_truth_trajectory_generation,
    "nominal_drn_generation": stage_nominal_drn_generation,
    "robustness_radius_generation": stage_robustness_radius_generation,
}

STAGE_ICONS = {
    "mobility": "🌐",
    "seed_selection": "🌱",
    "deterministic_reference": "📈",
    "stochastic_reference": "🎲",
    "nominal_drn_reference": "🧮",
    "truth_trajectory_generation": "🧬",
    "nominal_drn_generation": "🧮",
    "robustness_radius_generation": "🛡️",
}


def main() -> None:
    def _run_stage(ctx: dict) -> None:
        STAGES[ctx["item"]]()

    progress.run_pipeline(
        "1_load_data.py", list(STAGES), [("🚀", "run", _run_stage)],
        item_label="stages",
        item_format=lambda name: f"{STAGE_ICONS.get(name, '🔧')} {name}",
    )


if __name__ == "__main__":
    main()
