from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import drn
import mobility
import plotting
import progress


_NETWORK_GRID_ORDER = ("california", "kansas", "florida")


def stage_network() -> None:
    data_by_state = {}
    for state in cfg.STATES:
        mobility_path = cfg.state_dir(state, "mobility") / f"mobility_{state}.npz"
        with np.load(mobility_path, allow_pickle=True) as data:
            counties = list(data["counties"])
            N = data["N"]

        edges_df, _ = mobility.load_state_inputs(state)
        pop_data = dict(zip(counties, N))
        data_by_state[state] = (edges_df, pop_data)

        out_path = cfg.figures_dir() / f"network_{state}.png"
        plotting.plot_network(edges_df, pop_data, out_path, state_name=state)

    if set(_NETWORK_GRID_ORDER) <= set(data_by_state):
        combined_out = cfg.figures_dir() / "fig1_mobility_networks.png"
        plotting.plot_network_grid(data_by_state, combined_out, order=_NETWORK_GRID_ORDER)

def _read_manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.state_manifest_path(name))


def _read_analysis_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.analysis_manifest_path(name))


def _bootstrap_row(bootstrap_df: pd.DataFrame, analysis: str, metric: str, severity, view: str) -> tuple[float, float, float]:
    sub = bootstrap_df[
        (bootstrap_df["analysis"] == analysis) & (bootstrap_df["metric"] == metric) & (bootstrap_df["view"] == view)
    ]
    sub = sub[sub["severity"].isna()] if severity is None else sub[sub["severity"].astype(str) == str(severity)]
    r = sub.iloc[0]
    return float(r["point_estimate"]), float(r["ci_lower"]), float(r["ci_upper"])


def _bootstrap_pooled_series(bootstrap_df: pd.DataFrame, analysis: str, metric: str, severities: list) -> pd.DataFrame:
    rows = [
        dict(zip(("point", "lo", "hi"), _bootstrap_row(bootstrap_df, analysis, metric, sev, "overall")), severity=sev)
        for sev in severities
    ]
    return pd.DataFrame(rows)


def _load_epidemic_ensemble(truth_manifest: pd.DataFrame) -> dict:
    root = cfg.results_root()
    by_state = {}
    for state in cfg.STATES:
        rows = truth_manifest[(truth_manifest["state"] == state) & (truth_manifest["validation_status"] == "valid")]
        series = []
        t_ref = None
        for _, row in rows.iterrows():
            with np.load(root / row["output_path"], allow_pickle=True) as data:
                t = data["t"]
                I = data["I"]
                N = data["N"]
            if t_ref is None:
                t_ref = t
            series.append(100.0 * I.sum(axis=1) / N.sum())

        stacked = np.vstack(series)
        by_state[state] = {
            "t": t_ref,
            "median": np.median(stacked, axis=0),
            "q25": np.percentile(stacked, 25, axis=0),
            "q75": np.percentile(stacked, 75, axis=0),
            "n": stacked.shape[0],
        }
    return by_state


def _load_availability_by_state(nominal_manifest: pd.DataFrame) -> pd.DataFrame:
    root = cfg.results_root()
    valid = nominal_manifest[nominal_manifest["validation_status"] == "valid"]
    records = []
    for _, row in valid.iterrows():
        with np.load(root / row["output_path"], allow_pickle=True) as data:
            thresholds = data["availability_thresholds"]
            masks = data["availability_masks"]
        for thr, mask in zip(thresholds, masks):
            records.append({"state": row["state"], "threshold": int(thr), "fraction": float(mask.mean())})

    return pd.DataFrame(records)


def stage_fig3_epidemic_coverage() -> dict:
    truth_manifest = _read_manifest("truth_trajectory_manifest.csv")
    nominal_manifest = _read_manifest("nominal_drn_manifest.csv")

    epidemic_by_state = _load_epidemic_ensemble(truth_manifest)
    availability_df = _load_availability_by_state(nominal_manifest)

    pdf_path, png_path, png_px = plotting.plot_fig3_epidemic_coverage(
        epidemic_by_state, availability_df, cfg.STATES, cfg.figures_dir(), primary_threshold=cfg.PRIMARY_THRESHOLD
    )

    notes = [f"{state}: {epidemic_by_state[state]['n']} truth trajectories pooled" for state in cfg.STATES]
    notes.append(f"availability rows: {len(availability_df)}")
    return {
        "name": "fig3_epidemic_coverage",
        "outputs": [(pdf_path, "vector PDF"), (png_path, f"PNG {png_px[0]}x{png_px[1]}px @600dpi")],
        "sources": [
            str(cfg.state_manifest_path("truth_trajectory_manifest.csv")),
            str(cfg.state_manifest_path("nominal_drn_manifest.csv")),
            "300 truth-trajectory .npz files (states/<state>/truth/)",
            "300 nominal-DRN .npz files (states/<state>/nominal_drn/)",
        ],
        "notes": notes,
    }


def _load_pooled_radius_observations(radius_manifest: pd.DataFrame) -> dict:
    root = cfg.results_root()
    valid = radius_manifest[radius_manifest["validation_status"] == "valid"]
    hat_drn_parts, q_hat_parts, u_star_parts = [], [], []
    n_primary_total = 0
    for _, row in valid.iterrows():
        with np.load(root / row["output_path"], allow_pickle=True) as data:
            hat_drn = data["hat_drn"]
            q_hat = data["q_hat"]
            u_star = data["u_star"]
            primary_mask = data["primary_mask"]
        n_primary_total += int(primary_mask.sum())
        finite = primary_mask & np.isfinite(hat_drn) & np.isfinite(q_hat) & np.isfinite(u_star)
        hat_drn_parts.append(hat_drn[finite])
        q_hat_parts.append(q_hat[finite])
        u_star_parts.append(u_star[finite])

    hat_drn_all = np.concatenate(hat_drn_parts)
    q_hat_all = np.concatenate(q_hat_parts)
    u_star_all = np.concatenate(u_star_parts)
    return {
        "hat_drn": hat_drn_all,
        "q_hat": q_hat_all,
        "u_star": u_star_all,
        "n_primary_total": n_primary_total,
        "n_finite": int(hat_drn_all.size),
    }


def _pooled_classification_by_u(traj_df: pd.DataFrame, severities: list, states: list) -> pd.DataFrame:
    rows = []
    for sev in severities:
        sub = traj_df[np.isclose(traj_df["u"].astype(float), float(sev))]
        n_avail = sub["n_available"].to_numpy(dtype=float)
        total = float(n_avail.sum())
        growth = float((sub["certified_growth_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        decline = float((sub["certified_decline_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        indeterminate = float((sub["indeterminate_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        rows.append({
            "severity": sev, "n_trajectories": len(sub), "n_available_total": int(total),
            "growth": growth, "decline": decline, "indeterminate": indeterminate,
        })
    return pd.DataFrame(rows)


def _build_survival_curve(pooled: dict) -> dict:
    u_star = pooled["u_star"]
    u_sorted = np.sort(u_star)
    n = u_sorted.size
    grid = np.linspace(0.0, 1.0, 2001)
    idx = np.searchsorted(u_sorted, grid, side="left")
    survival_pct = 100.0 * (n - idx) / n
    return {
        "grid": grid,
        "survival_pct": survival_pct,
        "median": float(np.median(u_star)),
        "pct_u_ge_0_6": float(100.0 * np.mean(u_star >= 0.6)),
        "n": int(n),
    }


def _load_trajectory_group_medians(radius_manifest: pd.DataFrame) -> pd.DataFrame:
    root = cfg.results_root()
    valid = radius_manifest[radius_manifest["validation_status"] == "valid"]
    rows = []
    for _, row in valid.iterrows():
        with np.load(root / row["output_path"], allow_pickle=True) as data:
            hat_drn = data["hat_drn"]
            q_hat = data["q_hat"]
            u_star = data["u_star"]
            primary_mask = data["primary_mask"]
        finite = primary_mask & np.isfinite(hat_drn) & np.isfinite(q_hat) & np.isfinite(u_star)
        dist = np.abs(hat_drn[finite] - 1.0)
        q = q_hat[finite]
        u = u_star[finite]
        n_eligible = dist.size

        n_near = max(5, int(round(0.10 * n_eligible)))
        near_idx = np.argsort(dist)[:n_near]
        q_near = q[near_idx]
        u_near = u[near_idx]

        order_by_q = np.argsort(q_near, kind="stable")
        group_splits = np.array_split(order_by_q, 5)
        for gi, idxs in enumerate(group_splits):
            rows.append({
                "state": row["state"], "seed_county": row["seed_county"], "realization": int(row["realization"]),
                "group": gi + 1, "n": int(idxs.size), "median_u_star": float(np.median(u_near[idxs])),
            })
    return pd.DataFrame(rows)


def stage_fig4_underreporting_robustness() -> dict:
    traj_df = _read_analysis_csv("full_study_bounded_state_underreporting_trajectories.csv")
    traj_df = traj_df[traj_df["count_group"] == "overall"].copy()
    radius_manifest = _read_manifest("robustness_radius_manifest.csv")

    severities = list(cfg.STATE_U_LEVELS)
    classification_df = _pooled_classification_by_u(traj_df, severities, cfg.STATES)

    pooled_radius = _load_pooled_radius_observations(radius_manifest)
    survival = _build_survival_curve(pooled_radius)
    group_df = _load_trajectory_group_medians(radius_manifest)

    pdf_path, png_path, png_px = plotting.plot_fig4_underreporting_robustness(
        classification_df, survival, group_df, cfg.figures_dir()
    )

    classification_text = ", ".join(
        f"u={r.severity}: growth={r.growth * 100:.2f}%, decline={r.decline * 100:.2f}%, indeterminate={r.indeterminate * 100:.4f}%"
        for r in classification_df.itertuples()
    )
    notes = [
        f"panel A pooled classification by u: {classification_text}",
        f"panel B pooled primary-mask observations: {pooled_radius['n_primary_total']} total, {pooled_radius['n_finite']} finite",
        f"panel B median robustness radius u*: {survival['median']:.4f}",
        f"panel B percentage with u* >= 0.6: {survival['pct_u_ge_0_6']:.4f}%",
        "panel C: per trajectory, the 10% of primary-mask observations closest to the DRN threshold are split into 5 equal-count q groups; each box is that group's per-trajectory medians",
    ]
    return {
        "name": "fig4_underreporting_robustness",
        "outputs": [(pdf_path, "vector PDF"), (png_path, f"PNG {png_px[0]}x{png_px[1]}px @600dpi")],
        "sources": [
            str(cfg.analysis_manifest_path("full_study_bounded_state_underreporting_trajectories.csv")),
            str(cfg.state_manifest_path("robustness_radius_manifest.csv")),
            "300 robustness-radius .npz files (states/<mobility network>/robustness_radius/)",
        ],
        "notes": notes,
    }


def _pooled_classification_by_b(traj_df: pd.DataFrame, severities: list, states: list) -> pd.DataFrame:
    rows = []
    for sev in severities:
        sub = traj_df[np.isclose(traj_df["b"].astype(float), float(sev))]
        n_avail = sub["n_available"].to_numpy(dtype=float)
        total = float(n_avail.sum())
        growth = float((sub["certified_growth_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        decline = float((sub["certified_decline_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        indeterminate = float((sub["indeterminate_rate"].to_numpy(dtype=float) * n_avail).sum() / total)
        rows.append({
            "severity": sev, "n_trajectories": len(sub), "n_available_total": int(total),
            "growth": growth, "decline": decline, "indeterminate": indeterminate,
        })
    return pd.DataFrame(rows)


def stage_fig5_coupling_uncertainty() -> dict:
    bootstrap_df = _read_analysis_csv("bootstrap_confidence_intervals.csv")
    traj_df = _read_analysis_csv("full_study_bounded_coupling_uncertainty_trajectories.csv")
    traj_df = traj_df[(traj_df["count_group"] == "overall") & (traj_df["regime"] == "overall")].copy()

    severities = list(cfg.COUPLING_B_LEVELS)

    classification_df = _pooled_classification_by_b(traj_df, severities, cfg.STATES)
    indeterminate_df = _bootstrap_pooled_series(bootstrap_df, "coupling_uncertainty", "indeterminate_rate", severities)
    misclass_df = _bootstrap_pooled_series(bootstrap_df, "coupling_uncertainty", "point_misclassification_rate", severities)

    pdf_path, png_path, png_px = plotting.plot_fig5_coupling_uncertainty(
        classification_df, indeterminate_df, misclass_df, cfg.figures_dir()
    )

    classification_text = ", ".join(
        f"b={r.severity}: growth={r.growth * 100:.2f}%, decline={r.decline * 100:.2f}%, indeterminate={r.indeterminate * 100:.4f}%"
        for r in classification_df.itertuples()
    )
    indeterminate_text = ", ".join(f"b={r.severity}: {r.point * 100:.4f}% [{r.lo * 100:.4f}, {r.hi * 100:.4f}]" for r in indeterminate_df.itertuples())
    misclass_text = ", ".join(f"b={r.severity}: {r.point * 100:.4f}% [{r.lo * 100:.4f}, {r.hi * 100:.4f}]" for r in misclass_df.itertuples())
    notes = [
        f"panel A pooled classification by b: {classification_text}",
        f"panel B pooled indeterminate_rate (trajectory-bootstrap, view='overall'): {indeterminate_text}",
        f"panel B pooled point_misclassification_rate (trajectory-bootstrap, view='overall'): {misclass_text}",
    ]
    return {
        "name": "fig5_coupling_uncertainty",
        "outputs": [(pdf_path, "vector PDF"), (png_path, f"PNG {png_px[0]}x{png_px[1]}px @600dpi")],
        "sources": [
            str(cfg.analysis_manifest_path("bootstrap_confidence_intervals.csv")),
            str(cfg.analysis_manifest_path("full_study_bounded_coupling_uncertainty_trajectories.csv")),
        ],
        "notes": notes,
    }


_MANUSCRIPT_FIGURE_STAGES = {
    "fig3_epidemic_coverage": stage_fig3_epidemic_coverage,
    "fig4_underreporting_robustness": stage_fig4_underreporting_robustness,
    "fig5_coupling_uncertainty": stage_fig5_coupling_uncertainty,
}


_MANUSCRIPT_FIGURE_ORDER = list(_MANUSCRIPT_FIGURE_STAGES)


def _format_report_section(r: dict) -> str:
    lines = [f"== {r['name']} =="]
    for path, kind in r["outputs"]:
        path = Path(path)
        size = path.stat().st_size if path.exists() else None
        lines.append(f"  output: {path} [{kind}]" + (f", {size} bytes" if size is not None else " (MISSING)"))
    for src in r["sources"]:
        lines.append(f"  source: {src}")
    for note in r["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def _write_manuscript_figures_report(results: list[dict]) -> Path:
    report_path = cfg.reports_dir("generation") / "manuscript_figures_report.txt"

    sections: dict[str, str] = {}
    if report_path.exists():
        for block in report_path.read_text().split("\n== ")[1:]:
            name = block.split(" ==", 1)[0]
            sections[name] = "== " + block.rstrip("\n")

    for r in results:
        sections[r["name"]] = _format_report_section(r)

    lines = ["Manuscript figure generation report", ""]
    for name in _MANUSCRIPT_FIGURE_ORDER:
        if name in sections:
            lines.append(sections[name])
            lines.append("")

    return report_path


def _run_manuscript_figure_stage(name: str) -> None:
    result = _MANUSCRIPT_FIGURE_STAGES[name]()
    report_path = _write_manuscript_figures_report([result])
    progress.deferred_write(f"    report -> {report_path}")


STAGES = {
    "network": stage_network,
    "fig3_epidemic_coverage": lambda: _run_manuscript_figure_stage("fig3_epidemic_coverage"),
    "fig4_underreporting_robustness": lambda: _run_manuscript_figure_stage("fig4_underreporting_robustness"),
    "fig5_coupling_uncertainty": lambda: _run_manuscript_figure_stage("fig5_coupling_uncertainty"),
}


STAGE_ICONS = {
    "network": "🌐",
    "fig3_epidemic_coverage": "📊",
    "fig4_underreporting_robustness": "🛡️",
    "fig5_coupling_uncertainty": "📐",
}


def main() -> None:
    def _run_stage(ctx: dict) -> None:
        STAGES[ctx["item"]]()

    progress.run_pipeline(
        "3_make_figures.py", list(STAGES), [("🚀", "run", _run_stage)],
        item_label="stages",
        item_format=lambda name: f"{STAGE_ICONS.get(name, '🔧')} {name}",
    )


if __name__ == "__main__":
    main()
