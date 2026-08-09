import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

SHAPEFILE_PATH = (
    Path(__file__).resolve().parent / "data" / "shapefile" / "cb_2018_us_county_500k.shp"
)
STATE_FIPS = {"kansas": "20", "california": "06", "florida": "12"}


def _geo_positions(G, state_name):
    if not SHAPEFILE_PATH.exists():
        return None, None

    import geopandas as gpd

    gdf = gpd.read_file(SHAPEFILE_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gdf["centroid"] = gdf.centroid

    fips = STATE_FIPS.get(state_name.lower())
    gdf_state = gdf[gdf["STATEFP"] == fips] if fips else gdf

    pos = {}
    for node in G.nodes():
        name = str(node).replace(" County", "").strip()
        match = gdf_state[gdf_state["NAME"] == name]
        if not match.empty:
            centroid = match.iloc[0]["centroid"]
            pos[node] = (centroid.x, centroid.y)

    if len(pos) < G.number_of_nodes():
        return None, None
    return pos, gdf_state


def _network_graph_and_layout(df_edges, state_name):
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", edge_attr="weight", create_using=nx.DiGraph()
    )
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    pos, gdf_state = _geo_positions(G, state_name)
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    return G, pos, gdf_state


def _geo_aspect_ratio(gdf_state) -> float:
    if gdf_state is None:
        return 1.0
    minx, miny, maxx, maxy = gdf_state.total_bounds
    width, height = maxx - minx, maxy - miny
    if height <= 0:
        return 1.0
    mean_lat_rad = np.radians((miny + maxy) / 2.0)
    return (width * np.cos(mean_lat_rad)) / height


def _draw_network_panel(
    ax, G, pos, gdf_state, pop_data, min_node_size=500, node_size_range=2000,
    edge_alpha=0.25, min_edge_width=0.5, edge_width_range=5.0,
) -> None:
    ax.set_xmargin(0)
    ax.set_ymargin(0)

    max_pop = max(pop_data.values()) if pop_data else 1
    node_sizes = [min_node_size + (pop_data.get(n, 1) / max_pop) * node_size_range for n in G.nodes()]

    weights = np.array([float(d["weight"]) for _, _, d in G.edges(data=True)])
    edge_widths = min_edge_width + (weights / weights.max()) * edge_width_range if len(weights) else min_edge_width

    if gdf_state is not None:
        gdf_state.plot(ax=ax, color="#f0f0f5", edgecolor="silver", linewidth=1.5)

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color="#5B7FA6", edgecolors="gray", alpha=0.9
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=edge_alpha,
        edge_color="#85540f",
        arrows=False,
    )
    ax.axis("off")


def plot_network(df_edges, pop_data, output_path, state_name="Network", label_font_size=8):
    G, pos, gdf_state = _network_graph_and_layout(df_edges, state_name)

    fig, ax = plt.subplots(figsize=(16, 12))
    _draw_network_panel(ax, G, pos, gdf_state, pop_data)
    fig.tight_layout(pad=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)



GRID_NODE_SIZES = {
    "kansas": (500, 2000),
    "california": (150, 800),
    "florida": (100, 600),
}
GRID_EDGE_STYLE = {
    "kansas": (0.3, 0.6, 5.0),
    "california": (0.10, 0.05, 1.0),
    "florida": (0.10, 0.05, 1.0),
}


def plot_network_grid(data_by_state: dict, output_path, order=("california", "kansas", "florida")):
    missing = [s for s in order if s not in data_by_state]
    if missing:
        raise ValueError(f"plot_network_grid: missing network data for state(s): {missing}")

    layouts = {state: _network_graph_and_layout(data_by_state[state][0], state) for state in order}
    aspects = [_geo_aspect_ratio(layouts[state][2]) for state in order]

    panel_height = 8.0
    fig = plt.figure(figsize=(panel_height * sum(aspects), panel_height))
    gs = fig.add_gridspec(1, len(order), width_ratios=aspects, wspace=0.0)

    for i, state in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        G, pos, gdf_state = layouts[state]
        _, pop_data = data_by_state[state]
        min_size, size_range = GRID_NODE_SIZES[state]
        edge_alpha, min_edge_width, edge_width_range = GRID_EDGE_STYLE[state]
        _draw_network_panel(
            ax, G, pos, gdf_state, pop_data, min_node_size=min_size, node_size_range=size_range,
            edge_alpha=edge_alpha, min_edge_width=min_edge_width, edge_width_range=edge_width_range,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)



def plot_epidemic_curves(t, S, I, R, output_path, title="Epidemic curve"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, label="Susceptible", color="#1f77b4")
    ax.plot(t, I, label="Infected", color="#d62728")
    ax.plot(t, R, label="Recovered", color="#2ca02c")
    ax.set_xlabel("Day")
    ax.set_ylabel("People")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_drn_availability(t, I, availability, output_path, state_name="", primary_threshold=10):
    I = np.asarray(I)
    n_counties = I.shape[1]

    zero_pct = 100.0 * (I == 0).sum(axis=1) / n_counties
    low_pct = 100.0 * ((I > 0) & (I < primary_threshold)).sum(axis=1) / n_counties
    primary_pct = 100.0 * (I >= primary_threshold).sum(axis=1) / n_counties

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax_a.stackplot(
        t,
        zero_pct,
        low_pct,
        primary_pct,
        labels=["I = 0", f"0 < I < {primary_threshold}", f"I >= {primary_threshold}"],
        colors=["#d9d9d9", "#f2a154", "#3b7dd8"],
    )
    ax_a.set_xlabel("Day")
    ax_a.set_ylabel("Percentage of counties")
    ax_a.set_ylim(0, 100)
    ax_a.set_title("A. County status over time")
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(alpha=0.3)

    thresholds = sorted(availability.keys())
    fractions = [availability[thr].fraction_county_times * 100.0 for thr in thresholds]
    bars = ax_b.bar([str(thr) for thr in thresholds], fractions, color="#3b7dd8")
    ax_b.set_xlabel("Threshold (I >=)")
    ax_b.set_ylabel("County-times satisfying threshold (%)")
    ax_b.set_title("B. County-time availability by threshold")
    ax_b.grid(alpha=0.3, axis="y")
    for bar, frac in zip(bars, fractions):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{frac:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle(f"DRN availability - {state_name}")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def plot_drn_heatmap(t, drn, primary_mask, first_infection_time, counties, output_path, state_name=""):
    order = np.argsort(first_infection_time)
    drn_sorted = drn[:, order].T
    primary_sorted = primary_mask[:, order].T
    counties_sorted = [counties[i] for i in order]

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_drn = np.log2(drn_sorted)
    masked = np.ma.masked_where(~primary_sorted, log2_drn)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgray")
    norm = mcolors.CenteredNorm(vcenter=0.0)

    n_counties = len(counties_sorted)
    fig_height = max(6.0, 0.16 * n_counties)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[t.min(), t.max(), n_counties - 0.5, -0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(n_counties))
    ax.set_yticklabels(counties_sorted, fontsize=4)
    ax.set_xlabel("Day")
    ax.set_ylabel("County (sorted by first infection time)")
    ax.set_title(
        f"Nominal DRN heatmap - {state_name}\n"
        f"(gray = unavailable, I below primary threshold)"
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    tick_vals = [-2, -1, 0, 1, 2]
    if norm.vmin is not None and norm.vmax is not None:
        tick_vals = [v for v in tick_vals if norm.vmin <= v <= norm.vmax]
    if not tick_vals:
        tick_vals = [0]
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([("1 (threshold)" if v == 0 else f"{2.0**v:g}") for v in tick_vals])
    cbar.set_label("DRN (log2 scale): <1 decline, 1 = epidemic threshold, >1 growth")

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



STATE_COLORS = {
    "kansas": "#0072B2",
    "florida": "#D55E00",
    "california": "#009E73",
}
STATE_LABELS = {"kansas": "Kansas", "florida": "Florida", "california": "California"}


def set_manuscript_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(fig, ax, label: str) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 - 0.045, bbox.y1 + 0.015, label, fontsize=11, fontweight="bold", va="bottom", ha="left")


def _save_vector_and_raster(fig, output_dir, stem: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    width_in, height_in = fig.get_size_inches()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    png_size_px = (round(width_in * 600), round(height_in * 600))
    return pdf_path, png_path, png_size_px


def plot_fig3_epidemic_coverage(epidemic_by_state: dict, availability_df, states: list, output_dir, primary_threshold: int):
    set_manuscript_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    handles = []
    for state in states:
        d = epidemic_by_state[state]
        color = STATE_COLORS[state]
        (line,) = ax_a.plot(d["t"], d["median"], color=color, linewidth=1.3, label=STATE_LABELS[state])
        ax_a.fill_between(d["t"], d["q25"], d["q75"], color=color, alpha=0.18, linewidth=0)
        handles.append(line)
    ax_a.set_xlabel("Day")
    ax_a.set_ylabel("Infectious population (%)")
    ax_a.set_xlim([0,250])

    thresholds = sorted(availability_df["threshold"].unique())
    width = 0.8 / len(states)
    x_base = np.arange(len(thresholds))
    for i, state in enumerate(states):
        sub = availability_df[availability_df["state"] == state]
        color = STATE_COLORS[state]
        offset = (i - (len(states) - 1) / 2) * width
        for j, thr in enumerate(thresholds):
            vals = sub.loc[sub["threshold"] == thr, "fraction"].to_numpy() * 100.0
            med = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ax_b.errorbar(
                [x_base[j] + offset], [med], yerr=[[med - q25], [q75 - med]],
                fmt="o", color=color, markersize=3.5, capsize=2, linewidth=1.0, elinewidth=1.0,
            )
    ax_b.set_xticks(x_base)
    ax_b.set_xticklabels([f"≥ {t}" for t in thresholds])
    if primary_threshold in thresholds:
        primary_idx = thresholds.index(primary_threshold)
        ax_b.get_xticklabels()[primary_idx].set_fontweight("bold")
        ax_b.axvspan(x_base[primary_idx] - 0.45, x_base[primary_idx] + 0.45, color="#cccccc", alpha=0.25, zorder=0)
    ax_b.set_ylabel("County-time coverage (%)")
    ax_b.set_xlabel(r"$I_i$ threshold")
    ax_b.set_ylim(0, 100)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(
        handles, [STATE_LABELS[s] for s in states], loc="upper center", ncol=len(states),
        bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9,
    )
    _panel_label(fig, ax_a, "A")
    _panel_label(fig, ax_b, "B")
    return _save_vector_and_raster(fig, output_dir, "fig3_epidemic_coverage")


def _plot_indeterminacy_vs_misclassification(ax, indeterminate_df, misclass_df) -> None:
    color_indeterminate, color_misclass = "#999999", "#0072B2"
    for df_, color, label in (
        (indeterminate_df, color_indeterminate, "Indeterminate rate"),
        (misclass_df, color_misclass, "Point-estimate misclassification rate"),
    ):
        d = df_.sort_values("severity")
        yerr_lo = np.clip(d["point"] - d["lo"], 0.0, None) * 100.0
        yerr_hi = np.clip(d["hi"] - d["point"], 0.0, None) * 100.0
        ax.errorbar(
            d["severity"], d["point"] * 100.0, yerr=[yerr_lo, yerr_hi],
            fmt="o-", color=color, markersize=4, linewidth=1.4, capsize=3, elinewidth=1.0, label=label, zorder=3,
        )

    ax.set_xlabel(r"Coupling uncertainty bound $b$")
    ax.set_ylabel("Rate (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)


def _plot_classification_composition(ax, classification_df, xlabel: str, interval_kind: str = "DRN") -> None:
    df = classification_df.sort_values("severity").reset_index(drop=True)
    x = np.arange(len(df))
    growth = df["growth"].to_numpy() * 100.0
    decline = df["decline"].to_numpy() * 100.0
    indeterminate = df["indeterminate"].to_numpy() * 100.0

    color_growth, color_decline, color_indeterminate = "#0072B2", "#D55E00", "#999999"
    ax.bar(x, growth, color=color_growth, width=0.6, label="Robust growth", zorder=3)
    ax.bar(x, decline, bottom=growth, color=color_decline, width=0.6, label="Robust decline", zorder=3)
    ax.bar(x, indeterminate, bottom=growth + decline, color=color_indeterminate, width=0.6, label="Indeterminate", zorder=3)

    for xi, ind in zip(x, indeterminate):
        ax.annotate(f"{ind:.2f}%", (xi, 100.0), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in df["severity"]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Share of available {interval_kind} intervals (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])


def _plot_survival_curve(ax, survival: dict) -> None:
    ax.step(survival["grid"], survival["survival_pct"], where="post", color="#0072B2", linewidth=1.4, zorder=3)

    median = survival["median"]
    pct_06 = survival["pct_u_ge_0_6"]
    ax.axvline(0.6, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(median, color="#333333", linewidth=0.9, linestyle="--", zorder=1)

    ax.annotate(
        f"median $u^*$ = {median:.3f}", xy=(median, 50.0), xycoords="data",
        xytext=(-70, 18), textcoords="offset points", fontsize=7.5, ha="right",
        arrowprops=dict(arrowstyle="-", color="#333333", linewidth=0.8, shrinkA=0, shrinkB=3),
    )
    ax.annotate(
        f"{pct_06:.2f}% have $u^*$" + r"$\geq$" + "0.6", (0.6, pct_06),
        textcoords="offset points", xytext=(-8, -14), fontsize=7.5, ha="right", va="top",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel(r"Maximum underreporting fraction $u$")
    ax.set_ylabel(r"Observations with $u_i^\star \geq u$ (%)")


def _plot_near_threshold_boxplots(ax, group_df) -> None:
    groups = [1, 2, 3, 4, 5]
    data = [group_df.loc[group_df["group"] == g, "median_u_star"].to_numpy() for g in groups]
    means = [float(np.mean(d)) for d in data]

    box_color = "#0072B2"
    ax.boxplot(
        data, positions=groups, widths=0.5, patch_artist=True, showfliers=True,
        boxprops=dict(facecolor=box_color, alpha=0.25, edgecolor=box_color, linewidth=1.1),
        medianprops=dict(color=box_color, linewidth=1.6),
        whiskerprops=dict(color=box_color, linewidth=1.0),
        capprops=dict(color=box_color, linewidth=1.0),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor=box_color, markeredgecolor="none", alpha=0.35),
    )
    ax.plot(groups, means, marker="D", markersize=6, color="black", linestyle="none", zorder=5)

    ax.set_xticks(groups)
    ax.set_xticklabels(["Lowest 20%", "20–40%", "40–60%", "60–80%", "Highest 20%"])
    ax.set_xlabel(r"External infection share $\hat{q}_i$")
    ax.set_ylabel(r"Trajectory-level median $u_i^\star$")


def plot_fig4_underreporting_robustness(classification_df, survival: dict, group_df, output_dir):
    set_manuscript_style()
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_classification_composition(ax_a, classification_df, r"Maximum underreporting fraction $u$")
    handles, labels = ax_a.get_legend_handles_labels()

    _plot_survival_curve(ax_b, survival)
    _plot_near_threshold_boxplots(ax_c, group_df)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    bbox_a = ax_a.get_position()
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(bbox_a.x0 + 0.0, bbox_a.y1 + 0.0),
        ncol=3, frameon=False, fontsize=7, handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
    )
    for ax, label in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        _panel_label(fig, ax, label)
    return _save_vector_and_raster(fig, output_dir, "fig4_underreporting_robustness")


def plot_fig5_coupling_uncertainty(classification_df, indeterminate_df, misclass_df, output_dir):
    set_manuscript_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    _plot_classification_composition(ax_a, classification_df, r"Coupling uncertainty bound $b$", interval_kind="coupling")
    handles, labels = ax_a.get_legend_handles_labels()

    _plot_indeterminacy_vs_misclassification(ax_b, indeterminate_df, misclass_df)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    bbox_a = ax_a.get_position()
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(bbox_a.x0 + 0.02, bbox_a.y1 + 0.015),
        ncol=3, frameon=False, fontsize=7, handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
    )
    ax_a.set_title("A", loc="left", fontsize=11, fontweight="bold", pad=8)
    ax_b.set_title("B", loc="left", fontsize=11, fontweight="bold", pad=8)
    return _save_vector_and_raster(fig, output_dir, "fig5_coupling_uncertainty")
