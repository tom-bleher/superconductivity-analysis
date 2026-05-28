# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "taulab",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _md_intro(mo):
    mo.md(r"""
    # Superconductivity — Part A

    A $\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$ sample is measured by four-probe
    resistance as it is cooled through, and heated back across, its
    superconducting transition.

    The transition temperature is taken where the resistance falls most
    steeply — the peak of $\mathrm{d}R/\mathrm{d}T$:

    $$T_c = T_c^{\,\max\,\mathrm{d}R/\mathrm{d}T}.$$
    """)
    return


@app.cell
def _imports():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.signal import savgol_filter

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
    })

    from taulab.stats import resolution_sigma

    return Line2D, Path, mo, np, pd, plt, resolution_sigma, savgol_filter


@app.cell
def _paths(Path):
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data" / "part_a"
    MEAS_DIR = DATA / "measurements"
    OUT_DIR = ROOT / "results" / "part_a"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return MEAS_DIR, OUT_DIR


@app.cell
def _constants():
    SAVGOL_MAIN_SPAN_K = 3.75
    SAVGOL_POLYORDER = 3
    return SAVGOL_MAIN_SPAN_K, SAVGOL_POLYORDER


@app.cell
def _instrument(np, resolution_sigma):
    """Per-point instrument uncertainties.

    Rigol DM3058 5½-digit DMM — accuracy + resolution in quadrature, with
    σ_res = LSD/√12 (uniform-distribution width of the last displayed digit):

      σ_X = sqrt(σ_acc² + σ_res²)

    Range selection verified against the data:
      - |V| ≤ 2.7 mV across all runs → fixed 100 mV DCV range (LSD = 1 µV).
      - |I| spans 30–240 mA → auto-switched per point: 200 mA range
        (LSD = 1 µA) for |I| ≤ 0.2 A, else 2 A range (LSD = 10 µA).

    σ_R propagates through R = V/I (partial-derivative quadrature). The
    per-run σ_T helper is used for plot x-error bars and diagnostics; the
    extracted Tc uncertainties use the local derivative-peak spacing instead.
    """
    V_RANGE, V_LSD = 0.1, 1e-6
    V_RES = resolution_sigma(V_LSD)

    def sigma_V(V):
        V = np.asarray(V)
        return np.sqrt((0.00015 * np.abs(V) + 0.00004 * V_RANGE) ** 2 + V_RES**2)

    def sigma_I(I):
        I = np.asarray(I)
        I_RANGE = np.where(np.abs(I) <= 0.2, 0.2, 2.0)
        I_RES = np.where(
            np.abs(I) <= 0.2,
            resolution_sigma(1e-6),
            resolution_sigma(1e-5),
        )
        return np.sqrt((0.0025 * np.abs(I) + 0.00020 * I_RANGE) ** 2 + I_RES**2)

    def sigma_R(V, I):
        V, I = np.asarray(V), np.asarray(I)
        return np.sqrt((sigma_V(V) / I) ** 2 + (V * sigma_I(I) / I**2) ** 2)

    def sigma_T_local(T):
        """Per-point T uncertainty from local sample spacing: (½·span)/√12.

        Each point's T is one sample of a continuously drifting sweep; modeling
        the true value as uniform over the local inter-point gap gives
        σ_i = Δ_local/√12 with Δ_local = (T[i+1] − T[i−1])/2 (symmetric
        half-span; one-sided gap at the ends). Returns a per-point array so the
        x-error bars track the real density — tight where the sweep lingered,
        wide where it ran fast — and use the same spacing definition as the
        derivative-peak uncertainty in `derivative_peak`. No instrumental
        resolution term: only the converted-to-K column is logged, not the raw
        Pt-sensor reading, so the meter LSD can't be recovered, and even a
        generous mK-scale value would be dwarfed by this spacing.
        """
        T = np.asarray(T, dtype=float)
        if len(T) < 2:
            return np.zeros_like(T)
        span = np.empty_like(T)
        span[1:-1] = (T[2:] - T[:-2]) / 2.0
        span[0] = T[1] - T[0]
        span[-1] = T[-1] - T[-2]
        return np.abs(span) / np.sqrt(12)

    return sigma_R, sigma_T_local


@app.cell
def _load(MEAS_DIR, pd):
    """Load every frozen per-run CSV.

    The data files are deduplicated and clipped to the common 80-105 K
    transition window before they enter this notebook. The load sorts each
    frame by ascending temperature so the pipeline's monotonic-T invariant is
    self-enforced (np.interp and savgol_filter produce garbage otherwise).
    """
    T_MIN, T_MAX = 80.0, 105.0
    _meas = {}
    for _path in sorted(MEAS_DIR.glob("partA_*.csv")):
        _meas[_path.stem] = pd.read_csv(_path).sort_values("temperature_K").reset_index(drop=True)
    measurements = _meas
    return T_MAX, T_MIN, measurements


@app.cell
def _md_tc_methods(SAVGOL_MAIN_SPAN_K, SAVGOL_POLYORDER, mo):
    mo.md(rf"""
    ## $T_c$ extraction

    $R(T)$ is resampled onto a uniform $T$ grid, smoothed with a
    Savitzky–Golay filter, and differentiated analytically. The window is a
    ${SAVGOL_MAIN_SPAN_K:g}\,\mathrm{{K}}$ cubic span
    (`polyorder = {SAVGOL_POLYORDER}`) — the narrowest setting that suppresses
    spurious derivative peaks without shifting $T_c$. The transition is the
    location of the steepest rise,

    $$T_c=\arg\max_T \frac{{\mathrm{{d}}R}}{{\mathrm{{d}}T}}.$$

    The reported uncertainty is the local temperature sampling resolution at
    that peak (index $i$):

    $$\sigma_{{T_c}} = \frac{{(T_{{i+1}} - T_{{i-1}})}}{{2\sqrt{{12}}}}.$$
    """)
    return


@app.cell
def _savgol_helpers(np, savgol_filter):
    """Shared Savitzky-Golay smoother on a uniform temperature grid.

    The window is specified in kelvin, not sample count, so runs with different
    point density get the same physical smoothing scale.
    """

    def _window_length(n, dT, span_K, polyorder):
        w = max(polyorder + 2, int(round(span_K / dT)))
        if w % 2 == 0:
            w += 1
        max_w = n if n % 2 else n - 1
        return min(w, max_w)

    def savgol_trace(T, R, span_K, polyorder):
        T, R = np.asarray(T, dtype=float), np.asarray(R, dtype=float)
        if len(T) < polyorder + 2:
            return R, np.full_like(R, np.nan)

        T_uni = np.linspace(T.min(), T.max(), len(T))
        R_uni = np.interp(T_uni, T, R)
        dT = (T_uni[-1] - T_uni[0]) / (len(T_uni) - 1)
        w = _window_length(len(T_uni), dT, span_K, polyorder)
        if w <= polyorder:
            return R, np.full_like(R, np.nan)

        R_s_uni = savgol_filter(R_uni, w, polyorder=polyorder)
        dR_uni = savgol_filter(R_uni, w, polyorder=polyorder, deriv=1, delta=dT)
        return np.interp(T, T_uni, R_s_uni), np.interp(T, T_uni, dR_uni)

    return (savgol_trace,)


@app.cell
def _analysis(
    SAVGOL_MAIN_SPAN_K,
    SAVGOL_POLYORDER,
    measurements,
    np,
    pd,
    savgol_trace,
):
    """Analyze each run once and keep that record as the single source of truth."""

    def _local_sample_sigma(T, i):
        left = abs(T[i] - T[i - 1]) if i > 0 else np.nan
        right = abs(T[i + 1] - T[i]) if i < len(T) - 1 else np.nan
        return float(np.nanmean([left, right])) / np.sqrt(12)

    def _derivative_peak(T, R):
        if len(T) < 11:
            return float("nan"), float("nan")

        _, dR = savgol_trace(
            T,
            R,
            span_K=SAVGOL_MAIN_SPAN_K,
            polyorder=SAVGOL_POLYORDER,
        )
        if np.isnan(dR).all():
            return float("nan"), float("nan")

        i_peak = int(np.nanargmax(dR))
        T_peak = float(T[i_peak])
        sigma = _local_sample_sigma(T, i_peak)
        return T_peak, sigma

    def analyze_run(measurement_id, df):
        meta = df.iloc[0]
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        Tc_derivative, sigma_derivative = _derivative_peak(T, R)

        return dict(
            measurement_id=measurement_id,
            sample_current_mA_nominal=float(meta["sample_current_mA_nominal"]),
            series_resistor=meta["series_resistor"],
            direction=meta["direction"],
            field_condition=meta["field_condition"],
            tc_derivative_K=Tc_derivative,
            tc_derivative_err_K=sigma_derivative,
        )

    runs = {
        measurement_id: analyze_run(measurement_id, df)
        for measurement_id, df in measurements.items()
    }
    tc_summary = (
        pd.DataFrame(runs.values())
        .sort_values("measurement_id")
        .reset_index(drop=True)
    )
    return runs, tc_summary


@app.cell
def _trace_helpers(
    SAVGOL_MAIN_SPAN_K,
    SAVGOL_POLYORDER,
    savgol_trace,
    sigma_R,
    sigma_T_local,
):
    """Display-only smoothed R(T) + dR/dT trace for the plotted (possibly
    clipped) T-range. Tc values are *not* computed here — the plots draw the
    compute-once values from `runs`, so the lines match the summary table.

    Returns per-point σ_R/σ_T error arrays, the SG-smoothed $R$, and the
    analytic $\\mathrm{d}R/\\mathrm{d}T$.
    """

    def trace(df):
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        R_err = sigma_R(df["voltage_V"].to_numpy(), df["current_A"].to_numpy())
        T_err = sigma_T_local(T)
        R_s, dR_dT = savgol_trace(
            T,
            R,
            span_K=SAVGOL_MAIN_SPAN_K,
            polyorder=SAVGOL_POLYORDER,
        )

        return dict(T=T, R=R, R_err=R_err, T_err=T_err, R_smoothed=R_s, dR_dT=dR_dT)

    return (trace,)


@app.cell
def _pair_keys(tc_summary):
    """Decide which runs share a heat/cool partner and which share a magnet partner.

    `heat_cool_groups`: {(current_mA, resistor): [measurement_id, ...]} for
        no-magnet groups that contain BOTH a heat and a cool run.
    `magnet_groups`: {(current_mA, resistor, direction): [...]} for groups
        that contain BOTH a magnet and a no-magnet run.
    `solo_ids`: everything else, plotted on its own.
    """
    heat_cool_groups, magnet_groups = {}, {}
    _paired = set()
    _no_mag = tc_summary[tc_summary["field_condition"] == "no_magnet"]
    for (_I, _R), _g in _no_mag.groupby(["sample_current_mA_nominal", "series_resistor"]):
        if set(_g["direction"]) >= {"heat", "cool"}:
            heat_cool_groups[(float(_I), _R)] = _g["measurement_id"].tolist()
            _paired.update(_g["measurement_id"])
    for (_I, _R, _d), _g in tc_summary.groupby(
        ["sample_current_mA_nominal", "series_resistor", "direction"]
    ):
        if set(_g["field_condition"]) >= {"magnet", "no_magnet"}:
            magnet_groups[(float(_I), _R, _d)] = _g["measurement_id"].tolist()
            _paired.update(_g["measurement_id"])
    solo_ids = [m for m in tc_summary["measurement_id"] if m not in _paired]
    return heat_cool_groups, magnet_groups, solo_ids


@app.cell
def _fmt():
    """Math-mode formatters for currents, series resistors, and direction labels."""
    _RESISTOR_TEX = {
        "100ohm": r"100\,\Omega",
        "1kohm":  r"1\,\mathrm{k}\Omega",
    }
    _DIRECTION_TEX = {"cool": "cooling", "heat": "heating"}

    def fmt_I(mA):
        return rf"I = {int(mA)}\,\mathrm{{mA}}"

    def fmt_R(s):
        return rf"R_{{\mathrm{{s}}}} = {_RESISTOR_TEX.get(s, s)}"

    def fmt_dir(d):
        return _DIRECTION_TEX.get(d, d)

    def fmt_field(c):
        return r"B = 0" if c == "no_magnet" else r"B \neq 0"

    return fmt_I, fmt_R, fmt_dir, fmt_field


@app.cell
def _two_panel(Line2D, T_MAX, T_MIN, plt):
    """Shared two-panel layout: R(T) on top, dR/dT below, common x-axis.

    The bottom-axis legend names the single reported Tc criterion so it never
    collides with the per-trace legend on the top axis.
    """

    _CRIT_HANDLES = [
        Line2D([0], [0], color="#444444", ls=(0, (5, 3)), lw=1.0,
               label=r"$T_c^{\,\max\,\mathrm{d}R/\mathrm{d}T}$"),
    ]

    def build(_title=None, _subtitle=None):
        fig, (ax_r, ax_d) = plt.subplots(
            2, 1, figsize=(8.0, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        )
        if _title:
            ax_r.set_title(_title, pad=24 if _subtitle else 12, loc="left")
        if _subtitle:
            ax_r.text(
                0.0, 1.02, _subtitle, transform=ax_r.transAxes,
                ha="left", va="bottom", fontsize=9.5, color="#888888",
            )
        ax_r.set_ylabel(r"$R$  (m$\Omega$)")
        ax_d.set_xlabel(r"Temperature  $T$  (K)")
        ax_d.set_ylabel(r"$\mathrm{d}R/\mathrm{d}T$  (m$\Omega$/K)")
        for ax in (ax_r, ax_d):
            ax.grid(True, axis="y", alpha=0.12, lw=0.5)
            ax.tick_params(direction="in", length=3.5, top=False, right=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_d.set_xlim(T_MIN - 1, T_MAX + 1)
        ax_d.legend(
            handles=_CRIT_HANDLES, loc="upper right",
            frameon=False, fontsize=8.5,
        )
        return fig, ax_r, ax_d

    def legend_with_info(ax, info):
        """Upper-left legend with the run-condition string as a label-only row."""
        _handles, _labels = ax.get_legend_handles_labels()
        _handles.append(Line2D([], [], color="none"))
        _labels.append(info)
        ax.legend(_handles, _labels, loc="upper left", frameon=False)

    def draw(ax_r, ax_d, tr, tc_drdt, color, marker, label):
        # PRL-style: open markers in series color, capless error bars on
        # each raw point, smoothed line in the same saturated color.
        # Tc is the compute-once value from `runs` (not recomputed on the
        # clipped display data), so lines match the table.
        ax_r.errorbar(
            tr["T"], tr["R"] * 1e3,
            xerr=tr["T_err"], yerr=tr["R_err"] * 1e3,
            fmt=marker, ms=3.8, mfc="none", mec=color, mew=0.8,
            ecolor=color, elinewidth=0.5, capsize=0,
            alpha=0.7, zorder=2,
        )
        ax_r.plot(
            tr["T"], tr["R_smoothed"] * 1e3, lw=1.8, color=color, zorder=3,
            label=(
                rf"{label}  —  "
                rf"$T_c = {tc_drdt:.2f}\,$K"
            ),
        )
        # Tc reference lines: thin, low alpha — guides, not data.
        ax_r.axvline(tc_drdt, color=color, ls=(0, (5, 3)), lw=0.9, alpha=0.5, zorder=1)
        ax_d.plot(tr["T"], tr["dR_dT"] * 1e3, color=color, lw=1.6, alpha=0.8)
        _idx_pk = int(abs(tr["T"] - tc_drdt).argmin())
        ax_d.plot(
            [tr["T"][_idx_pk]], [tr["dR_dT"][_idx_pk] * 1e3],
            marker="v", ms=6, color=color, mec="white", mew=0.8, zorder=4,
        )
        ax_d.axvline(tc_drdt, color=color, ls=(0, (5, 3)), lw=0.9, alpha=0.5, zorder=1)

    return build, draw, legend_with_info


@app.cell
def _pair_clip():
    """Clip every member of a pair to the common temperature overlap.

    The 30 mA cool run stops at 99.7 K while the 30 mA heat run runs to
    104.8 K; the with-magnet 100 mA run stops at 100.1 K while no-magnet
    runs to 104.8 K. Comparing them on different x-ranges is misleading
    *and* causes savgol's edge-of-trim artifact to dominate the dR/dT of
    the longer run (the bogus 101.88 K spike). Clipping to the intersection
    fixes both. Analysis cells (A1–A6) are untouched — only the pair plots.
    """

    def intersection(dfs):
        T_lo = max(float(d["temperature_K"].min()) for d in dfs)
        T_hi = min(float(d["temperature_K"].max()) for d in dfs)
        return T_lo, T_hi

    def clip(df, T_lo, T_hi):
        mask = (df["temperature_K"] >= T_lo) & (df["temperature_K"] <= T_hi)
        return df[mask].reset_index(drop=True)

    return clip, intersection


@app.cell
def _make_figure(build, clip, draw, legend_with_info, runs, trace):
    """One two-panel figure for any group of runs — the single body shared by
    the heat/cool, magnet, and solo plot cells.

    `members`   : list of (measurement_id, df).
    `styles`    : {style_key: (color, marker, label)}.
    `key_of`    : meta-row → style_key (e.g. direction or field_condition).
    `clip_range`: (T_lo, T_hi) to clip display data to the pair overlap, or
                  None for full range (solo). Tc lines come from `runs`, so
                  they're identical whether or not the display is clipped.
    """

    def make_figure(members, styles, key_of, info, title, subtitle=None, clip_range=None):
        fig, ax_r, ax_d = build(title, subtitle)
        for _mid, _df in members:
            _d = clip(_df, *clip_range) if clip_range else _df
            _color, _marker, _label = styles[key_of(_d.iloc[0])]
            _rec = runs[_mid]
            draw(
                ax_r,
                ax_d,
                trace(_d),
                _rec["tc_derivative_K"],
                _color,
                _marker,
                _label,
            )
        if clip_range:
            for _ax in (ax_r, ax_d):
                _ax.set_xlim(clip_range[0] - 0.5, clip_range[1] + 0.5)
        legend_with_info(ax_r, info)
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.89)
        return fig

    return (make_figure,)


@app.cell
def _dir_styles():
    """Shared style maps. Heat/cool keyed by direction (red/blue); magnet
    keyed by field_condition (green/purple) so a field pair isn't mistaken
    for a temperature-direction comparison."""
    DIR_STYLES = {
        "cool": ("#1f77b4", "s", "cooling"),
        "heat": ("#d62728", "o", "heating"),
    }
    FIELD_STYLES = {
        "no_magnet": ("#2ca02c", "s", r"$B = 0$"),
        "magnet":    ("#9467bd", "o", r"$B \neq 0$"),
    }
    return DIR_STYLES, FIELD_STYLES


@app.cell
def _md_plot_guide(mo, tc_summary):
    def _one(current_mA, direction, field):
        _row = tc_summary[
            (tc_summary["sample_current_mA_nominal"] == current_mA)
            & (tc_summary["direction"] == direction)
            & (tc_summary["field_condition"] == field)
        ]
        return _row.iloc[0]

    _heat_30 = _one(30.0, "heat", "no_magnet")
    _cool_30 = _one(30.0, "cool", "no_magnet")
    _heat_100 = _one(100.0, "heat", "no_magnet")
    _field_100 = _one(100.0, "heat", "magnet")
    _cool_240 = _one(240.0, "cool", "no_magnet")

    _lag_30 = _heat_30["tc_derivative_K"] - _cool_30["tc_derivative_K"]
    _field_shift = _heat_100["tc_derivative_K"] - _field_100["tc_derivative_K"]

    mo.md(rf"""
    ## Figure guide

    Each figure pairs the smoothed $R(T)$ (top) with its derivative (bottom);
    the dashed line marks $T_c$. Three comparisons stand out:

    - **Thermal hysteresis** — at $30\,\mathrm{{mA}}$, heating sits
      ${_lag_30:.2f}\,\mathrm{{K}}$ above cooling.
    - **Applied field** — at $100\,\mathrm{{mA}}$, the field lowers $T_c$ by
      ${_field_shift:.2f}\,\mathrm{{K}}$.
    - **Drive current** — the $240\,\mathrm{{mA}}$ cooling run gives the lowest
      transition, ${_cool_240["tc_derivative_K"]:.2f}\,\mathrm{{K}}$.
    """)
    return


@app.cell
def _heat_cool_plots(
    DIR_STYLES,
    fmt_I,
    fmt_R,
    heat_cool_groups,
    intersection,
    make_figure,
    measurements,
    mo,
):
    """One figure per heat/cool pair (no-magnet runs only)."""
    _figs = []
    for (_I, _R), _ids in sorted(heat_cool_groups.items()):
        _members = [(m, measurements[m]) for m in _ids]
        _range = intersection([measurements[m] for m in _ids])
        _info = rf"${fmt_I(_I)}$,  ${fmt_R(_R)}$,  $B = 0$"
        _figs.append(make_figure(
            _members, DIR_STYLES, lambda meta: meta["direction"], _info,
            "Heating versus cooling",
            r"$\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$",
            _range,
        ))
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _magnet_plots(
    FIELD_STYLES,
    fmt_I,
    fmt_R,
    fmt_dir,
    intersection,
    magnet_groups,
    make_figure,
    measurements,
    mo,
):
    """One figure per magnet-vs-no-magnet pair."""
    _figs = []
    for (_I, _R, _d), _ids in sorted(magnet_groups.items()):
        _members = [(m, measurements[m]) for m in _ids]
        _range = intersection([measurements[m] for m in _ids])
        _info = rf"${fmt_I(_I)}$,  ${fmt_R(_R)}$,  {fmt_dir(_d)}"
        _figs.append(make_figure(
            _members, FIELD_STYLES, lambda meta: meta["field_condition"], _info,
            "Zero field versus applied field",
            r"$\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$",
            _range,
        ))
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _solo_plots(
    DIR_STYLES,
    fmt_I,
    fmt_R,
    fmt_dir,
    fmt_field,
    make_figure,
    measurements,
    mo,
    solo_ids,
):
    """Same two-panel style for runs without a pair — one figure each."""
    _figs = []
    for _mid in sorted(solo_ids):
        _meta = measurements[_mid].iloc[0]
        _info = (
            rf"${fmt_I(_meta['sample_current_mA_nominal'])}$,  "
            rf"${fmt_R(_meta['series_resistor'])}$,  "
            rf"{fmt_dir(_meta['direction'])},  ${fmt_field(_meta['field_condition'])}$"
        )
        _figs.append(make_figure(
            [(_mid, measurements[_mid])], DIR_STYLES,
            lambda meta: meta["direction"], _info,
            "Single transition run",
            r"$\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$",
            None,
        ))
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _md_final(mo):
    mo.md(r"""
    ## Summary

    Derivative-peak $T_c$ for every run, with its local-sampling uncertainty.
    """)
    return


@app.cell
def _final_table(pd, tc_summary):
    """Display table: metadata + derivative-peak Tc.

    Tc columns are formatted as "val ± σ (rel%)" with rel = σ/|val|·100.
    The CSV written below stays numeric for downstream processing.
    """
    def _fmt(val, err):
        rel = (err / abs(val) * 100.0) if val else float("nan")
        return f"{val:.2f} ± {err:.2f} ({rel:.2f}%)"

    _rows = []
    for _, _r in tc_summary.iterrows():
        _rows.append({
            "I (mA)":             int(_r["sample_current_mA_nominal"]),
            "R_s":                _r["series_resistor"],
            "sweep":              _r["direction"],
            "field":              _r["field_condition"],
            "Tc [K]":             _fmt(_r["tc_derivative_K"], _r["tc_derivative_err_K"]),
        })
    final_table = pd.DataFrame(_rows)
    return (final_table,)


@app.cell
def _show_final(final_table):
    final_table
    return


@app.cell
def _write(OUT_DIR, tc_summary):
    tc_path = OUT_DIR / "tc_summary.csv"
    tc_summary.to_csv(tc_path, index=False)
    print(f"wrote {tc_path}")
    return


if __name__ == "__main__":
    app.run()
