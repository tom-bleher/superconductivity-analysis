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

    Four-probe resistance of a $\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$ sample is
    measured across the superconducting transition.

    One transition temperature is reported for each run: the ODR-refined
    maximum-slope point of $R(T)$.
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

    from taulab.fits import odr_fit
    from taulab.stats import resolution_sigma

    return (
        Line2D,
        Path,
        mo,
        np,
        odr_fit,
        pd,
        plt,
        resolution_sigma,
        savgol_filter,
    )


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
    local σ_T helper describes the finite sampling bracket in temperature.
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
        """Per-point T uncertainty from the local temperature bracket.

        Each point's T is one sample of a continuously drifting sweep; modeling
        the true value as uniform over the neighboring-sample bracket gives
        σ_i = (T[i+1] − T[i−1]) / √12 for interior points. No instrumental
        resolution term is included because only the converted-to-K column is
        logged, not the raw Pt-sensor reading.
        """
        T = np.asarray(T, dtype=float)
        if len(T) < 2:
            return np.zeros_like(T)
        span = np.empty_like(T)
        span[1:-1] = T[2:] - T[:-2]
        span[0] = T[1] - T[0]
        span[-1] = T[-1] - T[-2]
        return np.abs(span) / np.sqrt(12)

    return sigma_I, sigma_R, sigma_T_local, sigma_V


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
    ## Method

    Each run is sorted by temperature, resampled to a uniform grid, smoothed
    with a ${SAVGOL_MAIN_SPAN_K:g}\,\mathrm{{K}}$ cubic Savitzky-Golay filter
    (`polyorder = {SAVGOL_POLYORDER}`), and differentiated. The peak is refined
    by a local ODR quadratic fit,

    $$\frac{{\mathrm{{d}}R}}{{\mathrm{{d}}T}} = a + b(T-T_0) + c(T-T_0)^2,\qquad T_c = T_0 - \frac{{b}}{{2c}}.$$

    The reported uncertainty combines the ODR vertex uncertainty with the
    neighboring-sample bracket:

    $$\sigma_{{T_c}} = \sqrt{{\sigma_{{\mathrm{{fit}}}}^2 + \sigma_{{\mathrm{{samp}}}}^2}}, \qquad \sigma_{{\mathrm{{samp}}}} = \frac{{T_{{i+1}} - T_{{i-1}}}}{{\sqrt{{12}}}}.$$

    The parabolic fit is used only locally around the derivative maximum. Its
    vertex covariance gives a better peak-localization uncertainty than the
    sampled maximum alone, and is treated as a proxy for the uncertainty
    introduced by smoothing.
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
    odr_fit,
    pd,
    savgol_trace,
    sigma_I,
    sigma_R,
    sigma_T_local,
    sigma_V,
):
    """Analyze each run once and keep that record as the single source of truth."""

    def _quadratic(beta, x):
        return beta[0] + beta[1] * x + beta[2] * x**2

    def _peak_window(dR, i_peak):
        threshold = 0.8 * float(dR[i_peak])
        lo = i_peak
        hi = i_peak + 1
        while lo > 0 and np.isfinite(dR[lo - 1]) and dR[lo - 1] >= threshold:
            lo -= 1
        while hi < len(dR) and np.isfinite(dR[hi]) and dR[hi] >= threshold:
            hi += 1
        if hi - lo < 5:
            lo = max(0, i_peak - 5)
            hi = min(len(dR), i_peak + 6)
        return slice(lo, hi)

    def _fit_vertex(T, dR, i_peak, sigma_T):
        fit_slice = _peak_window(dR, i_peak)
        T_fit = T[fit_slice]
        y_fit = dR[fit_slice]
        x_err_fit = sigma_T[fit_slice]
        ok = np.isfinite(T_fit) & np.isfinite(y_fit)
        T_fit = T_fit[ok]
        y_fit = y_fit[ok]
        x_err_fit = x_err_fit[ok]
        if len(T_fit) < 4:
            return float(T[i_peak]), 0.0, len(T_fit), float("nan"), None

        T0 = float(T[i_peak])
        x = T_fit - T0
        coeff = np.polyfit(x, y_fit, 2)
        residual = y_fit - np.polyval(coeff, x)
        dof = max(len(T_fit) - 3, 1)
        sigma_y = float(np.sqrt(np.sum(residual**2) / dof))
        if not np.isfinite(sigma_y) or sigma_y <= 0:
            sigma_y = max(float(np.ptp(y_fit)), float(np.nanmax(np.abs(y_fit))), 1.0) * 1e-12

        x_err = np.asarray(x_err_fit, dtype=float)
        y_err = np.full_like(y_fit, sigma_y, dtype=float)
        try:
            result = odr_fit(
                _quadratic,
                [coeff[2], coeff[1], coeff[0]],
                x,
                x_err,
                y_fit,
                y_err,
                param_names=["a", "b", "c"],
            )
            _, b, c = result.params
            if not np.isfinite(c) or c >= 0:
                return float(T[i_peak]), 0.0, len(T_fit), float("nan"), None
            dx = float(-b / (2 * c))
            T_vertex = T0 + dx
            if T_vertex < float(T_fit.min()) or T_vertex > float(T_fit.max()):
                return float(T[i_peak]), 0.0, len(T_fit), float("nan"), None
            cov = result.cov if result.cov is not None else np.zeros((3, 3))
            jac = np.array([0.0, -1.0 / (2 * c), b / (2 * c**2)])
            sigma_fit = float(np.sqrt(max(jac @ cov @ jac, 0.0)))
            fit_T = np.linspace(float(T_fit.min()), float(T_fit.max()), 120)
            fit_x = fit_T - T0
            fit_dR = result.params[0] + b * fit_x + c * fit_x**2
            fit_overlay = dict(
                T=fit_T,
                dR_dT=fit_dR,
                T_vertex=T_vertex,
                dR_dT_vertex=float(result.params[0] + b * dx + c * dx**2),
            )
            return T_vertex, sigma_fit, len(T_fit), float(result.redchi), fit_overlay
        except Exception:
            return float(T[i_peak]), 0.0, len(T_fit), float("nan"), None

    def _derivative_peak(T, R):
        if len(T) < 11:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0, float("nan"), None

        _, dR = savgol_trace(
            T,
            R,
            span_K=SAVGOL_MAIN_SPAN_K,
            polyorder=SAVGOL_POLYORDER,
        )
        if np.isnan(dR).all():
            return float("nan"), float("nan"), float("nan"), float("nan"), 0, float("nan"), None

        i_peak = int(np.nanargmax(dR))
        sigma_T = sigma_T_local(T)
        sigma_sample = float(sigma_T[i_peak])
        T_peak, sigma_fit, n_fit, redchi, fit_overlay = _fit_vertex(T, dR, i_peak, sigma_T)
        sigma_total = float(np.sqrt(sigma_sample**2 + sigma_fit**2))
        return T_peak, sigma_total, sigma_sample, sigma_fit, n_fit, redchi, fit_overlay

    def analyze_run(measurement_id, df):
        meta = df.iloc[0]
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        V = df["voltage_V"].to_numpy()
        I = df["current_A"].to_numpy()
        (
            Tc_derivative,
            sigma_derivative,
            sigma_sample,
            sigma_fit,
            fit_points,
            fit_redchi,
            fit_overlay,
        ) = _derivative_peak(T, R)
        sigma_V_values = sigma_V(V)
        sigma_I_values = sigma_I(I)
        sigma_R_values = sigma_R(V, I)
        sigma_T_values = sigma_T_local(T)

        return dict(
            measurement_id=measurement_id,
            sample_current_mA_nominal=float(meta["sample_current_mA_nominal"]),
            series_resistor=meta["series_resistor"],
            direction=meta["direction"],
            field_condition=meta["field_condition"],
            tc_derivative_K=Tc_derivative,
            tc_derivative_err_K=sigma_derivative,
            tc_sampling_err_K=sigma_sample,
            tc_fit_err_K=sigma_fit,
            tc_fit_points=fit_points,
            tc_fit_redchi=fit_redchi,
            sigma_V_median_V=float(np.nanmedian(sigma_V_values)),
            sigma_I_median_A=float(np.nanmedian(sigma_I_values)),
            sigma_R_median_ohm=float(np.nanmedian(sigma_R_values)),
            sigma_T_median_K=float(np.nanmedian(sigma_T_values)),
        ), fit_overlay

    runs, fit_overlays = {}, {}
    for measurement_id, df in measurements.items():
        runs[measurement_id], fit_overlays[measurement_id] = analyze_run(measurement_id, df)
    tc_summary = (
        pd.DataFrame(runs.values())
        .sort_values("measurement_id")
        .reset_index(drop=True)
    )
    return fit_overlays, runs, tc_summary


@app.cell
def _md_error_model(mo):
    mo.md(r"""
    ## Measurement Errors

    Voltage and current uncertainties use the meter accuracy plus last-digit
    resolution in quadrature. Resistance is propagated directly from
    $R=V/I$:

    $$\sigma_R = \sqrt{\left(\frac{\sigma_V}{I}\right)^2 + \left(\frac{V\sigma_I}{I^2}\right)^2}.$$

    The table reports typical per-run values; $\sigma_T$ is the local
    temperature sampling bracket.
    """)
    return


@app.cell
def _error_table(pd, tc_summary):
    _rows = []
    for _, _r in tc_summary.iterrows():
        _rows.append({
            "I (mA)": int(_r["sample_current_mA_nominal"]),
            "sweep": _r["direction"],
            "field": _r["field_condition"],
            "typ. σV (µV)": f"{_r['sigma_V_median_V'] * 1e6:.2f}",
            "typ. σI (µA)": f"{_r['sigma_I_median_A'] * 1e6:.1f}",
            "typ. σR (mΩ)": f"{_r['sigma_R_median_ohm'] * 1e3:.3f}",
            "typ. σT (K)": f"{_r['sigma_T_median_K']:.3f}",
        })
    error_table = pd.DataFrame(_rows)
    return (error_table,)


@app.cell
def _show_error_table(error_table):
    error_table
    return


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
def _two_panel(Line2D, T_MAX, T_MIN, np, plt):
    """Shared two-panel layout: R(T) on top, dR/dT below, common x-axis.

    The bottom-axis legend names the single reported Tc criterion so it never
    collides with the per-trace legend on the top axis.
    """

    _CRIT_HANDLES = [
        Line2D([0], [0], color="#444444", ls=(0, (5, 3)), lw=1.0,
               label=r"$T_c^{\,\max\,\mathrm{d}R/\mathrm{d}T}$"),
        Line2D([0], [0], color="#444444", ls="-", lw=1.8, marker="v",
               markerfacecolor="#444444", markeredgecolor="white", markersize=6,
               label="local ODR parabola"),
    ]

    def build(_title=None):
        fig, (ax_r, ax_d) = plt.subplots(
            2, 1, figsize=(8.0, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        )
        if _title:
            fig.suptitle(_title, x=0.54, y=0.965, fontsize=14, fontweight="medium")
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

    def draw(ax_r, ax_d, tr, tc_drdt, fit_overlay, color, marker, label):
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
        _y_tc = float(np.interp(tc_drdt, tr["T"], tr["dR_dT"]))
        if fit_overlay is not None:
            ax_d.plot(
                fit_overlay["T"],
                fit_overlay["dR_dT"] * 1e3,
                color=color,
                lw=1.8,
                alpha=0.95,
                zorder=3,
            )
            _y_tc = fit_overlay["dR_dT_vertex"]
        ax_d.plot(
            [tc_drdt], [_y_tc * 1e3],
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
def _make_figure(
    build,
    clip,
    draw,
    fit_overlays,
    legend_with_info,
    runs,
    trace,
):
    """One two-panel figure for any group of runs — the single body shared by
    the heat/cool, magnet, and solo plot cells.

    `members`   : list of (measurement_id, df).
    `styles`    : {style_key: (color, marker, label)}.
    `key_of`    : meta-row → style_key (e.g. direction or field_condition).
    `clip_range`: (T_lo, T_hi) to clip display data to the pair overlap, or
                  None for full range (solo). Tc lines come from `runs`, so
                  they're identical whether or not the display is clipped.
    """

    def make_figure(members, styles, key_of, info, title, clip_range=None):
        fig, ax_r, ax_d = build(title)
        for _mid, _df in members:
            _d = clip(_df, *clip_range) if clip_range else _df
            _color, _marker, _label = styles[key_of(_d.iloc[0])]
            _rec = runs[_mid]
            draw(
                ax_r,
                ax_d,
                trace(_d),
                _rec["tc_derivative_K"],
                fit_overlays.get(_mid),
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
    ## Main Comparisons

    Dashed vertical lines mark derivative-peak $T_c$; the solid segment on
    each derivative peak is the local ODR parabola.

    - **Thermal hysteresis:** at $30\,\mathrm{{mA}}$, heating is
      ${_lag_30:.2f}\,\mathrm{{K}}$ above cooling.
    - **Applied field:** at $100\,\mathrm{{mA}}$, the field lowers $T_c$ by
      ${_field_shift:.2f}\,\mathrm{{K}}$.
    - **High current:** the $240\,\mathrm{{mA}}$ cooling run gives the lowest
      $T_c$, ${_cool_240["tc_derivative_K"]:.2f}\,\mathrm{{K}}$.
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
            "Heating and cooling comparison",
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
            "Applied-field comparison",
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
            None,
        ))
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _md_final(mo):
    mo.md(r"""
    ## Results

    Reported values are ODR-refined derivative-peak $T_c$ with fit and
    sampling uncertainties combined in quadrature.
    The numeric CSV is written to `results/part_a/tc_summary.csv`.
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
            "σ_samp [K]":         f"{_r['tc_sampling_err_K']:.2f}",
            "σ_fit [K]":          f"{_r['tc_fit_err_K']:.2f}",
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
