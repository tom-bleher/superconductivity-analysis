# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
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

    Four-probe resistance $R(T)$ of a $\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$ sample
    is measured across the superconducting transition.

    Each run reports a single transition temperature, the point of steepest
    ascent of a cubic smoothing spline through $R(T)$,
    $$T_c = \arg\max_T R'(T).$$
    This avoids resistance-threshold choices and fixes the analysis to one
    estimator, one smoothing rule, and one uncertainty budget.
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
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import minimize_scalar

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
    return Line2D, Path, UnivariateSpline, minimize_scalar, mo, np, pd, plt


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
    SPLINE_DEGREE = 3
    SPLINE_TARGET_MOHM = 0.04
    SPLINE_SENSITIVITY_TARGETS_MOHM = (0.04, 0.05, 0.06)
    SPLINE_GRID_POINTS = 2000
    SPLINE_BRACKET_STEPS = 12
    return (
        SPLINE_BRACKET_STEPS,
        SPLINE_DEGREE,
        SPLINE_GRID_POINTS,
        SPLINE_SENSITIVITY_TARGETS_MOHM,
        SPLINE_TARGET_MOHM,
    )


@app.cell
def _instrument(np):
    """Per-point instrument and sampling uncertainties.

    Rigol DM3058 5.5-digit DMM uncertainties are accuracy and last-digit
    resolution in quadrature. Resistance uncertainty is propagated directly
    from R = V/I. Temperature uncertainty is the local sampling-bin width,
    modeled as a uniform distribution over the Voronoi cell around each logged
    temperature point.
    """
    V_RANGE, V_LSD = 0.1, 1e-6
    V_RES = V_LSD / np.sqrt(12.0)

    def sigma_V(V):
        V = np.asarray(V, dtype=float)
        return np.sqrt((0.00015 * np.abs(V) + 0.00004 * V_RANGE) ** 2 + V_RES**2)

    def sigma_I(I):
        I = np.asarray(I, dtype=float)
        I_RANGE = np.where(np.abs(I) <= 0.2, 0.2, 2.0)
        I_RES = np.where(
            np.abs(I) <= 0.2,
            1e-6 / np.sqrt(12.0),
            1e-5 / np.sqrt(12.0),
        )
        return np.sqrt((0.0025 * np.abs(I) + 0.00020 * I_RANGE) ** 2 + I_RES**2)

    def sigma_R(V, I):
        V, I = np.asarray(V, dtype=float), np.asarray(I, dtype=float)
        return np.sqrt((sigma_V(V) / I) ** 2 + (V * sigma_I(I) / I**2) ** 2)

    def sigma_T_local(T):
        T = np.asarray(T, dtype=float)
        if len(T) < 2:
            return np.zeros_like(T, dtype=float)

        order = np.argsort(T)
        T_sorted = T[order]
        width = np.empty_like(T_sorted, dtype=float)
        width[0] = T_sorted[1] - T_sorted[0]
        width[-1] = T_sorted[-1] - T_sorted[-2]
        if len(T_sorted) > 2:
            width[1:-1] = 0.5 * (T_sorted[2:] - T_sorted[:-2])

        sigma_sorted = np.abs(width) / np.sqrt(12.0)
        sigma = np.empty_like(sigma_sorted)
        sigma[order] = sigma_sorted
        return sigma

    def sigma_T_at(T, T0):
        T = np.asarray(T, dtype=float)
        if len(T) == 0:
            return float("nan")
        sigma_T = sigma_T_local(T)
        return float(sigma_T[int(np.nanargmin(np.abs(T - T0)))])

    return sigma_I, sigma_R, sigma_T_at, sigma_T_local, sigma_V


@app.cell
def _load(MEAS_DIR, pd):
    """Load the accepted per-run CSV files.

    These cleaned CSVs are the source of truth for Part A. Each frame is clipped
    to the common transition window and sorted by temperature before analysis.
    """
    T_MIN, T_MAX = 80.0, 105.0
    _meas = {}
    for _path in sorted(MEAS_DIR.glob("partA_*.csv")):
        _df = pd.read_csv(_path)
        _df = _df[
            _df["temperature_K"].between(T_MIN, T_MAX)
            & _df["temperature_K"].notna()
            & _df["resistance_ohm"].notna()
        ]
        _meas[_path.stem] = _df.sort_values("temperature_K").reset_index(drop=True)
    measurements = _meas
    return T_MAX, T_MIN, measurements


@app.cell
def _md_tc_methods(
    SPLINE_SENSITIVITY_TARGETS_MOHM,
    SPLINE_TARGET_MOHM,
    mo,
):
    _target_list = ", ".join(f"{_t:g}" for _t in SPLINE_SENSITIVITY_TARGETS_MOHM)
    mo.md(rf"""
    ## Method

    Each run $R(T)$ is fit with a cubic smoothing spline $\hat R$ using SciPy's
    [`UnivariateSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html),
    targeting an RMS residual of
    $\sigma_\mathrm{{target}} = {SPLINE_TARGET_MOHM:g}\,\mathrm{{m}}\Omega$ —
    a level chosen by testing a few values, smooth enough to suppress
    point-to-point noise while staying on the measured shoulders. The transition
    temperature is the steepest-ascent point,

    $$T_c = \arg\max_{{T}} \hat R{{}}'(T),$$

    found on a dense grid and refined by bounded maximization of the analytic
    derivative.

    Its uncertainty combines two terms in quadrature,

    $$\sigma_{{T_c}} = \sqrt{{\sigma_\mathrm{{sampling}}^2 + \sigma_\mathrm{{smoothing}}^2}}.$$

    $\sigma_\mathrm{{sampling}} = \dfrac{{T_{{i+1}} - T_{{i-1}}}}{{2\sqrt{{12}}}}$, evaluated
    at the sample $T_i$ nearest $T_c$, models the local temperature bin as a
    uniform distribution over the spacing to its neighbors.
    $\sigma_\mathrm{{smoothing}} = \max_k \lvert T_c(\sigma_k) - T_c \rvert$ is the
    largest shift in $T_c$ as the target is swept over
    $\sigma_k \in \{{{_target_list}\}}\,\mathrm{{m}}\Omega$.
    """)
    return


@app.cell
def _spline_model_helpers(
    SPLINE_BRACKET_STEPS,
    SPLINE_DEGREE,
    SPLINE_GRID_POINTS,
    SPLINE_TARGET_MOHM,
    UnivariateSpline,
    minimize_scalar,
    np,
):
    def prepare_spline_inputs(T, R):
        T = np.asarray(T, dtype=float)
        R = np.asarray(R, dtype=float)
        ok = np.isfinite(T) & np.isfinite(R)
        T, R = T[ok], R[ok]
        order = np.argsort(T)
        T, R = T[order], R[order]
        if len(T) < SPLINE_DEGREE + 1:
            raise ValueError("not enough points for a cubic smoothing spline")
        return T, R

    def fit_spline(T, R, target_mohm=None):
        T, R = prepare_spline_inputs(T, R)
        target_mohm = SPLINE_TARGET_MOHM if target_mohm is None else float(target_mohm)
        s = len(T) * (target_mohm * 1e-3) ** 2
        spline = UnivariateSpline(T, R, k=SPLINE_DEGREE, s=s)
        rmse_mohm = float(np.sqrt(np.mean((R - spline(T)) ** 2))) * 1e3
        return dict(T=T, R=R, spline=spline, rmse_mohm=rmse_mohm)

    def derivative_peak(spline, T_min, T_max):
        d_spline = spline.derivative()
        T_grid = np.linspace(float(T_min), float(T_max), SPLINE_GRID_POINTS)
        dR_grid = np.asarray(d_spline(T_grid), dtype=float)
        finite = np.isfinite(dR_grid)
        if not np.any(finite):
            return float("nan"), float("nan"), T_grid, dR_grid

        finite_idx = np.flatnonzero(finite)
        i_peak = int(finite_idx[np.argmax(dR_grid[finite])])
        lo = float(T_grid[max(0, i_peak - SPLINE_BRACKET_STEPS)])
        hi = float(T_grid[min(len(T_grid) - 1, i_peak + SPLINE_BRACKET_STEPS)])

        if hi <= lo:
            tc = float(T_grid[i_peak])
        else:
            result = minimize_scalar(
                lambda T: -float(d_spline(T)),
                bounds=(lo, hi),
                method="bounded",
                options={"xatol": 1e-10},
            )
            if result.success and np.isfinite(result.x):
                tc = float(result.x)
            else:
                tc = float(T_grid[i_peak])

        return tc, float(d_spline(tc)), T_grid, dR_grid

    return derivative_peak, fit_spline


@app.cell
def _analysis(
    SPLINE_SENSITIVITY_TARGETS_MOHM,
    SPLINE_TARGET_MOHM,
    derivative_peak,
    fit_spline,
    measurements,
    np,
    pd,
    sigma_T_at,
):
    """Analyze each run once; the summary is the single source of truth."""

    def _tc_for_target(T, R, target_mohm):
        fit = fit_spline(T, R, target_mohm=target_mohm)
        tc, _, _, _ = derivative_peak(fit["spline"], fit["T"].min(), fit["T"].max())
        return tc

    def analyze_run(measurement_id, df):
        meta = df.iloc[0]
        T = df["temperature_K"].to_numpy(dtype=float)
        R = df["resistance_ohm"].to_numpy(dtype=float)

        fit = fit_spline(T, R, target_mohm=SPLINE_TARGET_MOHM)
        tc, dR_peak, T_grid, dR_grid = derivative_peak(
            fit["spline"],
            fit["T"].min(),
            fit["T"].max(),
        )

        target_values = np.array(SPLINE_SENSITIVITY_TARGETS_MOHM, dtype=float)
        tc_sweep = np.array([
            _tc_for_target(T, R, target_mohm)
            for target_mohm in target_values
        ])
        finite_tc = tc_sweep[np.isfinite(tc_sweep)]
        sigma_smoothing = float(np.max(np.abs(finite_tc - tc))) if len(finite_tc) else float("nan")
        sigma_sampling = sigma_T_at(fit["T"], tc)
        tc_err = float(np.hypot(sigma_sampling, sigma_smoothing))

        overlay = dict(
            T=T_grid,
            R=fit["spline"](T_grid),
            dR_dT=dR_grid,
            tc=tc,
            dR_dT_peak=dR_peak,
        )

        record = dict(
            measurement_id=measurement_id,
            sample_current_mA_nominal=float(meta["sample_current_mA_nominal"]),
            series_resistor=meta["series_resistor"],
            direction=meta["direction"],
            field_condition=meta["field_condition"],
            tc_K=tc,
            tc_err_K=tc_err,
            tc_sampling_err_K=sigma_sampling,
            tc_smoothing_err_K=sigma_smoothing,
            spline_target_mohm=SPLINE_TARGET_MOHM,
            spline_sensitivity_targets_mohm=";".join(f"{_t:g}" for _t in target_values),
            spline_rmse_mohm=fit["rmse_mohm"],
        )
        return record, overlay

    runs, spline_overlays = {}, {}
    for measurement_id, df in measurements.items():
        runs[measurement_id], spline_overlays[measurement_id] = analyze_run(
            measurement_id,
            df,
        )

    tc_summary = (
        pd.DataFrame(runs.values())
        .sort_values("measurement_id")
        .reset_index(drop=True)
    )
    return spline_overlays, tc_summary


@app.cell
def _md_error_model(mo):
    mo.md(r"""
    ## Measurement Errors

    Voltage and current uncertainties use meter accuracy plus last-digit
    resolution in quadrature. Resistance is propagated from
    $R=V/I$:

    $$\sigma_R = \sqrt{\left(\frac{\sigma_V}{I}\right)^2 + \left(\frac{V\sigma_I}{I^2}\right)^2}.$$

    The temperature term below is the typical local sampling-bin uncertainty.
    """)
    return


@app.cell
def _error_table(
    measurements,
    np,
    pd,
    sigma_I,
    sigma_R,
    sigma_T_local,
    sigma_V,
):
    _rows = []
    for _mid, _df in measurements.items():
        _meta = _df.iloc[0]
        _V = _df["voltage_V"].to_numpy(dtype=float)
        _I = _df["current_A"].to_numpy(dtype=float)
        _T = _df["temperature_K"].to_numpy(dtype=float)
        _rows.append({
            "I (mA)": int(_meta["sample_current_mA_nominal"]),
            "sweep": _meta["direction"],
            "field": _meta["field_condition"],
            "typ. sigma V (uV)": f"{np.nanmedian(sigma_V(_V)) * 1e6:.2f}",
            "typ. sigma I (uA)": f"{np.nanmedian(sigma_I(_I)) * 1e6:.1f}",
            "typ. sigma R (mOhm)": f"{np.nanmedian(sigma_R(_V, _I)) * 1e3:.3f}",
            "typ. sigma T (K)": f"{np.nanmedian(sigma_T_local(_T)):.3f}",
        })
    error_table = pd.DataFrame(_rows)
    return (error_table,)


@app.cell
def _show_error_table(error_table):
    error_table
    return


@app.cell
def _md_fit_diagnostics(mo):
    mo.md(r"""
    ## Spline Fit Diagnostics

    These check that the spline tracks the measured curve. The RMS residual
    should sit near the $0.04\,\mathrm{m}\Omega$ target; the blocked-holdout RMS
    and rolling-mean residual flag any local systematic miss.
    """)
    return


@app.cell
def _spline_diagnostics(
    SPLINE_TARGET_MOHM,
    fit_spline,
    measurements,
    np,
    pd,
):
    def _residual_stats(T, R, target_mohm):
        fit = fit_spline(T, R, target_mohm=target_mohm)
        residual = (fit["R"] - fit["spline"](fit["T"])) * 1e3
        window = max(5, len(residual) // 8)
        if window % 2 == 0:
            window += 1
        window = min(window, len(residual) if len(residual) % 2 else len(residual) - 1)
        if window < 3:
            rolling = np.array([np.nan])
        else:
            rolling = np.convolve(residual, np.ones(window) / window, mode="valid")
        return dict(
            rms_mohm=fit["rmse_mohm"],
            max_abs_mohm=float(np.max(np.abs(residual))),
            max_rolling_mean_mohm=float(np.nanmax(np.abs(rolling))),
        )

    def _blocked_holdout_rms(T, R, target_mohm, blocks=5):
        T = np.asarray(T, dtype=float)
        R = np.asarray(R, dtype=float)
        errors = []
        for offset in range(blocks):
            test = np.arange(len(T)) % blocks == offset
            train = ~test
            if train.sum() < 8 or test.sum() == 0:
                continue
            fit = fit_spline(T[train], R[train], target_mohm=target_mohm)
            errors.extend(((R[test] - fit["spline"](T[test])) * 1e3).tolist())
        errors = np.asarray(errors, dtype=float)
        return float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan")

    rows = []
    for _measurement_id, _df in measurements.items():
        meta = _df.iloc[0]
        T = _df["temperature_K"].to_numpy(dtype=float)
        R = _df["resistance_ohm"].to_numpy(dtype=float)
        stats = _residual_stats(T, R, SPLINE_TARGET_MOHM)
        rows.append({
            "I (mA)": int(meta["sample_current_mA_nominal"]),
            "sweep": meta["direction"],
            "field": meta["field_condition"],
            "target (mOhm)": f"{SPLINE_TARGET_MOHM:.2f}",
            "RMS resid (mOhm)": f"{stats['rms_mohm']:.3f}",
            "holdout RMS (mOhm)": f"{_blocked_holdout_rms(T, R, SPLINE_TARGET_MOHM):.3f}",
            "max |resid| (mOhm)": f"{stats['max_abs_mohm']:.3f}",
            "max rolling mean (mOhm)": f"{stats['max_rolling_mean_mohm']:.3f}",
        })
    spline_diagnostics = pd.DataFrame(rows)
    return (spline_diagnostics,)


@app.cell
def _show_spline_diagnostics(spline_diagnostics):
    spline_diagnostics
    return


@app.cell
def _trace_helpers(sigma_R, sigma_T_local):
    def trace(df):
        T = df["temperature_K"].to_numpy(dtype=float)
        R = df["resistance_ohm"].to_numpy(dtype=float)
        R_err = sigma_R(df["voltage_V"].to_numpy(dtype=float), df["current_A"].to_numpy(dtype=float))
        T_err = sigma_T_local(T)
        return dict(T=T, R=R, R_err=R_err, T_err=T_err)

    return (trace,)


@app.cell
def _pair_keys(tc_summary):
    """Group paired runs for readable comparison plots."""
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
def _two_panel(T_MAX, T_MIN, plt):
    def build(_title=None):
        fig, (ax_r, ax_d) = plt.subplots(
            2, 1, figsize=(7.2, 5.6), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.07},
        )
        if _title:
            fig.suptitle(_title, x=0.5, y=0.97, fontsize=13, fontweight="medium")
        ax_r.set_ylabel(r"$R\;\;(\mathrm{m}\Omega)$")
        ax_d.set_xlabel(r"$T\;\;(\mathrm{K})$")
        ax_d.set_ylabel(r"$R'\;\;(\mathrm{m}\Omega/\mathrm{K})$")
        for ax in (ax_r, ax_d):
            ax.grid(True, axis="y", alpha=0.1, lw=0.5)
            ax.tick_params(direction="in", length=3, top=False, right=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax_d.set_xlim(T_MIN - 1, T_MAX + 1)
        return fig, ax_r, ax_d

    def legend_with_info(ax, info):
        ax.legend(
            loc="upper left", frameon=False, fontsize=8.5,
            title=f"${info}$" if not info.startswith("$") else info,
            title_fontsize=9, alignment="left", handlelength=1.6,
        )

    def draw(ax_r, ax_d, tr, sp, color, marker, label):
        tc = sp["tc"]
        ax_r.errorbar(
            tr["T"], tr["R"] * 1e3,
            xerr=tr["T_err"], yerr=tr["R_err"] * 1e3,
            fmt=marker, ms=3.6, mfc="none", mec=color, mew=0.7,
            ecolor=color, elinewidth=0.5, capsize=0,
            alpha=0.6, zorder=2,
        )
        ax_r.plot(
            sp["T"], sp["R"] * 1e3, lw=1.6, color=color, zorder=3,
            label=rf"{label},  $T_c = {tc:.2f}\,$K",
        )
        ax_r.axvline(tc, color=color, ls=(0, (5, 3)), lw=0.8, alpha=0.45, zorder=1)

        ax_d.plot(sp["T"], sp["dR_dT"] * 1e3, color=color, lw=1.4, alpha=0.85)
        ax_d.axvline(tc, color=color, ls=(0, (5, 3)), lw=0.8, alpha=0.45, zorder=1)

    return build, draw, legend_with_info


@app.cell
def _pair_clip():
    """Clip paired display ranges to their common temperature overlap."""

    def intersection(dfs):
        T_lo = max(float(d["temperature_K"].min()) for d in dfs)
        T_hi = min(float(d["temperature_K"].max()) for d in dfs)
        return T_lo, T_hi

    def clip(df, T_lo, T_hi):
        mask = (df["temperature_K"] >= T_lo) & (df["temperature_K"] <= T_hi)
        return df[mask].reset_index(drop=True)

    return clip, intersection


@app.cell
def _make_figure(build, clip, draw, legend_with_info, spline_overlays, trace):
    def make_figure(members, styles, key_of, info, title, clip_range=None):
        fig, ax_r, ax_d = build(title)
        for _mid, _df in members:
            _d = clip(_df, *clip_range) if clip_range else _df
            _color, _marker, _label = styles[key_of(_d.iloc[0])]
            draw(
                ax_r,
                ax_d,
                trace(_d),
                spline_overlays[_mid],
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

    _lag_30 = _heat_30["tc_K"] - _cool_30["tc_K"]
    _field_shift = _heat_100["tc_K"] - _field_100["tc_K"]

    mo.md(rf"""
    ## Main Comparisons

    Dashed lines mark $T_c = \arg\max_T R'(T)$; the lower panel shows $R'(T)$.

    - **Thermal hysteresis:** at $30\,\mathrm{{mA}}$, heating is
      ${_lag_30:.2f}\,\mathrm{{K}}$ above cooling.
    - **Applied field:** at $100\,\mathrm{{mA}}$, the field lowers $T_c$ by
      ${_field_shift:.2f}\,\mathrm{{K}}$.
    - **High current:** the $240\,\mathrm{{mA}}$ cooling run gives the lowest
      $T_c$, ${_cool_240["tc_K"]:.2f}\,\mathrm{{K}}$.
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

    Each $T_c = \arg\max_T R'(T)$ is reported with the combined uncertainty
    $\sigma_{T_c}$. The figure orders the runs by $T_c$. Tables and figure are
    written to `results/part_a/`.
    """)
    return


@app.cell
def _final_table(pd, tc_summary):
    def _fmt(val, err):
        rel = (err / abs(val) * 100.0) if val else float("nan")
        return f"{val:.2f} +/- {err:.2f} ({rel:.2f}%)"

    _rows = []
    for _, _r in tc_summary.iterrows():
        _rows.append({
            "I (mA)": int(_r["sample_current_mA_nominal"]),
            "R_s": _r["series_resistor"],
            "sweep": _r["direction"],
            "field": _r["field_condition"],
            "Tc [K]": _fmt(_r["tc_K"], _r["tc_err_K"]),
            "sigma samp [K]": f"{_r['tc_sampling_err_K']:.3f}",
            "sigma smooth [K]": f"{_r['tc_smoothing_err_K']:.3f}",
            "target [mOhm]": f"{_r['spline_target_mohm']:.2f}",
            "sweep [mOhm]": _r["spline_sensitivity_targets_mohm"],
            "RMS resid [mOhm]": f"{_r['spline_rmse_mohm']:.3f}",
        })
    final_table = pd.DataFrame(_rows)
    return (final_table,)


@app.cell
def _show_final(final_table):
    final_table
    return


@app.cell
def _summary_plot(
    DIR_STYLES,
    Line2D,
    OUT_DIR,
    fmt_I,
    fmt_dir,
    fmt_field,
    plt,
    tc_summary,
):
    """Forest-style overview of the reported Tc per run, ordered by value.

    Color and marker encode sweep direction (heating / cooling), matching the
    two-panel comparison plots.
    """
    _df = tc_summary.sort_values("tc_K").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for _i, _r in _df.iterrows():
        _color, _marker, _ = DIR_STYLES[_r["direction"]]
        ax.errorbar(
            _r["tc_K"], _i, xerr=_r["tc_err_K"],
            fmt=_marker, ms=6, color=_color,
            mfc=_color, mec=_color, mew=1.2,
            ecolor=_color, elinewidth=0.9, capsize=0, zorder=3,
        )
        ax.text(
            _r["tc_K"] + _r["tc_err_K"] + 0.18, _i,
            rf"${_r['tc_K']:.2f}$", va="center", ha="left",
            fontsize=8, color="#444444",
        )

    _labels = [
        rf"${fmt_I(_r['sample_current_mA_nominal'])}$,  {fmt_dir(_r['direction'])},  "
        rf"${fmt_field(_r['field_condition'])}$"
        for _, _r in _df.iterrows()
    ]
    ax.set_yticks(range(len(_df)))
    ax.set_yticklabels(_labels)
    ax.set_ylim(-0.6, len(_df) - 0.4)
    ax.set_xlabel(r"$T_c\;\;(\mathrm{K})$")
    _lo = float((_df["tc_K"] - _df["tc_err_K"]).min())
    _hi = float((_df["tc_K"] + _df["tc_err_K"]).max())
    ax.set_xlim(_lo - 0.6, _hi + 1.6)
    ax.grid(True, axis="x", alpha=0.1, lw=0.5)
    ax.tick_params(axis="x", direction="in", length=3, top=False)
    ax.tick_params(axis="y", length=0)
    for _side in ("top", "right", "left"):
        ax.spines[_side].set_visible(False)

    _legend = [
        Line2D([0], [0], marker=DIR_STYLES["heat"][1], ls="none",
               color=DIR_STYLES["heat"][0], label="heating"),
        Line2D([0], [0], marker=DIR_STYLES["cool"][1], ls="none",
               color=DIR_STYLES["cool"][0], label="cooling"),
    ]
    ax.legend(handles=_legend, loc="lower right", frameon=False,
              fontsize=8.5, handletextpad=0.4, labelspacing=0.3)
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.13, top=0.96)

    summary_fig = fig
    summary_plot_path = OUT_DIR / "tc_summary.png"
    fig.savefig(summary_plot_path)
    print(f"wrote {summary_plot_path}")
    return (summary_fig,)


@app.cell
def _show_summary_plot(summary_fig):
    summary_fig
    return


@app.cell
def _write(OUT_DIR, spline_diagnostics, tc_summary):
    tc_path = OUT_DIR / "tc_summary.csv"
    diagnostics_path = OUT_DIR / "spline_diagnostics.csv"
    tc_summary.to_csv(tc_path, index=False)
    spline_diagnostics.to_csv(diagnostics_path, index=False)
    print(f"wrote {tc_path}")
    print(f"wrote {diagnostics_path}")
    return


if __name__ == "__main__":
    app.run()
