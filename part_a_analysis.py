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

    We study the phase transition of a
    $\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$ superconductor of type II.

    We measure $I,V,T$ and via Ohm's law translate to a $R(T)$ scatter.
    We perform a Savitzky-Golay local-polynomial filter to get a smooth $R(T)$ curve
    composed of polynomials of order 3. We then differentiate the smoothed curve to get $R'(T)$ and define the critical temperature as
    $$T_c = \arg\max_T R'(T).$$

    ### Uncertainties

    For each $T$ measurement, the local midpoint-to-midpoint bin has width
    $\frac{(T_{{i+1}}-T_{{i-1}})}{2}$. Modeling the unknown location inside that bin as
    uniform gives

    \[
    \sigma_\mathrm{{sampling}} =
    \frac{T_{{i+1}}-T_{{i-1}}}{2\sqrt{{12}}}.
    \]

    Additionally, to compensate for the fact that the exact $T_c$ point
    is difficult to assign precisely we assume that the true $T_c$ is somewhere near the peak of the derivative.
    We define the width of the $R'(T)$ peaking as a plausible range to find the true critical temperature.
    We differentiate once more to get $R''(T)$ and define the range as

    \[
    \Delta T = \max\left(T_c-T_L,\;T_R-T_c\right).
    \]

    Where $T_L$ and $T_R$ are the temperatures at the left and right zero crossings of $R''(T)$.
    We then treat $T_c$ as the peak of a Gaussian likelihood and $\Delta T$ as an approximate two-sigma width, giving

    \[
    \sigma_{{T_c}}=\frac{{\Delta T}}{{2}}.
    \]
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
    return Line2D, Path, mo, np, pd, plt, savgol_filter


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
    SG_POLYORDER = 3
    SG_WINDOW_K = 3.25
    SG_GRID_STEP_K = 0.02
    return SG_GRID_STEP_K, SG_POLYORDER, SG_WINDOW_K


@app.cell
def _instrument(np):
    """Per-point instrument and sampling uncertainties.

    Rigol DM3058 5.5-digit DMM uncertainties are accuracy and last-digit
    resolution in quadrature. Resistance uncertainty is propagated directly
    from R = V/I. Temperature uncertainty is the local midpoint-to-midpoint bin
    around each measured temperature, modeled as a uniform distribution.
    """
    V_RANGE, V_LSD = 0.1, 1e-6
    V_RES = V_LSD / np.sqrt(12.0)

    def sigma_V(V):
        V = np.asarray(V, dtype=float)
        return np.sqrt((0.00015 * np.abs(V) + 0.00004 * V_RANGE) ** 2 + V_RES**2)

    def sigma_current(current):
        current = np.asarray(current, dtype=float)
        I_RANGE = np.where(np.abs(current) <= 0.2, 0.2, 2.0)
        I_RES = np.where(
            np.abs(current) <= 0.2,
            1e-6 / np.sqrt(12.0),
            1e-5 / np.sqrt(12.0),
        )
        return np.sqrt(
            (0.0025 * np.abs(current) + 0.00020 * I_RANGE) ** 2 + I_RES**2
        )

    def sigma_R(voltage, current):
        voltage = np.asarray(voltage, dtype=float)
        current = np.asarray(current, dtype=float)
        return np.sqrt(
            (sigma_V(voltage) / current) ** 2
            + (voltage * sigma_current(current) / current**2) ** 2
        )

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
            width[1:-1] = T_sorted[2:] - T_sorted[:-2]

        sigma_sorted = np.abs(width) / (2.0 * np.sqrt(12.0))
        sigma = np.empty_like(sigma_sorted)
        sigma[order] = sigma_sorted
        return sigma

    def sigma_T_at(T, T0):
        T = np.asarray(T, dtype=float)
        if len(T) == 0:
            return float("nan")
        sigma_T = sigma_T_local(T)
        return float(sigma_T[int(np.nanargmin(np.abs(T - T0)))])

    return sigma_R, sigma_T_at, sigma_T_local


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
def _savgol_model_helpers(
    SG_GRID_STEP_K,
    SG_POLYORDER,
    SG_WINDOW_K,
    np,
    savgol_filter,
):
    def prepare_savgol_inputs(T, R):
        T = np.asarray(T, dtype=float)
        R = np.asarray(R, dtype=float)
        ok = np.isfinite(T) & np.isfinite(R)
        T, R = T[ok], R[ok]
        order = np.argsort(T)
        T, R = T[order], R[order]
        if len(T) < SG_POLYORDER + 3:
            raise ValueError("not enough points for Savitzky-Golay filtering")
        return T, R

    def _window_points(n_grid, window_K):
        points = int(round(float(window_K) / SG_GRID_STEP_K))
        if points % 2 == 0:
            points += 1
        minimum = SG_POLYORDER + 2
        if minimum % 2 == 0:
            minimum += 1
        points = max(points, minimum)
        if points > n_grid:
            points = n_grid if n_grid % 2 == 1 else n_grid - 1
        if points <= SG_POLYORDER:
            raise ValueError("Savitzky-Golay window is too short")
        return points

    def _transition_bounds(T, R):
        n_edge = max(5, len(T) // 10)
        low = float(np.nanmedian(R[:n_edge]))
        high = float(np.nanmedian(R[-n_edge:]))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return float(T.min()), float(T.max())
        normalized = (R - low) / (high - low)
        in_transition = (normalized >= 0.05) & (normalized <= 0.95)
        if np.count_nonzero(in_transition) < 5:
            return float(T.min()), float(T.max())
        left = max(float(T.min()), float(T[in_transition][0]) - 1.0)
        right = min(float(T.max()), float(T[in_transition][-1]) + 1.0)
        return (left, right) if right > left else (float(T.min()), float(T.max()))

    def fit_savgol(T, R, window_K=None):
        T, R = prepare_savgol_inputs(T, R)
        window_K = SG_WINDOW_K if window_K is None else float(window_K)
        T_grid = np.arange(float(T.min()), float(T.max()) + SG_GRID_STEP_K / 2.0, SG_GRID_STEP_K)
        R_uniform = np.interp(T_grid, T, R)
        window_points = _window_points(len(T_grid), window_K)
        R_smooth = savgol_filter(
            R_uniform,
            window_points,
            SG_POLYORDER,
            deriv=0,
            mode="interp",
        )
        dR_grid = savgol_filter(
            R_uniform,
            window_points,
            SG_POLYORDER,
            deriv=1,
            delta=SG_GRID_STEP_K,
            mode="interp",
        )
        return dict(
            T=T,
            R=R,
            T_grid=T_grid,
            R_grid=R_smooth,
            dR_grid=dR_grid,
            window_K=window_K,
            window_points=window_points,
            polyorder=SG_POLYORDER,
            search_bounds=_transition_bounds(T, R),
        )

    def _quadratic_peak(T_grid, y_grid, i_peak):
        if i_peak <= 0 or i_peak >= len(T_grid) - 1:
            return float(T_grid[i_peak]), float(y_grid[i_peak])
        x = T_grid[i_peak - 1 : i_peak + 2]
        y = y_grid[i_peak - 1 : i_peak + 2]
        if not np.all(np.isfinite(y)):
            return float(T_grid[i_peak]), float(y_grid[i_peak])
        a, b, c = np.polyfit(x, y, deg=2)
        if a >= 0:
            return float(T_grid[i_peak]), float(y_grid[i_peak])
        vertex = float(-b / (2.0 * a))
        if x[0] <= vertex <= x[-1]:
            return vertex, float(a * vertex**2 + b * vertex + c)
        return float(T_grid[i_peak]), float(y_grid[i_peak])

    def derivative_peak(fit):
        T_grid = fit["T_grid"]
        dR_grid = np.asarray(fit["dR_grid"], dtype=float)
        lo, hi = fit["search_bounds"]
        finite = np.isfinite(dR_grid)
        in_bounds = (T_grid >= lo) & (T_grid <= hi)
        valid = finite & in_bounds
        if not np.any(valid):
            return float("nan"), float("nan"), T_grid, dR_grid

        valid_idx = np.flatnonzero(valid)
        i_peak = int(valid_idx[np.argmax(dR_grid[valid])])
        tc, dR_peak = _quadratic_peak(T_grid, dR_grid, i_peak)

        return tc, dR_peak, T_grid, dR_grid

    return derivative_peak, fit_savgol


@app.cell
def _analysis(
    SG_GRID_STEP_K,
    SG_POLYORDER,
    SG_WINDOW_K,
    derivative_peak,
    fit_savgol,
    measurements,
    np,
    pd,
    sigma_T_at,
):
    """Analyze each run once; the summary is the single source of truth."""

    def _linear_zero(T0, y0, T1, y1):
        if y1 == y0:
            return float(0.5 * (T0 + T1))
        return float(T0 - y0 * (T1 - T0) / (y1 - y0))

    def _curvature_zero_temperature(T_grid, curvature, i_center, side):
        finite = np.isfinite(curvature)

        if side == "left":
            for i in range(i_center, 0, -1):
                if (
                    finite[i]
                    and finite[i - 1]
                    and curvature[i] < 0 <= curvature[i - 1]
                ):
                    return _linear_zero(
                        T_grid[i - 1],
                        curvature[i - 1],
                        T_grid[i],
                        curvature[i],
                    )
            mask = finite & (T_grid <= T_grid[i_center])
        else:
            for i in range(i_center, len(T_grid) - 1):
                if (
                    finite[i]
                    and finite[i + 1]
                    and curvature[i] < 0 <= curvature[i + 1]
                ):
                    return _linear_zero(
                        T_grid[i],
                        curvature[i],
                        T_grid[i + 1],
                        curvature[i + 1],
                    )
            mask = finite & (T_grid >= T_grid[i_center])

        local_T = T_grid[mask]
        local_curvature = curvature[mask]
        if len(local_T) == 0:
            return float(T_grid[i_center])
        return float(local_T[int(np.nanargmin(np.abs(local_curvature)))])

    def _transition_interval(tc, T_grid, dR_grid):
        curvature = np.gradient(
            np.gradient(dR_grid, T_grid, edge_order=2),
            T_grid,
            edge_order=2,
        )
        finite = np.isfinite(curvature)
        i_center = int(np.nanargmin(np.abs(T_grid - tc)))

        if not (finite[i_center] and curvature[i_center] < 0):
            concave_down = np.flatnonzero(finite & (curvature < 0))
            if len(concave_down):
                i_center = int(
                    concave_down[np.argmin(np.abs(T_grid[concave_down] - tc))]
                )

        T_left = _curvature_zero_temperature(T_grid, curvature, i_center, "left")
        T_right = _curvature_zero_temperature(T_grid, curvature, i_center, "right")
        if T_right < T_left:
            T_left, T_right = T_right, T_left

        delta = float(max(abs(tc - T_left), abs(T_right - tc)))
        interval_left = float(tc - delta)
        interval_right = float(tc + delta)
        interval_width = float(2.0 * delta)
        curvature_width = float(T_right - T_left)

        detail = dict(
            T_left=interval_left,
            T_right=interval_right,
            curvature_left=T_left,
            curvature_right=T_right,
            curvature_width_K=curvature_width,
            delta_K=delta,
            width_K=interval_width,
            curvature_at_center=float(curvature[i_center]),
        )
        return delta, interval_width, detail

    def analyze_run(measurement_id, df):
        meta = df.iloc[0]
        T = df["temperature_K"].to_numpy(dtype=float)
        R = df["resistance_ohm"].to_numpy(dtype=float)

        fit = fit_savgol(T, R, window_K=SG_WINDOW_K)
        tc, _dR_peak, T_grid, dR_grid = derivative_peak(fit)

        interval_delta, interval_width, interval_detail = (
            _transition_interval(tc, T_grid, dR_grid)
        )
        tc_sigma = float(interval_delta / 2.0)
        sigma_sampling = sigma_T_at(fit["T"], tc)

        overlay = dict(
            T=T_grid,
            R=fit["R_grid"],
            dR_dT=dR_grid,
            tc=tc,
            transition_Tleft=interval_detail["T_left"],
            transition_Tright=interval_detail["T_right"],
        )

        record = dict(
            measurement_id=measurement_id,
            sample_current_mA_nominal=float(meta["sample_current_mA_nominal"]),
            series_resistor=meta["series_resistor"],
            direction=meta["direction"],
            field_condition=meta["field_condition"],
            tc_K=tc,
            tc_sigma_K=tc_sigma,
            tc_sampling_sigma_K=sigma_sampling,
            tc_interval_delta_K=interval_detail["delta_K"],
            tc_interval_width_K=interval_width,
            tc_interval_left_K=interval_detail["T_left"],
            tc_interval_right_K=interval_detail["T_right"],
            tc_interval_method="derivative_curvature_zero_crossing",
            tc_derivative_curvature_left_K=interval_detail["curvature_left"],
            tc_derivative_curvature_right_K=interval_detail["curvature_right"],
            tc_derivative_curvature_width_K=interval_detail[
                "curvature_width_K"
            ],
            tc_derivative_curvature_at_center=interval_detail[
                "curvature_at_center"
            ],
            method="Savitzky-Golay",
            sg_polyorder=SG_POLYORDER,
            sg_window_K=fit["window_K"],
            sg_window_points=fit["window_points"],
            sg_grid_step_K=SG_GRID_STEP_K,
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
def _trace_helpers(sigma_R, sigma_T_local):
    def trace(df):
        T = df["temperature_K"].to_numpy(dtype=float)
        R = df["resistance_ohm"].to_numpy(dtype=float)
        R_err = sigma_R(
            df["voltage_V"].to_numpy(dtype=float),
            df["current_A"].to_numpy(dtype=float),
        )
        T_err = sigma_T_local(T)
        return dict(T=T, R=R, R_err=R_err, T_err=T_err)

    return (trace,)


@app.cell
def _pair_keys(tc_summary):
    """Group paired runs for readable comparison plots."""
    heat_cool_groups, magnet_groups = {}, {}
    _paired = set()
    _no_mag = tc_summary[tc_summary["field_condition"] == "no_magnet"]
    for (_current_mA, _series_resistor), _group in _no_mag.groupby(
        ["sample_current_mA_nominal", "series_resistor"]
    ):
        if set(_group["direction"]) >= {"heat", "cool"}:
            heat_cool_groups[(float(_current_mA), _series_resistor)] = (
                _group["measurement_id"].tolist()
            )
            _paired.update(_group["measurement_id"])
    for (_current_mA, _series_resistor, _direction), _group in tc_summary.groupby(
        ["sample_current_mA_nominal", "series_resistor", "direction"]
    ):
        if set(_group["field_condition"]) >= {"magnet", "no_magnet"}:
            magnet_groups[(float(_current_mA), _series_resistor, _direction)] = (
                _group["measurement_id"].tolist()
            )
            _paired.update(_group["measurement_id"])
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

    return fmt_I, fmt_R, fmt_dir


@app.cell
def _transition_plot_helpers(T_MAX, T_MIN, np, plt):
    def build(_title=None):
        fig, (ax_r, ax_d) = plt.subplots(
            2, 1, figsize=(7.2, 4.9), sharex=True,
            gridspec_kw={"height_ratios": [2.35, 1.0], "hspace": 0.08},
        )
        if _title:
            fig.suptitle(_title, x=0.5, y=0.97, fontsize=13, fontweight="medium")
        ax_r.set_ylabel(r"$R\;\;(\mathrm{m}\Omega)$")
        ax_d.set_ylabel(r"$R'\;\;(\mathrm{m}\Omega/\mathrm{K})$")
        ax_d.set_xlabel(r"$T\;\;(\mathrm{K})$")
        for ax in (ax_r, ax_d):
            ax.grid(True, axis="y", alpha=0.1, lw=0.5)
            ax.tick_params(direction="in", length=3, top=False, right=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(T_MIN - 1, T_MAX + 1)
        return fig, ax_r, ax_d

    def legend_with_info(ax, info):
        ax.legend(
            loc="upper left", frameon=False, fontsize=8.2,
            title=f"${info}$" if not info.startswith("$") else info,
            title_fontsize=8.8, alignment="left", handlelength=1.6,
        )

    def draw(ax_r, ax_d, tr, sp, color, marker, label):
        tc = sp["tc"]
        tleft = sp.get("transition_Tleft")
        tright = sp.get("transition_Tright")
        if (
            tleft is not None
            and tright is not None
            and np.isfinite(tleft)
            and np.isfinite(tright)
        ):
            ax_d.axvspan(
                min(tleft, tright),
                max(tleft, tright),
                color=color,
                alpha=0.075,
                lw=0,
                zorder=0,
            )
        ax_r.plot(
            tr["T"], tr["R"] * 1e3,
            marker=marker, ls="none", ms=3.2, mfc="white", mec=color, mew=0.65,
            alpha=0.48, zorder=2,
        )
        ax_r.plot(
            sp["T"], sp["R"] * 1e3, lw=1.6, color=color, zorder=3,
            label=rf"{label},  $T_c = {tc:.2f}\,$K",
        )
        ax_r.axvline(tc, color=color, ls=(0, (5, 3)), lw=0.8, alpha=0.25, zorder=1)

        ax_d.plot(sp["T"], sp["dR_dT"] * 1e3, color=color, lw=1.4, alpha=0.85)
        ax_d.axvline(tc, color=color, ls=(0, (5, 3)), lw=0.8, alpha=0.55, zorder=1)

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
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.10, top=0.89)
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
    ## Results

    In each figure, the upper panel is the measured $R(T)$ curve with the
    Savitzky-Golay smoothed curve. The shaded bands in the
    derivative panel show $\Delta T$.
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
    for (_current_mA, _series_resistor), _measurement_ids in sorted(
        heat_cool_groups.items()
    ):
        _members = [(m, measurements[m]) for m in _measurement_ids]
        _range = intersection([measurements[m] for m in _measurement_ids])
        _info = rf"${fmt_I(_current_mA)}$,  ${fmt_R(_series_resistor)}$"
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
    for (_current_mA, _series_resistor, _direction), _measurement_ids in sorted(
        magnet_groups.items()
    ):
        _members = [(m, measurements[m]) for m in _measurement_ids]
        _range = intersection([measurements[m] for m in _measurement_ids])
        _info = (
            rf"${fmt_I(_current_mA)}$,  ${fmt_R(_series_resistor)}$,  "
            rf"{fmt_dir(_direction)}"
        )
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
            rf"{fmt_dir(_meta['direction'])}"
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
def _reported_tc(np, tc_summary):
    def _weighted_estimate(_mask, _label):
        _value = tc_summary.loc[_mask, "tc_K"].to_numpy(dtype=float)
        _sigma = tc_summary.loc[_mask, "tc_sigma_K"].to_numpy(dtype=float)
        _valid = np.isfinite(_value) & np.isfinite(_sigma) & (_sigma > 0)
        if not np.any(_valid):
            return None
        _weights = 1.0 / _sigma[_valid] ** 2
        return dict(
            label=_label,
            tc_K=float(np.average(_value[_valid], weights=_weights)),
            sigma_K=float(np.sqrt(1.0 / np.sum(_weights))),
            n_runs=int(np.count_nonzero(_valid)),
        )

    _zero_field = tc_summary["field_condition"] == "no_magnet"
    reported_tc = _weighted_estimate(_zero_field, "heating and cooling")
    reported_tc_by_direction = {
        "heat": _weighted_estimate(
            _zero_field & (tc_summary["direction"] == "heat"),
            "heating",
        ),
        "cool": _weighted_estimate(
            _zero_field & (tc_summary["direction"] == "cool"),
            "cooling",
        ),
    }
    return reported_tc, reported_tc_by_direction


@app.cell
def _md_final(mo, reported_tc, reported_tc_by_direction):
    def _fmt_estimate(_estimate):
        if _estimate is None:
            return "not available"
        return (
            rf"$T_c = {_estimate['tc_K']:.2f}\pm{_estimate['sigma_K']:.2f}\,\mathrm{{K}}$"
        )

    _reported = (
        rf"The estimate is the $1/\sigma_{{T_c}}^2$ weighted mean of the "
        rf"{reported_tc['n_runs']} heating/cooling runs: "
        rf"{_fmt_estimate(reported_tc)}. The applied-field run is retained in the table as a "
        rf"comparison point but excluded from this estimate."
        if reported_tc is not None
        else "The table below reports each run individually; no weighted estimate is available."
    )
    _heating = _fmt_estimate(reported_tc_by_direction["heat"])
    _cooling = _fmt_estimate(reported_tc_by_direction["cool"])
    mo.md(rf"""
    ## Summary

    The table below summarizes the transition temperatures extracted from each run, ordered by value. 
    The estimate is a weighted mean of the heating/cooling runs.

    {_reported}

    Heating-only weighted mean: {_heating}.  
    Cooling-only weighted mean: {_cooling}.

    The quoted $\pm$ values are standard uncertainties. The summary plots use
    $2\sigma$ bands and run error bars.
    """)
    return


@app.cell
def _final_table(pd, tc_summary):
    def _condition(row):
        label = (
            f"{int(row['sample_current_mA_nominal'])} mA, "
            f"{row['series_resistor']}, {row['direction']}"
        )
        if row["field_condition"] == "magnet":
            label = f"{label}, applied field"
        return label

    _rows = []
    for _, _r in tc_summary.sort_values("tc_K").iterrows():
        _rows.append({
            "condition": _condition(_r),
            "Tc +/- sigma [K]": f"{_r['tc_K']:.2f} +/- {_r['tc_sigma_K']:.2f}",
            "two-sigma interval [K]": (
                f"{_r['tc_interval_left_K']:.2f}"
                f" - {_r['tc_interval_right_K']:.2f}"
            ),
            "curvature interval [K]": (
                f"{_r['tc_derivative_curvature_left_K']:.2f}"
                f" - {_r['tc_derivative_curvature_right_K']:.2f}"
            ),
            "sampling sigma [K]": f"{_r['tc_sampling_sigma_K']:.2f}",
            "transition Delta [K]": f"{_r['tc_interval_delta_K']:.2f}",
            "sigma [K]": f"{_r['tc_sigma_K']:.2f}",
        })
    final_table = pd.DataFrame(_rows)
    return (final_table,)


@app.cell
def _show_final(final_table):
    final_table  # type: ignore
    return


@app.cell
def _summary_plot_helper(DIR_STYLES, Line2D, OUT_DIR, fmt_I, fmt_R, plt):
    def make_tc_summary_plot(_df, _estimate, _title, _filename):
        _df = _df.sort_values("tc_K").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7.0, 3.9))

        if _estimate is not None:
            _final_tc = _estimate["tc_K"]
            _final_2sigma = 2.0 * _estimate["sigma_K"]
            ax.axvspan(
                _final_tc - _final_2sigma,
                _final_tc + _final_2sigma,
                color="black",
                alpha=0.08,
                lw=0,
                zorder=0,
            )
            ax.axvline(
                _final_tc,
                color="black",
                lw=1.0,
                ls=(0, (4, 3)),
                alpha=0.75,
                zorder=1,
            )
            ax.set_title(
                rf"{_title}: $T_c = {_final_tc:.2f}\pm{_estimate['sigma_K']:.2f}\,\mathrm{{K}}$",
                fontsize=10,
                color="black",
                pad=8,
            )

        for _i, _r in _df.iterrows():
            _color, _marker, _ = DIR_STYLES[_r["direction"]]
            _xerr = 2.0 * _r["tc_sigma_K"]
            ax.errorbar(
                _r["tc_K"], _i, xerr=_xerr,
                fmt=_marker, ms=5.8,
                color=_color, alpha=0.82,
                mfc=_color, mec=_color,
                mew=1.0,
                ecolor=_color, elinewidth=0.9,
                capsize=0, zorder=3,
            )
            ax.text(
                _r["tc_K"] + _xerr + 0.18, _i,
                rf"${_r['tc_K']:.2f}$", va="center", ha="left",
                fontsize=8,
                color="black",
            )

        _labels = [
            rf"${fmt_I(_r['sample_current_mA_nominal'])}$,  "
            rf"${fmt_R(_r['series_resistor'])}$"
            for _, _r in _df.iterrows()
        ]
        ax.set_yticks(range(len(_df)))
        ax.set_yticklabels(_labels)
        ax.set_ylim(-0.6, len(_df) - 0.4)
        ax.set_xlabel(r"$T_c\;\;(\mathrm{K})$, error bars show $2\sigma_{T_c}$")
        _lo = float((_df["tc_K"] - 2.0 * _df["tc_sigma_K"]).min())
        _hi = float((_df["tc_K"] + 2.0 * _df["tc_sigma_K"]).max())
        ax.set_xlim(_lo - 0.6, _hi + 1.6)
        ax.grid(True, axis="x", alpha=0.1, lw=0.5)
        ax.tick_params(axis="x", direction="in", length=3, top=False)
        ax.tick_params(axis="y", length=0)
        for _side in ("top", "right", "left"):
            ax.spines[_side].set_visible(False)

        _directions = set(_df["direction"])
        _legend = [
            Line2D([0], [0], color="black", lw=7, alpha=0.22,
                   label=r" $2\sigma$"),
        ]
        if "heat" in _directions:
            _legend.append(
                Line2D([0], [0], marker=DIR_STYLES["heat"][1], ls="none",
                       color=DIR_STYLES["heat"][0], label="heating")
            )
        if "cool" in _directions:
            _legend.append(
                Line2D([0], [0], marker=DIR_STYLES["cool"][1], ls="none",
                       color=DIR_STYLES["cool"][0], label="cooling")
            )
        ax.legend(handles=_legend, loc="lower right", frameon=False,
                  fontsize=8.5, handletextpad=0.4, labelspacing=0.3)
        fig.subplots_adjust(left=0.36, right=0.97, bottom=0.13, top=0.90)
        _path = OUT_DIR / _filename
        fig.savefig(_path, dpi=300)
        fig.savefig(_path.with_suffix(".pdf"))
        return fig

    return (make_tc_summary_plot,)


@app.cell
def _summary_plot(
    make_tc_summary_plot,
    reported_tc,
    reported_tc_by_direction,
    tc_summary,
):
    """Forest-style overview of the reported Tc per run, ordered by value.

    Color and marker encode sweep direction (heating / cooling), matching the
    two-panel comparison plots.
    """
    _zero_field = tc_summary[tc_summary["field_condition"] == "no_magnet"]
    summary_figs = [
        make_tc_summary_plot(
            _zero_field,
            reported_tc,
            "heating and cooling weighted estimate",
            "tc_summary.png",
        ),
        make_tc_summary_plot(
            _zero_field[_zero_field["direction"] == "cool"],
            reported_tc_by_direction["cool"],
            "cooling weighted estimate",
            "tc_summary_cool.png",
        ),
        make_tc_summary_plot(
            _zero_field[_zero_field["direction"] == "heat"],
            reported_tc_by_direction["heat"],
            "heating weighted estimate",
            "tc_summary_heat.png",
        ),
    ]
    return (summary_figs,)


@app.cell
def _write(OUT_DIR, tc_summary):
    tc_path = OUT_DIR / "tc_summary.csv"
    tc_summary.to_csv(tc_path, index=False)
    return


@app.cell
def _show_summary_plot(mo, summary_figs):
    mo.vstack(summary_figs)  # type: ignore
    return


if __name__ == "__main__":
    app.run()
