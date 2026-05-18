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

    $T_c$ from $R(T)$ sweeps on $\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$, two methods:

    - $T_c^{50\%}$ — $T$ at which $R = R_N/2$ (linear interp on smoothed $R$).
    - $T_c^{\max\,\mathrm{d}R/\mathrm{d}T}$ — peak of $\mathrm{d}R/\mathrm{d}T$, parabolic-vertex refined.

    Pair comparisons: heat vs. cool (same $I$, $B=0$) and $B=0$ vs. $B\neq 0$ (same $I$, heating).
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
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "mathtext.fontset": "cm",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })

    from taulab.stats import resolution_sigma, nsigma
    from taulab.fits import odr_fit, fit_functions

    return (
        Line2D, Path, fit_functions, mo, np, nsigma, odr_fit, pd, plt,
        resolution_sigma, savgol_filter,
    )


@app.cell
def _paths(Path):
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data" / "part_a"
    MEAS_DIR = DATA / "measurements"
    OUT_DIR = ROOT / "analysis" / "results" / "part_a"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return MEAS_DIR, OUT_DIR


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

    σ_R propagates through R = V/I (partial-derivative quadrature).
    σ_T combines a symbolic resolution term (CSV display LSD/√12 ≈ 0.3 µK)
    with the dominant sampling-gap term (5th-percentile gap / √12).
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
        R = np.abs(V / I)
        rel = np.sqrt((sigma_V(V) / np.where(V == 0, np.nan, V)) ** 2
                      + (sigma_I(I) / I) ** 2)
        return R * rel

    # Displayed LSD of T in the CSV (≈1 µK from inspecting trailing decimals).
    # Real-meter LSD isn't recoverable since only the converted-to-K column is
    # logged; this term is symbolic — dwarfed by σ_T,samp below — but kept in
    # quadrature for completeness so σ_T = √(σ_res² + σ_samp²) as stated.
    T_LSD = 1e-6
    T_RES = resolution_sigma(T_LSD)

    def sigma_T_sampling(T):
        """Per-run T uncertainty: √((LSD/√12)² + (5th-pct gap/√12)²)."""
        gaps = np.diff(np.sort(np.asarray(T)))
        gaps = gaps[gaps > 1e-4]  # drop float-precision artifacts
        if len(gaps) == 0:
            return float(T_RES)
        sigma_samp = float(np.percentile(gaps, 5) / np.sqrt(12))
        return float(np.sqrt(T_RES**2 + sigma_samp**2))

    return sigma_R, sigma_T_sampling


@app.cell
def _load(MEAS_DIR, pd):
    """Load every frozen per-run CSV.

    The data files are already sorted, deduplicated, and clipped to the common
    80-105 K transition window before they enter this notebook.
    """
    T_MIN, T_MAX = 80.0, 105.0
    _meas = {}
    for _path in sorted(MEAS_DIR.glob("partA_*.csv")):
        _meas[_path.stem] = pd.read_csv(_path)
    measurements = _meas
    return T_MAX, T_MIN, measurements


@app.cell
def _md_tc_methods(mo):
    mo.md(r"""
    ## $T_c$ extraction

    **$R_N$** — median of $R$ in the top decile of $T$.

    **$T_c^{50\%}$** — linear interp through $R_N/2$.
    $$\sigma_{T_c} = \sqrt{\sigma_\text{bracket}^2 + \sigma_\text{baseline}^2 + \sigma_T^2}$$

    - $\sigma_\text{bracket} = \tfrac{1}{2}\,|T_{i+1} - T_i|$ — sample-spacing limit
      on the interpolation bracket.
    - $\sigma_\text{baseline} = \tfrac{1}{2}(\max - \min)$ of $T_c$ recomputed with
      $R_N$ from the top 5 / 10 / 20 % of $T$ — captures sensitivity to where
      the normal state is sampled.

    **$T_c^{\max\,\mathrm{d}R/\mathrm{d}T}$** — SG smooth (order 3) on a uniform $T$
    grid → analytic derivative ($\mathrm{deriv}=1$) → parabolic fit on $\pm 5$ pts
    around the peak; $T_c$ = vertex $-b/(2a)$.
    $$\sigma_{T_c} = \sqrt{\sigma_\text{smooth}^2 + \sigma_\text{peak}^2 + \sigma_T^2}$$

    - $\sigma_\text{smooth} = \tfrac{1}{2}(\max - \min)$ of peak position over SG
      windows 5 / 11 / 21 — smoothing-bandwidth sensitivity.
    - $\sigma_\text{peak}$ — fit uncertainty on the vertex.

    **Temperature uncertainty $\sigma_T$** (per-run, applied to both criteria):
    $$\sigma_T = \sqrt{\sigma_{T,\text{res}}^2 + \sigma_{T,\text{samp}}^2}.$$
    $\sigma_{T,\text{res}} = \mathrm{LSD}_T/\sqrt{12}$ with $\mathrm{LSD}_T = 1\,\mu\mathrm{K}$
    from the CSV display precision (symbolic — the raw sensor reading isn't
    logged, so the meter LSD can't be recovered; this term contributes $\sim0.3\,\mu\mathrm{K}$
    and is dwarfed by the next). $\sigma_{T,\text{samp}} = (\text{5th-pct gap in }T)/\sqrt{12}$
    captures "where was $T$ really between two logged samples"; ~10 mK
    (30 mA / 1 kΩ cool) to ~60 mK (69 mA heat) per run.
    Excluded as systematic (cancels in $\Delta T_c$): Pt-sensor absolute
    calibration, $\sim0.3\,\mathrm{K}$ — shifts every $T_c$ identically.

    **Per-point $\sigma_R$** also propagates from the Rigol DM3058 spec
    ($\sigma = \sqrt{\sigma_\text{acc}^2 + \sigma_\text{res}^2}$, $\sigma_\text{res} = \mathrm{LSD}/\sqrt{12}$)
    and is shown as a shaded band on the $R(T)$ panel. It's tiny in the
    normal state and diverges near $R \to 0$ where $V$ drops to the meter's
    $\sim\!\mu\mathrm{V}$ floor — useful for showing where the data is trustworthy,
    but small enough at mid-transition that it doesn't drive the $T_c$ uncertainty.
    """)
    return


@app.cell
def _rn_helpers(np):
    """A1 — R_N from the median of the highest-T decile."""

    def normal_resistance(df, frac=0.10):
        n = len(df)
        k = max(3, int(round(n * frac)))
        top = df.nlargest(k, "temperature_K")
        return float(np.median(top["resistance_ohm"]))

    def normal_resistance_triplet(df):
        return tuple(normal_resistance(df, f) for f in (0.05, 0.10, 0.20))

    return normal_resistance, normal_resistance_triplet


@app.cell
def _tc50(normal_resistance, normal_resistance_triplet, np, sigma_T_sampling):
    """A2 — Tc(50%) by interpolation through R_N/2.

    σ = quadrature(bracket half-width, baseline spread over R_N at 5/10/20%,
    per-run sampling σ_T).
    """

    def _interp(df, R_N):
        target = R_N / 2.0
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        for i in range(len(R) - 1):
            Ra, Rb = R[i], R[i + 1]
            if (Ra - target) * (Rb - target) <= 0 and Rb != Ra:
                Ta, Tb = T[i], T[i + 1]
                return float(Ta + (target - Ra) * (Tb - Ta) / (Rb - Ra)), float(0.5 * abs(Tb - Ta))
        return float("nan"), float("nan")

    def tc_midpoint(df):
        R_N = normal_resistance(df, 0.10)
        Tc, sigma_bracket = _interp(df, R_N)
        Tcs = []
        for R_N_alt in normal_resistance_triplet(df):
            Tc_alt, _ = _interp(df, R_N_alt)
            if not np.isnan(Tc_alt):
                Tcs.append(Tc_alt)
        sigma_baseline = 0.5 * (max(Tcs) - min(Tcs)) if len(Tcs) > 1 else 0.0
        sigma_T = sigma_T_sampling(df["temperature_K"].to_numpy())
        sigma = float(np.sqrt(sigma_bracket**2 + sigma_baseline**2 + sigma_T**2))
        return Tc, sigma, R_N

    return (tc_midpoint,)


@app.cell
def _tc_inf(fit_functions, np, odr_fit, savgol_filter, sigma_T_sampling):
    """A3 — Tc(max dR/dT) from Savitzky-Golay smoothing + parabolic peak fit.

    The local parabola is fit by orthogonal-distance regression
    (`taulab.fits.odr_fit` with `fit_functions.polynomial`, order 2) so that
    both σ_T (per-run sampling-gap term) and σ_{dR/dT} (high-pass noise
    estimate) enter the vertex uncertainty. The fit returns covariance for
    (A0, A1, A2) in y = A0 + A1·x + A2·x²; the vertex x* = -A1/(2·A2)
    propagates as σ_x* = √(jᵀ Σ j) with j = (0, -1/(2A2), A1/(2A2²)).

    Returns (T_peak, σ, χ²/dof) where σ = quadrature(window-spread,
    ODR vertex error, per-run σ_T) and χ²/dof comes from the ODR fit
    against the high-pass noise (≈1 means the peak is well-described by a
    parabola at the working smoothing).
    """

    def _smoothed_deriv(T, R, window):
        if window % 2 == 0:
            window -= 1
        if window < 5 or window > len(R):
            return None
        T_uni = np.linspace(T.min(), T.max(), len(T))
        R_uni = np.interp(T_uni, T, R)
        dT = (T_uni[-1] - T_uni[0]) / (len(T_uni) - 1)
        dR_uni = savgol_filter(R_uni, window, polyorder=3, deriv=1, delta=dT)
        return np.interp(T, T_uni, dR_uni)

    def tc_inflection(df):
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        if len(T) < 11:
            return float("nan"), float("nan"), float("nan")
        cap = max(5, (len(T) // 3) | 1)

        peaks, dR_main, idx_main = [], None, None
        for w in (5, 11, 21):
            w_use = min(w, cap)
            dR = _smoothed_deriv(T, R, w_use)
            if dR is None:
                continue
            i_pk = int(np.argmax(dR))
            peaks.append(float(T[i_pk]))
            if w == 11 or dR_main is None:
                dR_main, idx_main = dR, i_pk

        sigma_smooth = 0.5 * (max(peaks) - min(peaks)) if len(peaks) > 1 else 0.0

        sigma_T = sigma_T_sampling(T)
        half = 5
        lo, hi = max(0, idx_main - half), min(len(T), idx_main + half + 1)
        T_peak = float(T[idx_main])
        sigma_peak = 0.0
        chi2_dof = float("nan")
        if hi - lo >= 4:
            T_fit, y_fit = T[lo:hi], dR_main[lo:hi]
            # Per-point σ on dR/dT from a high-pass noise estimate:
            # std(dR_main - dR_wide) over the peak window. Signal-independent
            # — captures fluctuations not absorbed by smoothing.
            w_wide = max(2 * w_use + 1, 31)
            w_wide = min(w_wide, len(T) if len(T) % 2 == 1 else len(T) - 1)
            if w_wide % 2 == 0:
                w_wide -= 1
            dR_wide = _smoothed_deriv(T, R, w_wide) if w_wide >= 5 else None
            sigma_y = (
                float(np.std((dR_main - dR_wide)[lo:hi], ddof=1)) or 1e-12
                if dR_wide is not None else float(np.std(y_fit, ddof=1)) or 1e-12
            )
            sx = np.full_like(T_fit, sigma_T, dtype=float)
            sy = np.full_like(y_fit,  sigma_y, dtype=float)
            # taulab.fits.polynomial: y = A0 + A1·x + A2·x²
            res = odr_fit(
                fit_functions.polynomial, None,
                T_fit, sx, y_fit, sy,
                param_names=["A0", "A1", "A2"],
            )
            A0, A1, A2 = res.params
            cov = res.cov if res.cov is not None else np.zeros((3, 3))
            T_peak = float(-A1 / (2 * A2))
            # Jacobian of -A1/(2 A2) wrt (A0, A1, A2):
            j = np.array([0.0, -1.0 / (2 * A2), A1 / (2 * A2**2)])
            sigma_peak = float(np.sqrt(max(j @ cov @ j, 0.0)))
            chi2_dof = float(res.redchi)

        sigma = float(np.sqrt(sigma_smooth**2 + sigma_peak**2 + sigma_T**2))
        return T_peak, sigma, chi2_dof

    return (tc_inflection,)


@app.cell
def _per_run(
    measurements,
    np,
    pd,
    sigma_R,
    sigma_T_sampling,
    tc_inflection,
    tc_midpoint,
):
    """Per-measurement Tc table with uncertainties + data-quality columns.

    Beyond the two Tc's: `chi2_dof_dRdT_peak` (parabolic-fit goodness against
    a high-pass noise estimate — ≈1 when the peak is well-described by a
    parabola at the working smoothing window, ≫1 when it isn't), and
    `sigma_R_rel_at_RN` / `sigma_R_rel_at_RN_half` (relative σ_R at the normal
    state vs. mid-transition — shows where the data quality degrades).
    """
    def _row_for(_mid, _df):
        _Tc50, _sig50, _R_N = tc_midpoint(_df)
        _Tc_inf, _sig_inf, _chi2 = tc_inflection(_df)
        _meta = _df.iloc[0]
        _T = _df["temperature_K"].to_numpy()
        _R = _df["resistance_ohm"].to_numpy()
        _V = _df["voltage_V"].to_numpy()
        _I = _df["current_A"].to_numpy()

        def _crossing(_target):
            for _i in range(len(_R) - 1):
                if (_R[_i] - _target) * (_R[_i + 1] - _target) <= 0 and _R[_i + 1] != _R[_i]:
                    return float(
                        _T[_i]
                        + (_target - _R[_i]) * (_T[_i + 1] - _T[_i]) / (_R[_i + 1] - _R[_i])
                    )
            return float("nan")

        _T_on = _crossing(0.90 * _R_N)
        _T_zr = _crossing(0.10 * _R_N)
        _w = _T_on - _T_zr if not (np.isnan(_T_on) or np.isnan(_T_zr)) else float("nan")

        # Data-quality: σ_R/R near the normal state (top decile) and near
        # mid-transition (bracket of R_N/2). Median over the relevant subset.
        _sR_rel = sigma_R(_V, _I) / np.maximum(_R, 1e-12)
        _top_mask = _T >= np.percentile(_T, 90)
        _mid_mask = (_R >= 0.4 * _R_N) & (_R <= 0.6 * _R_N)
        _sR_rel_RN = float(np.median(_sR_rel[_top_mask])) if _top_mask.any() else float("nan")
        _sR_rel_mid = float(np.median(_sR_rel[_mid_mask])) if _mid_mask.any() else float("nan")

        return dict(
            measurement_id=_mid,
            sample_current_mA_nominal=float(_meta["sample_current_mA_nominal"]),
            series_resistor=_meta["series_resistor"],
            direction=_meta["direction"],
            field_condition=_meta["field_condition"],
            normal_resistance_ohm=_R_N,
            tc_midpoint_K=_Tc50,
            tc_50_err_K=_sig50,
            tc_inflection_K=_Tc_inf,
            tc_dRdT_err_K=_sig_inf,
            chi2_dof_dRdT_peak=_chi2,
            sigma_T_sampling_K=sigma_T_sampling(_T),
            sigma_R_rel_at_RN=_sR_rel_RN,
            sigma_R_rel_at_RN_half=_sR_rel_mid,
            tc_onset_K=_T_on,
            tc_zero_K=_T_zr,
            delta_Tc_width_K=_w,
            points_used_for_tc=len(_df),
        )

    tc_summary = (
        pd.DataFrame([_row_for(_mid, _df) for _mid, _df in measurements.items()])
        .sort_values("measurement_id")
        .reset_index(drop=True)
    )
    return (tc_summary,)


@app.cell
def _trace_helpers(normal_resistance, np, savgol_filter, sigma_R, sigma_T_sampling):
    """Smoothed R(T) + dR/dT trace, in the style of `plot_heat_cool_overlay.py`.

    Returns a dict per measurement: smoothed $R$, analytic $\\mathrm{d}R/\\mathrm{d}T$,
    both Tc values, the parabola at the peak, and per-point σ_R (used as an
    error band on the upper R(T) panel).
    """

    def trace(df, window=21, parabola_half=5):
        T = df["temperature_K"].to_numpy()
        R = df["resistance_ohm"].to_numpy()
        V = df["voltage_V"].to_numpy()
        I = df["current_A"].to_numpy()
        R_err = sigma_R(V, I)
        T_err = sigma_T_sampling(T)
        w = min(window, len(R) - (1 - len(R) % 2))
        if w % 2 == 0:
            w -= 1
        w = max(w, 5)
        # Smoothed R on the native grid (for the upper plot).
        R_s = savgol_filter(R, window_length=w, polyorder=3)
        # dR/dT via SG's analytic derivative on a uniform resample, then
        # interpolated back to native T — much smoother than `np.gradient`
        # of the smoothed series (verified: 6×–740× less jitter).
        T_uni = np.linspace(T.min(), T.max(), len(T))
        R_uni = np.interp(T_uni, T, R)
        dT_uni = (T_uni[-1] - T_uni[0]) / (len(T_uni) - 1)
        dR_uni = savgol_filter(R_uni, w, polyorder=3, deriv=1, delta=dT_uni)
        dR_dT = np.interp(T, T_uni, dR_uni)

        # Parabolic refinement of the dR/dT peak — same procedure as A3 in
        # _tc_inf, exposed here so the plot can overlay the fit and so the
        # vline at Tc(max dR/dT) matches the value in tc_summary.
        i_pk = int(np.argmax(dR_dT))
        lo, hi = max(0, i_pk - parabola_half), min(len(T), i_pk + parabola_half + 1)
        Tc_drdt = float(T[i_pk])
        para_T, para_y = None, None
        if hi - lo >= 4:
            T_fit, y_fit = T[lo:hi], dR_dT[lo:hi]
            a, b, c = np.polyfit(T_fit, y_fit, 2)
            if a < 0:  # only meaningful if the parabola has a maximum
                Tc_drdt = float(-b / (2 * a))
                para_T = np.linspace(T_fit.min(), T_fit.max(), 100)
                para_y = a * para_T**2 + b * para_T + c

        # Tc(50%) by interpolation through R_N/2 on the smoothed curve so the
        # crossing isn't biased by single-point noise.
        R_N = normal_resistance(df, 0.10)
        target = R_N / 2.0
        Tc50 = float("nan")
        for i in range(len(R_s) - 1):
            if (R_s[i] - target) * (R_s[i + 1] - target) <= 0 and R_s[i + 1] != R_s[i]:
                Tc50 = float(
                    T[i] + (target - R_s[i]) * (T[i + 1] - T[i]) / (R_s[i + 1] - R_s[i])
                )
                break

        return dict(
            T=T, R=R, R_err=R_err, T_err=T_err,
            R_smoothed=R_s, dR_dT=dR_dT,
            Tc=Tc_drdt, Tc50=Tc50, R_N=R_N,
            parabola_T=para_T, parabola_dR_dT=para_y,
        )

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

    The bottom-axis legend is reserved for the Tc-criterion line-style key
    (solid = 50% midpoint, dashed = max dR/dT) so it never collides with
    the per-trace legend on the top axis.
    """

    _CRIT_HANDLES = [
        Line2D([0], [0], color="k", ls="-.", lw=1.0,
               label=r"$T_c^{\,50\%}$"),
        Line2D([0], [0], color="k", ls="--", lw=1.0,
               label=r"$T_c^{\,\max\,\mathrm{d}R/\mathrm{d}T}$"),
        Line2D([0], [0], color="k", ls="-",  lw=1.8, marker="v",
               markerfacecolor="k", markeredgecolor="white", markersize=6,
               label=r"parabolic fit"),
    ]

    def build(_title=None):
        fig, (ax_r, ax_d) = plt.subplots(
            2, 1, figsize=(8.6, 6.4), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.07},
        )
        ax_r.set_title(
            r"Resistance vs Temperature for "
            r"$\mathrm{Bi_2Sr_2Ca_2Cu_3O_{10+x}}$",
            pad=10,
        )
        ax_r.set_ylabel(r"$R$  (m$\Omega$)")
        ax_d.set_xlabel(r"Temperature  $T$  (K)")
        ax_d.set_ylabel(r"$\mathrm{d}R/\mathrm{d}T$  (m$\Omega$/K)")
        for ax in (ax_r, ax_d):
            ax.grid(True, alpha=0.15, lw=0.5)
            ax.tick_params(direction="in", top=True, right=True)
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

    def draw(ax_r, ax_d, df, tr, color, marker, label):
        # PRL-style: open markers in series color, capless error bars,
        # smoothed line in the same saturated color, faint σ_R band kept
        # as a low-alpha confidence ribbon under the smoothed curve.
        ax_r.fill_between(
            tr["T"],
            (tr["R"] - tr["R_err"]) * 1e3,
            (tr["R"] + tr["R_err"]) * 1e3,
            color=color, alpha=0.20, linewidth=0, zorder=1,
        )
        ax_r.errorbar(
            tr["T"], tr["R"] * 1e3,
            xerr=tr["T_err"], yerr=tr["R_err"] * 1e3,
            fmt=marker, ms=4.0, mfc="none", mec=color, mew=0.8,
            ecolor=color, elinewidth=0.5, capsize=0,
            alpha=0.85, zorder=2,
        )
        ax_r.plot(
            tr["T"], tr["R_smoothed"] * 1e3, lw=1.6, color=color, zorder=3,
            label=(
                rf"{label}:  "
                rf"$T_c^{{\,50\%}}\!=\!{tr['Tc50']:.2f}\,$K,  "
                rf"$T_c^{{\,\max}}\!=\!{tr['Tc']:.2f}\,$K"
            ),
        )
        # Tc reference lines: thin, low alpha — guides, not data.
        # Two distinct dash patterns to tell the methods apart at small size.
        ax_r.axvline(tr["Tc50"], color=color, ls="-.", lw=0.9, alpha=0.65)
        ax_r.axvline(tr["Tc"],   color=color, ls="--", lw=0.9, alpha=0.65)
        ax_d.plot(tr["T"], tr["dR_dT"] * 1e3, color=color, lw=1.4, alpha=0.75)
        if tr["parabola_T"] is not None:
            ax_d.plot(
                tr["parabola_T"], tr["parabola_dR_dT"] * 1e3,
                color=color, lw=1.8, ls="-", alpha=1.0, zorder=3,
            )
            ax_d.plot(
                [tr["Tc"]],
                [(tr["parabola_dR_dT"] * 1e3).max()],
                marker="v", ms=6, color=color, mec="white", mew=0.8, zorder=4,
            )
        ax_d.axvline(tr["Tc50"], color=color, ls="-.", lw=0.9, alpha=0.65)
        ax_d.axvline(tr["Tc"],   color=color, ls="--", lw=0.9, alpha=0.65)

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
def _heat_cool_plots(
    build,
    clip,
    draw,
    fmt_I,
    fmt_R,
    heat_cool_groups,
    intersection,
    legend_with_info,
    measurements,
    mo,
    trace,
):
    """One figure per heat/cool pair (no-magnet runs only)."""
    _figs = []
    for (_I, _R), _ids in sorted(heat_cool_groups.items()):
        _pair = [measurements[m] for m in _ids]
        _T_lo, _T_hi = intersection(_pair)
        _info = rf"${fmt_I(_I)}$,  ${fmt_R(_R)}$,  $B = 0$"
        _fig, _ax_r, _ax_d = build("heating vs cooling")
        _styles = {
            "cool": ("#1f77b4", "s", "cooling"),
            "heat": ("#d62728", "o", "heating"),
        }
        for _df_full in _pair:
            _df = clip(_df_full, _T_lo, _T_hi)
            _dir = _df.iloc[0]["direction"]
            _color, _marker, _label = _styles[_dir]
            draw(_ax_r, _ax_d, _df, trace(_df), _color, _marker, _label)
        for _ax in (_ax_r, _ax_d):
            _ax.set_xlim(_T_lo - 0.5, _T_hi + 0.5)
        legend_with_info(_ax_r, _info)
        _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _magnet_plots(
    build,
    clip,
    draw,
    fmt_I,
    fmt_R,
    fmt_dir,
    intersection,
    legend_with_info,
    magnet_groups,
    measurements,
    mo,
    trace,
):
    """One figure per magnet-vs-no-magnet pair."""
    _figs = []
    for (_I, _R, _d), _ids in sorted(magnet_groups.items()):
        _pair = [measurements[m] for m in _ids]
        _T_lo, _T_hi = intersection(_pair)
        _info = rf"${fmt_I(_I)}$,  ${fmt_R(_R)}$,  {fmt_dir(_d)}"
        _fig, _ax_r, _ax_d = build("applied field vs zero field")
        # Distinct palette from heat/cool (red/blue) so the pair isn't
        # mistaken for a temperature-direction comparison.
        _styles = {
            "no_magnet": ("#2ca02c", "s", r"$B = 0$"),       # green
            "magnet":    ("#9467bd", "o", r"$B \neq 0$"),    # purple
        }
        for _df_full in _pair:
            _df = clip(_df_full, _T_lo, _T_hi)
            _field = _df.iloc[0]["field_condition"]
            _color, _marker, _label = _styles[_field]
            draw(_ax_r, _ax_d, _df, trace(_df), _color, _marker, _label)
        for _ax in (_ax_r, _ax_d):
            _ax.set_xlim(_T_lo - 0.5, _T_hi + 0.5)
        legend_with_info(_ax_r, _info)
        _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _solo_plots(
    build,
    draw,
    fmt_I,
    fmt_R,
    fmt_dir,
    fmt_field,
    legend_with_info,
    measurements,
    mo,
    solo_ids,
    trace,
):
    """Same two-panel style for runs without a pair — one figure each.

    Colors match the pair plots: heating = red ●, cooling = blue ■.
    """
    _styles = {
        "cool": ("#1f77b4", "s"),
        "heat": ("#d62728", "o"),
    }
    _figs = []
    for _mid in sorted(solo_ids):
        _df = measurements[_mid]
        _meta = _df.iloc[0]
        _dir = _meta["direction"]
        _info = (
            rf"${fmt_I(_meta['sample_current_mA_nominal'])}$,  "
            rf"${fmt_R(_meta['series_resistor'])}$,  "
            rf"{fmt_dir(_dir)},  ${fmt_field(_meta['field_condition'])}$"
        )
        _fig, _ax_r, _ax_d = build("single sweep")
        _color, _marker = _styles[_dir]
        draw(_ax_r, _ax_d, _df, trace(_df), _color, _marker, fmt_dir(_dir))
        legend_with_info(_ax_r, _info)
        _fig.tight_layout()
        _figs.append(_fig)
    mo.vstack(_figs) if _figs else None
    return


@app.cell
def _reference():
    """Reference T_c to compare every measurement against, via `taulab.stats.nsigma`.

    EDIT these once you have the accepted/literature value (e.g. the
    Bi-2223 bulk T_c ≈ 110 K, or whatever the lab brief specifies).
    `T_REF_SIGMA = 0.0` makes it an exact reference; set a nonzero value
    to fold the reference's own uncertainty into the discrepancy.
    """
    # Eltsev et al., arXiv:0909.1628v3 — high-quality Bi-2223 single crystal,
    # sample #4 (single-step transition): T_c mid-point ≈ 108 K, transition
    # width (10–90%) ≈ 2 K. No explicit σ; half the transition width is
    # taken as a conservative reference uncertainty.
    T_REF       = 108.0
    T_REF_SIGMA = 1.0
    return T_REF, T_REF_SIGMA


@app.cell
def _md_final(mo):
    mo.md(r"""
    ## Summary

    One row per run with both $T_c$ estimates and their $N_\sigma$ discrepancy
    against a reference value (see the `_reference` cell to set it):

    $$N_\sigma \;=\; \frac{|T_c - T_\text{ref}|}{\sqrt{\sigma_{T_c}^2 + \sigma_\text{ref}^2}}.$$

    Written to `analysis/results/part_a/tc_summary.csv`.
    """)
    return


@app.cell
def _final_table(T_REF, T_REF_SIGMA, nsigma, pd, tc_summary):
    """Per-run summary: metadata + both Tc methods + N_σ vs reference.

    Tc columns are formatted as "val ± σ (rel%)" with rel = σ/|val|·100.
    """
    def _fmt(val, err):
        rel = (err / abs(val) * 100.0) if val else float("nan")
        return f"{val:.2f} ± {err:.2f} ({rel:.2f}%)"

    _rows = []
    for _, _r in tc_summary.iterrows():
        _ns_50 = nsigma(
            (_r["tc_midpoint_K"],   _r["tc_50_err_K"]),
            (T_REF, T_REF_SIGMA),
        )
        _ns_dr = nsigma(
            (_r["tc_inflection_K"], _r["tc_dRdT_err_K"]),
            (T_REF, T_REF_SIGMA),
        )
        _rows.append({
            "I (mA)":             int(_r["sample_current_mA_nominal"]),
            "R_s":                _r["series_resistor"],
            "sweep":              _r["direction"],
            "field":              _r["field_condition"],
            "Tc(max dR/dT) [K]":  _fmt(_r["tc_inflection_K"], _r["tc_dRdT_err_K"]),
            "Tc(50%) [K]":        _fmt(_r["tc_midpoint_K"],   _r["tc_50_err_K"]),
            "N_σ (max dR/dT)":    round(_ns_dr, 2),
            "N_σ (50%)":          round(_ns_50, 2),
        })
    final_table = pd.DataFrame(_rows)
    return (final_table,)


@app.cell
def _show_final(final_table):
    final_table
    return


@app.cell
def _write(OUT_DIR, final_table):
    tc_path = OUT_DIR / "tc_summary.csv"
    final_table.to_csv(tc_path, index=False)
    print(f"wrote {tc_path}")
    return


if __name__ == "__main__":
    app.run()
