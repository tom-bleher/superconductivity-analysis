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
    # Superconductivity — Part B

    Magnetic response of the superconducting disk from Hall-probe data:

    - Empty-coil calibration: $B_\mathrm{coil} = a_1 I + a_0$.
    - Disk measurement: convert coil current to applied field $H$ and Hall voltage to local $B$.
    - Fit the low-field Meissner-like region and the high-field penetrated region.
    - Define $H^*$ as the intersection of those two fitted lines, then estimate
      $$J_c = \frac{H^*/\mu_0}{d}$$
      for disk thickness $d = 0.5\,\mu\mathrm{m}$.
    """)
    return


@app.cell
def _imports():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from taulab.fits import odr_fit, fit_functions
    from taulab.stats import nsigma

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "mathtext.fontset": "cm",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    })
    return Path, fit_functions, mo, np, nsigma, odr_fit, pd, plt


@app.cell
def _paths(Path):
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data" / "part_b"
    OUT_DIR = ROOT / "results" / "part_b"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DATA, OUT_DIR


@app.cell
def _constants(np):
    HALL_T_PER_V = 20.45
    HALL_FACTOR_REL_ERR = 0.0
    LOW_FIELD_MAX_MT = 0.25
    HIGH_FIELD_MIN_MT = 1.5
    DISK_THICKNESS_M = 0.5e-6
    DISK_THICKNESS_REL_ERR = 0.05
    DISK_THICKNESS_ERR_M = DISK_THICKNESS_M * DISK_THICKNESS_REL_ERR
    MU0 = 4 * np.pi * 1e-7
    return (
        DISK_THICKNESS_ERR_M,
        DISK_THICKNESS_M,
        HALL_FACTOR_REL_ERR,
        HALL_T_PER_V,
        HIGH_FIELD_MIN_MT,
        LOW_FIELD_MAX_MT,
        MU0,
    )


@app.cell
def _load(DATA, pd):
    coil_calibration = pd.read_csv(DATA / "coil_calibration.csv")
    disk_bh_raw = pd.read_csv(DATA / "disk_BH.csv")
    return coil_calibration, disk_bh_raw


@app.cell
def _md_calibration(mo):
    mo.md(r"""
    ## Empty-coil calibration

    The raw Hall voltage was converted upstream using the Hall-probe factor
    $20.45\,\mathrm{T/V}$. The calibration fit below is repeated from the curated
    CSV, so the notebook is independent of the original workbooks.

    The plotted error bars use the same meter model as Part A: accuracy and
    display resolution added in quadrature. The Hall-voltage channel is treated
    as a 100 mV DC-voltage range; the coil-current channel reaches 1.5 A, so a
    2 A DC-current range is used for the horizontal uncertainty. The current
    uncertainty is also propagated into the vertical calibration uncertainty via
    $$\sigma_{B,I}=a_1\sigma_I,$$
    and combined with the Hall-voltage contribution.

    The calibration line itself is fitted with orthogonal distance regression
    (ODR), so both horizontal and vertical uncertainties enter the fit.
    """)
    return


@app.cell
def _instrument_uncertainties(HALL_T_PER_V, np):
    """Per-point Part B readout uncertainties.

    Same convention as Part A: σ = sqrt(σ_acc² + σ_res²), with σ_res = LSD/√12.
    The Hall voltage is measured near the sub-mV scale on a 100 mV range; coil
    current reaches 1.5 A, so use the 2 A current range for the current readout.
    """
    HALL_V_RANGE, HALL_V_LSD = 0.1, 1e-6
    COIL_I_RANGE, COIL_I_LSD = 2.0, 1e-5
    _V_RES = HALL_V_LSD / np.sqrt(12)
    _I_RES = COIL_I_LSD / np.sqrt(12)

    def sigma_hall_voltage_V(V):
        V = np.asarray(V)
        return np.sqrt((0.00015 * np.abs(V) + 0.00004 * HALL_V_RANGE) ** 2 + _V_RES**2)

    def sigma_raw_B_mT(V):
        return sigma_hall_voltage_V(V) * HALL_T_PER_V * 1000

    def sigma_coil_current_A(I):
        I = np.asarray(I)
        return np.sqrt((0.0025 * np.abs(I) + 0.00020 * COIL_I_RANGE) ** 2 + _I_RES**2)

    return sigma_coil_current_A, sigma_raw_B_mT


@app.cell
def _fit_helpers(fit_functions, np, odr_fit, pd):
    """ODR line fit via taulab — y = A0 + A1·x (taulab polynomial convention).

    Re-mapped to the (slope, intercept) names used downstream:
      slope = A1, intercept = A0.
    Covariance is reordered to (slope, intercept) basis to keep all
    Jacobian propagation in this script consistent.
    """

    def linear_fit(x, y, xerr=None, yerr=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = None if xerr is None else np.asarray(xerr, dtype=float)
        sy = None if yerr is None else np.asarray(yerr, dtype=float)
        # Synthesize a tiny σ floor if a side wasn't supplied — taulab's
        # ODR requires both. 1e-12 in data units is below any real noise.
        if sx is None: sx = np.full_like(x, 1e-12)
        if sy is None: sy = np.full_like(y, 1e-12)
        res = odr_fit(
            fit_functions.linear, None,  # taulab auto-seeds from polyfit
            x, sx, y, sy,
            param_names=["A0", "A1"],
        )
        A0, A1 = res.params
        sA0, sA1 = res.errors
        # Reorder cov from (A0, A1) to (slope, intercept) = (A1, A0).
        cov_AA = res.cov if res.cov is not None else np.zeros((2, 2))
        cov = np.array([[cov_AA[1, 1], cov_AA[1, 0]],
                        [cov_AA[0, 1], cov_AA[0, 0]]])
        return dict(
            slope=float(A1),
            intercept=float(A0),
            slope_err=float(sA1),
            intercept_err=float(sA0),
            covariance=cov,
            chi2_dof=float(res.redchi),
            method="ODR",
            n=len(x),
        )

    def fit_row(name, fit):
        return pd.Series({
            "fit": name,
            "slope": fit["slope"],
            "slope_err": fit["slope_err"],
            "intercept_mT": fit["intercept"],
            "intercept_err_mT": fit["intercept_err"],
            "chi2_dof": fit["chi2_dof"],
            "method": fit["method"],
            "points": fit["n"],
        })

    return fit_row, linear_fit


@app.cell
def _calibration_fit(
    coil_calibration,
    linear_fit,
    np,
    sigma_coil_current_A,
    sigma_raw_B_mT,
):
    cal = coil_calibration.copy()
    cal["coil_current_err_A"] = sigma_coil_current_A(cal["coil_current_A"])
    cal["raw_B_err_mT"] = sigma_raw_B_mT(cal["hall_voltage_V"])
    cal_fit = linear_fit(
        cal["coil_current_A"],
        cal["raw_B_mT"],
        xerr=cal["coil_current_err_A"],
        yerr=cal["raw_B_err_mT"],
    )
    cal["current_equiv_B_err_mT"] = abs(cal_fit["slope"]) * cal["coil_current_err_A"]
    cal["combined_B_err_mT"] = np.sqrt(
        cal["raw_B_err_mT"]**2 + cal["current_equiv_B_err_mT"]**2
    )
    cal["fit_B_mT"] = cal_fit["slope"] * cal["coil_current_A"] + cal_fit["intercept"]
    cal["residual_B_mT"] = cal["raw_B_mT"] - cal["fit_B_mT"]
    return cal, cal_fit


@app.cell
def _disk_uncertainties(
    cal_fit,
    disk_bh_raw,
    np,
    sigma_coil_current_A,
    sigma_raw_B_mT,
):
    disk_bh = disk_bh_raw.copy()
    disk_bh["H_mT"] = cal_fit["slope"] * disk_bh["coil_current_A"]
    disk_bh["B_mT"] = disk_bh["raw_B_mT"] - cal_fit["intercept"]
    disk_bh["coil_current_err_A"] = sigma_coil_current_A(disk_bh["coil_current_A"])
    disk_bh["raw_B_err_mT"] = sigma_raw_B_mT(disk_bh["hall_voltage_V"])
    # Independent per-point uncertainty for ODR: current readout only.
    # Calibration-slope uncertainty is common to all H values, so it is kept as
    # a systematic term for the final H* budget instead of being used as if it
    # were uncorrelated point scatter.
    disk_bh["H_fit_err_mT"] = abs(cal_fit["slope"]) * disk_bh["coil_current_err_A"]
    disk_bh["H_calibration_err_mT"] = abs(disk_bh["coil_current_A"]) * cal_fit["slope_err"]
    disk_bh["H_err_mT"] = np.sqrt(
        disk_bh["H_fit_err_mT"] ** 2 + disk_bh["H_calibration_err_mT"] ** 2
    )
    # Same separation for B: Hall-voltage readout is pointwise; the calibration
    # intercept is a common offset (shown in total error bars, not used as ODR
    # point scatter for determining H*).
    disk_bh["B_fit_err_mT"] = disk_bh["raw_B_err_mT"]
    disk_bh["B_offset_err_mT"] = cal_fit["intercept_err"]
    disk_bh["B_err_mT"] = np.sqrt(
        disk_bh["B_fit_err_mT"] ** 2 + disk_bh["B_offset_err_mT"] ** 2
    )
    return (disk_bh,)


@app.cell
def _plot_calibration(cal, cal_fit, np, plt):
    fig_cal, (ax_cal, ax_res) = plt.subplots(
        2, 1, figsize=(8.0, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    _x = cal["coil_current_A"].to_numpy()
    _xerr = cal["coil_current_err_A"].to_numpy()
    _yerr = cal["combined_B_err_mT"].to_numpy()
    _xx = np.linspace(_x.min(), _x.max(), 300)
    _color = "#1f77b4"
    ax_cal.errorbar(
        _x, cal["raw_B_mT"], xerr=_xerr, yerr=_yerr,
        fmt="o", ms=4.0, mfc="none", mec=_color, mew=0.8,
        ecolor=_color, elinewidth=0.5, capsize=0,
        alpha=0.85, zorder=2, label="data",
    )
    ax_cal.plot(
        _xx,
        cal_fit["slope"] * _xx + cal_fit["intercept"],
        color="#d62728", lw=1.6, zorder=3,
        label=rf"$B = ({cal_fit['slope']:.4f})I {cal_fit['intercept']:+.4f}$ mT",
    )
    ax_cal.set_title("Empty-coil calibration")
    ax_cal.set_ylabel(r"$B_\mathrm{coil}$ (mT)")
    ax_cal.legend(frameon=False)
    ax_res.axhline(0, color="0.4", lw=0.6, ls=(0, (1, 2)))
    ax_res.errorbar(
        _x, cal["residual_B_mT"], xerr=_xerr, yerr=_yerr,
        fmt="o", ms=4.0, mfc="none", mec=_color, mew=0.8,
        ecolor=_color, elinewidth=0.5, capsize=0,
        alpha=0.85, zorder=2,
    )
    ax_res.set_xlabel("Coil current (A)")
    ax_res.set_ylabel("res. (mT)")
    for _ax in (ax_cal, ax_res):
        _ax.grid(True, alpha=0.15, lw=0.5)
        _ax.tick_params(direction="in", top=True, right=True)
    fig_cal.tight_layout()
    fig_cal
    return


@app.cell
def _md_bh(mo):
    mo.md(r"""
    ## Disk $B(H)$ curve

    The chosen fit windows match the cleaned-data export:

    - Low-field fit: $0 \le H \le 0.25\,\mathrm{mT}$.
    - High-field fit: $H \ge 1.5\,\mathrm{mT}$.

    These windows isolate the initial shielding response and the approximately
    penetrated high-field response.
    """)
    return


@app.cell
def _md_uncertainties(mo):
    mo.md(r"""
    ## $H^*$ and $J_c$ uncertainty

    **Hall conversion.** The handout gives the Hall factor
    $C_H = 20.45\,\mathrm{T/V}$, so
    $$B_\mathrm{raw} = C_H V_H.$$
    No uncertainty for $C_H$ is specified, so it is treated as exact in the
    numeric error budget.

    **Empty-coil calibration.** The empty-coil fit gives
    $$H = a_1 I + a_0.$$
    The calibration-slope uncertainty contributes
    $$\sigma_{H^*,\mathrm{cal}} = H^*\,\frac{\sigma_{a_1}}{a_1}.$$

    **Per-point disk uncertainties.** For the $B(H)$ plots,
    the independent readout uncertainties used by the ODR fits are
    $$\sigma_{H,\mathrm{fit}}=a_1\sigma_I,\qquad
    \sigma_{B,\mathrm{fit}}=\sigma_{B,\mathrm{Hall}}.$$
    The plotted total error bars also show the common calibration contributions,
    $$\sigma_{H,\mathrm{plot}}=\sqrt{(a_1\sigma_I)^2+(I\sigma_{a_1})^2},$$
    $$\sigma_{B,\mathrm{plot}}=\sqrt{\sigma_{B,\mathrm{Hall}}^2+
    \sigma_{a_0}^2}.$$
    These common calibration terms are not treated as independent point scatter.

    **Intersection fit.** For the two fitted lines
    $$B_\mathrm{low}=m_1H+b_1,\qquad B_\mathrm{high}=m_2H+b_2,$$
    the penetration field is
    $$H^*=\frac{b_1-b_2}{m_2-m_1}.$$
    $\sigma_{H^*,\mathrm{fit}}$ is propagated from the slope/intercept covariance
    matrices of the two ODR line fits, so the horizontal $\sigma_H$ and vertical
    $\sigma_B$ both enter the fitted parameters.

    **Fit-window sensitivity.** As in Part A's baseline/smoothing sensitivity,
    repeat the extraction for several reasonable low/high fit windows and take
    $$\sigma_{H^*,\mathrm{window}} = \frac{1}{2}\left(\max H^* - \min H^*\right).$$

    **Total $H^*$ uncertainty.**
    $$\sigma_{H^*}=\sqrt{\sigma_{H^*,\mathrm{fit}}^2
    +\sigma_{H^*,\mathrm{window}}^2+\sigma_{H^*,\mathrm{cal}}^2}.$$

    The Hall factor is a common scale factor for both $H$ and $B$. The handout
    gives $C_H=20.45\,\mathrm{T/V}$ but no tolerance, so its relative
    uncertainty is set to zero unless a spec is supplied.

    **Critical-current density.** Using
    $$J_c=\frac{H^*/\mu_0}{d},$$
    $$\sigma_{J_c}=J_c\sqrt{\left(\frac{\sigma_{H^*}}{H^*}\right)^2
    +\left(\frac{\sigma_d}{d}\right)^2}.$$
    The handout gives no tolerance for $d$. To make the budget explicit, we use
    a relative width/thickness uncertainty of $5\%$:
    $$\sigma_d = 0.05d.$$
    """)
    return


@app.cell
def _bh_fits(HIGH_FIELD_MIN_MT, LOW_FIELD_MAX_MT, disk_bh, linear_fit):
    low_region = disk_bh[disk_bh["H_mT"].between(0, LOW_FIELD_MAX_MT)].copy()
    high_region = disk_bh[disk_bh["H_mT"] >= HIGH_FIELD_MIN_MT].copy()
    low_fit = linear_fit(
        low_region["H_mT"],
        low_region["B_mT"],
        xerr=low_region["H_fit_err_mT"],
        yerr=low_region["B_fit_err_mT"],
    )
    high_fit = linear_fit(
        high_region["H_mT"],
        high_region["B_mT"],
        xerr=high_region["H_fit_err_mT"],
        yerr=high_region["B_fit_err_mT"],
    )
    return high_fit, high_region, low_fit, low_region


@app.cell
def _intersection(DISK_THICKNESS_M, MU0, high_fit, low_fit, np):
    m1, b1 = low_fit["slope"], low_fit["intercept"]
    m2, b2 = high_fit["slope"], high_fit["intercept"]
    den = m2 - m1
    H_star_mT = float((b1 - b2) / den)
    B_star_mT = float(m1 * H_star_mT + b1)

    # First-order propagation, keeping each line's slope/intercept covariance.
    dH_dm1 = (b1 - b2) / den**2
    dH_db1 = 1.0 / den
    dH_dm2 = -(b1 - b2) / den**2
    dH_db2 = -1.0 / den
    grad_low = np.array([dH_dm1, dH_db1])
    grad_high = np.array([dH_dm2, dH_db2])
    H_star_err_mT = float(np.sqrt(
        grad_low @ low_fit["covariance"] @ grad_low
        + grad_high @ high_fit["covariance"] @ grad_high
    ))

    Jc_A_per_m2 = float((H_star_mT * 1e-3 / MU0) / DISK_THICKNESS_M)
    Jc_err_A_per_m2 = float((H_star_err_mT * 1e-3 / MU0) / DISK_THICKNESS_M)
    return B_star_mT, H_star_err_mT, H_star_mT, Jc_A_per_m2, Jc_err_A_per_m2


@app.cell
def _window_uncertainty(DISK_THICKNESS_M, MU0, disk_bh, linear_fit, pd):
    """Analysis-window sensitivity.

    The handout gives the Hall conversion factor but not its uncertainty. After
    readout errors are included, the dominant quantifiable analysis choice is
    where the two linear B(H) regions are cut, so vary those cuts and use half
    the resulting range as a systematic term.
    """
    _rows = []
    for _low_max in (0.20, 0.25, 0.30, 0.35):
        for _high_min in (1.25, 1.50, 1.75, 2.00):
            _low = disk_bh[disk_bh["H_mT"].between(0, _low_max)]
            _high = disk_bh[disk_bh["H_mT"] >= _high_min]
            if len(_low) < 3 or len(_high) < 3:
                continue
            _lf = linear_fit(
                _low["H_mT"],
                _low["B_mT"],
                xerr=_low["H_fit_err_mT"],
                yerr=_low["B_fit_err_mT"],
            )
            _hf = linear_fit(
                _high["H_mT"],
                _high["B_mT"],
                xerr=_high["H_fit_err_mT"],
                yerr=_high["B_fit_err_mT"],
            )
            _den = _hf["slope"] - _lf["slope"]
            _h_star = (_lf["intercept"] - _hf["intercept"]) / _den
            _jc = (_h_star * 1e-3 / MU0) / DISK_THICKNESS_M
            _rows.append({
                "low_fit_H_max_mT": _low_max,
                "high_fit_H_min_mT": _high_min,
                "H_star_mT": _h_star,
                "Jc_A_per_m2": _jc,
                "low_points": len(_low),
                "high_points": len(_high),
            })

    window_sensitivity = pd.DataFrame(_rows)
    H_star_window_err_mT = float(
        0.5 * (window_sensitivity["H_star_mT"].max() - window_sensitivity["H_star_mT"].min())
    )
    Jc_window_err_A_per_m2 = float(
        0.5 * (window_sensitivity["Jc_A_per_m2"].max() - window_sensitivity["Jc_A_per_m2"].min())
    )
    return H_star_window_err_mT, Jc_window_err_A_per_m2, window_sensitivity


@app.cell
def _plot_bh_full(
    HIGH_FIELD_MIN_MT,
    LOW_FIELD_MAX_MT,
    disk_bh,
    high_region,
    low_region,
    plt,
):
    fig_bh, _ax_bh = plt.subplots(figsize=(8.0, 5.2))
    _ax_bh.errorbar(
        disk_bh["H_mT"], disk_bh["B_mT"],
        xerr=disk_bh["H_err_mT"], yerr=disk_bh["B_err_mT"],
        fmt="o", ms=3.5, mfc="none", mec="0.35", mew=0.7,
        ecolor="0.5", elinewidth=0.5, capsize=0,
        alpha=0.7, zorder=2, label="cleaned disk data",
    )
    _ax_bh.errorbar(
        low_region["H_mT"], low_region["B_mT"],
        xerr=low_region["H_err_mT"], yerr=low_region["B_err_mT"],
        fmt="o", ms=4.5, mfc="none", mec="#1f77b4", mew=0.9,
        ecolor="#1f77b4", elinewidth=0.6, capsize=0,
        alpha=0.95, zorder=3,
        label=rf"low fit: $H \leq {LOW_FIELD_MAX_MT:.2f}$ mT",
    )
    _ax_bh.errorbar(
        high_region["H_mT"], high_region["B_mT"],
        xerr=high_region["H_err_mT"], yerr=high_region["B_err_mT"],
        fmt="o", ms=4.5, mfc="none", mec="#d62728", mew=0.9,
        ecolor="#d62728", elinewidth=0.6, capsize=0,
        alpha=0.95, zorder=3,
        label=rf"high fit: $H \geq {HIGH_FIELD_MIN_MT:.1f}$ mT",
    )
    _ax_bh.axvspan(0, LOW_FIELD_MAX_MT, color="#1f77b4", alpha=0.06, lw=0)
    _ax_bh.axvspan(HIGH_FIELD_MIN_MT, disk_bh["H_mT"].max(), color="#d62728", alpha=0.05, lw=0)
    _ax_bh.set_title(r"$B(H)$ with superconducting disk")
    _ax_bh.set_xlabel(r"Applied field $H$ (mT)")
    _ax_bh.set_ylabel(r"Measured field $B$ (mT)")
    _ax_bh.grid(True, alpha=0.15, lw=0.5)
    _ax_bh.tick_params(direction="in", top=True, right=True)
    _ax_bh.legend(frameon=False)
    fig_bh.tight_layout()
    fig_bh
    return


@app.cell
def _line_plot_helpers(np):
    def line_xy(fit, xmin, xmax):
        x = np.linspace(xmin, xmax, 300)
        y = fit["slope"] * x + fit["intercept"]
        return x, y

    return (line_xy,)


@app.cell
def _plot_fit_windows(
    HIGH_FIELD_MIN_MT,
    LOW_FIELD_MAX_MT,
    disk_bh,
    high_fit,
    line_xy,
    low_fit,
    plt,
):
    fig_windows, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    _kw = dict(
        fmt="o", ms=3.5, mfc="none", mec="0.45", mew=0.7,
        ecolor="0.55", elinewidth=0.5, capsize=0, alpha=0.75, zorder=2,
    )
    ax_low.errorbar(
        disk_bh["H_mT"], disk_bh["B_mT"],
        xerr=disk_bh["H_err_mT"], yerr=disk_bh["B_err_mT"], **_kw,
    )
    _x_low, _y_low = line_xy(low_fit, 0, LOW_FIELD_MAX_MT)
    ax_low.plot(_x_low, _y_low, color="#1f77b4", lw=1.6, zorder=3,
                label=rf"$B={low_fit['slope']:.3f}H{low_fit['intercept']:+.3f}$")
    ax_low.set_xlim(-0.02, LOW_FIELD_MAX_MT + 0.05)
    # Y-autoscale from only the points visible in the low-H window
    # (otherwise matplotlib spans the full B range from the high-H tail).
    _vis = disk_bh[disk_bh["H_mT"].between(-0.02, LOW_FIELD_MAX_MT + 0.05)]
    _y = _vis["B_mT"].to_numpy(); _ye = _vis["B_err_mT"].to_numpy()
    if len(_y):
        _lo, _hi = (_y - _ye).min(), (_y + _ye).max()
        _pad = 0.10 * (_hi - _lo) if _hi > _lo else 0.02
        ax_low.set_ylim(_lo - _pad, _hi + _pad)
    ax_low.set_title("Low-field fit")

    ax_high.errorbar(
        disk_bh["H_mT"], disk_bh["B_mT"],
        xerr=disk_bh["H_err_mT"], yerr=disk_bh["B_err_mT"], **_kw,
    )
    _x_high, _y_high = line_xy(high_fit, HIGH_FIELD_MIN_MT, disk_bh["H_mT"].max())
    ax_high.plot(_x_high, _y_high, color="#d62728", lw=1.6, zorder=3,
                 label=rf"$B={high_fit['slope']:.3f}H{high_fit['intercept']:+.3f}$")
    ax_high.set_xlim(HIGH_FIELD_MIN_MT - 0.1, disk_bh["H_mT"].max() + 0.15)
    _vis_h = disk_bh[disk_bh["H_mT"] >= HIGH_FIELD_MIN_MT - 0.1]
    _yh = _vis_h["B_mT"].to_numpy(); _yhe = _vis_h["B_err_mT"].to_numpy()
    if len(_yh):
        _lo, _hi = (_yh - _yhe).min(), (_yh + _yhe).max()
        _pad = 0.10 * (_hi - _lo) if _hi > _lo else 0.02
        ax_high.set_ylim(_lo - _pad, _hi + _pad)
    ax_high.set_title("High-field fit")

    for _ax in (ax_low, ax_high):
        _ax.set_xlabel(r"Applied field $H$ (mT)")
        _ax.set_ylabel(r"Measured field $B$ (mT)")
        _ax.grid(True, alpha=0.15, lw=0.5)
        _ax.tick_params(direction="in", top=True, right=True)
        _ax.legend(frameon=False)
    fig_windows.tight_layout()
    fig_windows
    return


@app.cell
def _plot_intersection(
    B_star_mT,
    HALL_FACTOR_REL_ERR,
    H_star_err_mT,
    H_star_mT,
    H_star_window_err_mT,
    cal_fit,
    disk_bh,
    high_fit,
    line_xy,
    low_fit,
    plt,
):
    fig_intersection, _ax_int = plt.subplots(figsize=(8.0, 5.2))
    _h_star_cal_err = abs(H_star_mT) * (
        (cal_fit["slope_err"] / cal_fit["slope"]) ** 2 + HALL_FACTOR_REL_ERR**2
    ) ** 0.5
    _h_star_total_err = (
        H_star_err_mT**2 + H_star_window_err_mT**2 + _h_star_cal_err**2
    ) ** 0.5
    _ax_int.errorbar(
        disk_bh["H_mT"], disk_bh["B_mT"],
        xerr=disk_bh["H_err_mT"], yerr=disk_bh["B_err_mT"],
        fmt="o", ms=3.5, mfc="none", mec="0.35", mew=0.7,
        ecolor="0.5", elinewidth=0.5, capsize=0,
        alpha=0.78, zorder=2, label="cleaned disk data",
    )
    _x_int, _y_low_int = line_xy(low_fit, 0, disk_bh["H_mT"].max())
    _, _y_high_int = line_xy(high_fit, 0, disk_bh["H_mT"].max())
    _ax_int.plot(_x_int, _y_low_int, color="#1f77b4", lw=1.6, zorder=3, label="low-field line")
    _ax_int.plot(_x_int, _y_high_int, color="#d62728", lw=1.6, zorder=3, label="high-field line")
    _ax_int.axvspan(
        H_star_mT - _h_star_total_err, H_star_mT + _h_star_total_err,
        color="black", alpha=0.08, lw=0,
        label=rf"$\sigma_{{H^*}}={_h_star_total_err:.3f}$ mT",
    )
    _ax_int.scatter([H_star_mT], [B_star_mT], s=70, color="black", marker="*", zorder=5,
                    label=rf"$H^*={H_star_mT:.3f}$ mT")
    _ax_int.axvline(H_star_mT, color="black", lw=0.9, ls=(0, (6, 3)), alpha=0.65)
    _ax_int.set_title("Penetration-field intersection")
    _ax_int.set_xlabel(r"Applied field $H$ (mT)")
    _ax_int.set_ylabel(r"Measured field $B$ (mT)")
    _ax_int.grid(True, alpha=0.15, lw=0.5)
    _ax_int.tick_params(direction="in", top=True, right=True)
    _ax_int.legend(frameon=False)
    fig_intersection.tight_layout()
    fig_intersection
    return


@app.cell
def _plot_intersection_zoom(
    B_star_mT,
    HALL_FACTOR_REL_ERR,
    H_star_err_mT,
    H_star_mT,
    H_star_window_err_mT,
    cal_fit,
    disk_bh,
    high_fit,
    line_xy,
    low_fit,
    plt,
):
    _h_star_cal_err = abs(H_star_mT) * (
        (cal_fit["slope_err"] / cal_fit["slope"]) ** 2 + HALL_FACTOR_REL_ERR**2
    ) ** 0.5
    _h_star_total_err = (
        H_star_err_mT**2 + H_star_window_err_mT**2 + _h_star_cal_err**2
    ) ** 0.5
    _x_pad = 0.45
    _y_pad = 0.35
    _x_lo, _x_hi = max(0.0, H_star_mT - _x_pad), H_star_mT + _x_pad
    _y_lo, _y_hi = B_star_mT - _y_pad, B_star_mT + _y_pad
    _mask_zoom = disk_bh["H_mT"].between(_x_lo, _x_hi)
    _disk_zoom = disk_bh[_mask_zoom]

    fig_intersection_zoom, _ax_zoom = plt.subplots(figsize=(7.2, 5.0))
    _ax_zoom.errorbar(
        _disk_zoom["H_mT"], _disk_zoom["B_mT"],
        xerr=_disk_zoom["H_err_mT"], yerr=_disk_zoom["B_err_mT"],
        fmt="o", ms=4.5, mfc="none", mec="0.35", mew=0.8,
        ecolor="0.5", elinewidth=0.5, capsize=0,
        alpha=0.85, zorder=2, label="cleaned disk data",
    )
    _x_zoom, _y_low_zoom = line_xy(low_fit, _x_lo, _x_hi)
    _, _y_high_zoom = line_xy(high_fit, _x_lo, _x_hi)
    _ax_zoom.plot(_x_zoom, _y_low_zoom, color="#1f77b4", lw=1.6, zorder=3, label="low-field line")
    _ax_zoom.plot(_x_zoom, _y_high_zoom, color="#d62728", lw=1.6, zorder=3, label="high-field line")
    _ax_zoom.axvspan(
        H_star_mT - _h_star_total_err, H_star_mT + _h_star_total_err,
        color="black", alpha=0.10, lw=0,
        label=rf"$\sigma_{{H^*}}={_h_star_total_err:.3f}$ mT",
    )
    _ax_zoom.scatter([H_star_mT], [B_star_mT], s=90, color="black", marker="*", zorder=5,
                     label=rf"$H^*={H_star_mT:.3f}$ mT")
    _ax_zoom.axvline(H_star_mT, color="black", lw=0.9, ls=(0, (6, 3)), alpha=0.65)
    _ax_zoom.axhline(B_star_mT, color="black", lw=0.9, ls=(0, (1, 2)), alpha=0.55)
    _ax_zoom.set_xlim(_x_lo, _x_hi)
    _ax_zoom.set_ylim(_y_lo, _y_hi)
    _ax_zoom.set_title("Penetration-field intersection (zoom)")
    _ax_zoom.set_xlabel(r"Applied field $H$ (mT)")
    _ax_zoom.set_ylabel(r"Measured field $B$ (mT)")
    _ax_zoom.grid(True, alpha=0.15, lw=0.5)
    _ax_zoom.tick_params(direction="in", top=True, right=True)
    _ax_zoom.legend(frameon=False, loc="upper left")
    fig_intersection_zoom.tight_layout()
    fig_intersection_zoom
    return


@app.cell
def _results(
    B_star_mT,
    DISK_THICKNESS_ERR_M,
    DISK_THICKNESS_M,
    HALL_FACTOR_REL_ERR,
    HALL_T_PER_V,
    HIGH_FIELD_MIN_MT,
    H_star_err_mT,
    H_star_mT,
    H_star_window_err_mT,
    Jc_A_per_m2,
    Jc_err_A_per_m2,
    Jc_window_err_A_per_m2,
    LOW_FIELD_MAX_MT,
    cal_fit,
    disk_bh,
    fit_row,
    high_fit,
    low_fit,
    np,
    pd,
):
    H_star_calibration_err_mT = float(
        abs(H_star_mT)
        * np.sqrt((cal_fit["slope_err"] / cal_fit["slope"]) ** 2 + HALL_FACTOR_REL_ERR**2)
    )
    H_star_total_err_mT = float(np.sqrt(
        H_star_err_mT**2
        + H_star_window_err_mT**2
        + H_star_calibration_err_mT**2
    ))
    Jc_calibration_err_A_per_m2 = float(
        abs(Jc_A_per_m2) * H_star_calibration_err_mT / abs(H_star_mT)
    )
    Jc_thickness_err_A_per_m2 = float(
        abs(Jc_A_per_m2) * DISK_THICKNESS_ERR_M / DISK_THICKNESS_M
    )
    Jc_total_err_A_per_m2 = float(np.sqrt(
        Jc_err_A_per_m2**2
        + Jc_window_err_A_per_m2**2
        + Jc_calibration_err_A_per_m2**2
        + Jc_thickness_err_A_per_m2**2
    ))

    fit_table = pd.DataFrame([
        fit_row("coil calibration B(I)", cal_fit),
        fit_row("low-field B(H)", low_fit),
        fit_row("high-field B(H)", high_fit),
    ])

    results = pd.DataFrame([{
        "hall_T_per_V": HALL_T_PER_V,
        "hall_factor_rel_err": HALL_FACTOR_REL_ERR,
        "low_fit_H_max_mT": LOW_FIELD_MAX_MT,
        "high_fit_H_min_mT": HIGH_FIELD_MIN_MT,
        "coil_slope_mT_per_A": cal_fit["slope"],
        "coil_slope_err_mT_per_A": cal_fit["slope_err"],
        "coil_intercept_mT": cal_fit["intercept"],
        "coil_intercept_err_mT": cal_fit["intercept_err"],
        "low_slope": low_fit["slope"],
        "low_slope_err": low_fit["slope_err"],
        "low_intercept_mT": low_fit["intercept"],
        "low_intercept_err_mT": low_fit["intercept_err"],
        "high_slope": high_fit["slope"],
        "high_slope_err": high_fit["slope_err"],
        "high_intercept_mT": high_fit["intercept"],
        "high_intercept_err_mT": high_fit["intercept_err"],
        "H_star_mT": H_star_mT,
        "H_star_fit_err_mT": H_star_err_mT,
        "H_star_window_err_mT": H_star_window_err_mT,
        "H_star_calibration_err_mT": H_star_calibration_err_mT,
        "H_star_total_err_mT": H_star_total_err_mT,
        "B_star_mT": B_star_mT,
        "Jc_A_per_m2": Jc_A_per_m2,
        "Jc_fit_err_A_per_m2": Jc_err_A_per_m2,
        "Jc_window_err_A_per_m2": Jc_window_err_A_per_m2,
        "Jc_calibration_err_A_per_m2": Jc_calibration_err_A_per_m2,
        "Jc_thickness_err_A_per_m2": Jc_thickness_err_A_per_m2,
        "Jc_total_err_A_per_m2": Jc_total_err_A_per_m2,
        "Jc_total_rel_err_percent": 100 * Jc_total_err_A_per_m2 / np.abs(Jc_A_per_m2),
        "disk_thickness_m": DISK_THICKNESS_M,
        "disk_thickness_err_m": DISK_THICKNESS_ERR_M,
        "disk_points": len(disk_bh),
    }])
    return fit_table, results


@app.cell
def _reference():
    """Reference J_c to compare against, via `taulab.stats.nsigma`.

    Eltsev et al. (arXiv:0909.1628v3) report J_c for high-quality
    Bi-2223 single crystals of order 10^9 A/m² at low T, with values
    comparable to Bi-2223/Ag conductors. The number below is a
    teaching-lab placeholder; EDIT once you have the value/σ you want
    to compare your J_c against (e.g. from Eltsev Fig. 5 at your
    chosen reference temperature/field).
    """
    JC_REF_A_PER_M2       = 1.0e9
    JC_REF_SIGMA_A_PER_M2 = 0.5e9
    return JC_REF_A_PER_M2, JC_REF_SIGMA_A_PER_M2


@app.cell
def _md_summary(mo):
    mo.md(r"""
    ## Results

    The uncertainty on $H^*$ is propagated from the two fitted lines' slope and
    intercept covariance matrices, then combined in quadrature with a fit-window
    sensitivity term and the empty-coil calibration-slope term.

    The lab handout gives the Hall conversion factor $20.45\,\mathrm{T/V}$ but
    not its uncertainty. The disk width/thickness uncertainty is taken as a
    $5\%$ relative uncertainty. Coil-geometry, Hall/disk alignment, and
    demagnetization systematics are still not quantifiable from the supplied
    files.
    """)
    return


@app.cell
def _show_results(
    JC_REF_A_PER_M2, JC_REF_SIGMA_A_PER_M2, nsigma, pd, results,
):
    """Compact summary: value ± σ (rel%) and N_σ against the literature J_c."""
    def _fmt(val, err, unit="", sig=3):
        rel = (err / abs(val) * 100.0) if val else float("nan")
        return f"{val:.{sig}g} ± {err:.{sig}g}{unit} ({rel:.2f}%)"

    _r = results.iloc[0]
    _Hs, _Hs_err = _r["H_star_mT"], _r["H_star_total_err_mT"]
    _Jc, _Jc_err = _r["Jc_A_per_m2"], _r["Jc_total_err_A_per_m2"]
    _ns = nsigma((_Jc, _Jc_err), (JC_REF_A_PER_M2, JC_REF_SIGMA_A_PER_M2))

    final_table = pd.DataFrame([
        {"quantity": "coil slope a₁ [mT/A]",
         "value":    _fmt(_r["coil_slope_mT_per_A"], _r["coil_slope_err_mT_per_A"])},
        {"quantity": "low-field slope",
         "value":    _fmt(_r["low_slope"], _r["low_slope_err"])},
        {"quantity": "high-field slope",
         "value":    _fmt(_r["high_slope"], _r["high_slope_err"])},
        {"quantity": "H* [mT]",
         "value":    _fmt(_Hs, _Hs_err)},
        {"quantity": "J_c [A/m²]",
         "value":    _fmt(_Jc, _Jc_err)},
        {"quantity": f"N_σ vs J_c,ref = {JC_REF_A_PER_M2:.2g} ± {JC_REF_SIGMA_A_PER_M2:.2g}",
         "value":    f"{_ns:.2f}"},
    ])
    return (final_table,)


@app.cell
def _display_final(final_table):
    final_table
    return


@app.cell
def _show_window_sensitivity(window_sensitivity):
    window_sensitivity.round({
        "low_fit_H_max_mT": 2,
        "high_fit_H_min_mT": 2,
        "H_star_mT": 3,
        "Jc_A_per_m2": 2,
    })
    return


@app.cell
def _show_fit_table(fit_table):
    fit_table.round({
        "slope": 5,
        "slope_err": 5,
        "intercept_mT": 5,
        "intercept_err_mT": 5,
        "chi2_dof": 8,
    })
    return


@app.cell
def _write(OUT_DIR, cal, disk_bh, final_table, fit_table, results, window_sensitivity):
    cal.to_csv(OUT_DIR / "coil_calibration_refit.csv", index=False)
    disk_bh.to_csv(OUT_DIR / "disk_BH_with_uncertainties.csv", index=False)
    fit_table.to_csv(OUT_DIR / "fit_table.csv", index=False)
    window_sensitivity.to_csv(OUT_DIR / "fit_window_sensitivity.csv", index=False)
    results.to_csv(OUT_DIR / "part_b_results.csv", index=False)
    final_table.to_csv(OUT_DIR / "part_b_summary.csv", index=False)
    print(f"wrote {OUT_DIR / 'part_b_results.csv'}")
    return


if __name__ == "__main__":
    app.run()
