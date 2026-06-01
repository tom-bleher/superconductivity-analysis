# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Sensitivity study for Part A resistive critical temperatures.

This script keeps the operational criterion used in the notebook,

    Tc = argmax_T dR/dT,

but tests how much Tc moves when the derivative is estimated by several
reasonable smoothers. It also estimates a propagated Monte Carlo uncertainty
for one deliberately simple primary estimator: a cubic smoothing spline with a
0.06 mOhm residual scale.

The output CSVs are intended as audit artifacts, not as a replacement for the
main Marimo report until the chosen estimator is settled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, make_smoothing_spline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parent
MEAS_DIR = ROOT / "data" / "part_a" / "measurements"
OUT_DIR = ROOT / "results" / "part_a"

T_MIN = 80.0
T_MAX = 105.0
GRID_POINTS = 5000
PRIMARY_METHOD = "spline_k3_target_0.06mohm"


@dataclass(frozen=True)
class TcEstimate:
    measurement_id: str
    method: str
    tc_K: float
    family: str
    accepted: bool
    note: str = ""


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    estimate: Callable[[pd.DataFrame], float]
    accepted: bool = True
    use_for_ranking: bool = True
    note: str = ""


def sigma_voltage(voltage: np.ndarray) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=float)
    voltage_range = 0.1
    voltage_lsd = 1e-6
    voltage_resolution = voltage_lsd / np.sqrt(12.0)
    return np.sqrt(
        (0.00015 * np.abs(voltage) + 0.00004 * voltage_range) ** 2
        + voltage_resolution**2
    )


def sigma_current(current: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    current_range = np.where(np.abs(current) <= 0.2, 0.2, 2.0)
    current_resolution = np.where(
        np.abs(current) <= 0.2,
        1e-6 / np.sqrt(12.0),
        1e-5 / np.sqrt(12.0),
    )
    return np.sqrt(
        (0.0025 * np.abs(current) + 0.00020 * current_range) ** 2
        + current_resolution**2
    )


def sigma_resistance(voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    return np.sqrt(
        (sigma_voltage(voltage) / current) ** 2
        + (voltage * sigma_current(current) / current**2) ** 2
    )


def sigma_temperature_local(temperature: np.ndarray) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    if len(temperature) < 2:
        return np.zeros_like(temperature)

    order = np.argsort(temperature)
    sorted_temperature = temperature[order]
    width = np.empty_like(sorted_temperature)
    width[0] = sorted_temperature[1] - sorted_temperature[0]
    width[-1] = sorted_temperature[-1] - sorted_temperature[-2]
    if len(sorted_temperature) > 2:
        width[1:-1] = sorted_temperature[2:] - sorted_temperature[:-2]

    sigma_sorted = np.abs(width) / (2.0 * np.sqrt(12.0))
    sigma = np.empty_like(sigma_sorted)
    sigma[order] = sigma_sorted
    return sigma


def load_measurements() -> dict[str, pd.DataFrame]:
    measurements: dict[str, pd.DataFrame] = {}
    for path in sorted(MEAS_DIR.glob("partA_*.csv")):
        df = pd.read_csv(path)
        df = df[
            df["temperature_K"].between(T_MIN, T_MAX)
            & df["temperature_K"].notna()
            & df["resistance_ohm"].notna()
        ].copy()
        measurements[path.stem] = df.sort_values("temperature_K").reset_index(
            drop=True
        )
    return measurements


def ordered_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    temperature = df["temperature_K"].to_numpy(dtype=float)
    resistance = df["resistance_ohm"].to_numpy(dtype=float)
    ok = np.isfinite(temperature) & np.isfinite(resistance)
    temperature = temperature[ok]
    resistance = resistance[ok]
    order = np.argsort(temperature)
    return temperature[order], resistance[order]


def transition_bounds(
    temperature: np.ndarray,
    resistance: np.ndarray,
    margin_K: float = 1.0,
    low_fraction: float = 0.05,
    high_fraction: float = 0.95,
) -> tuple[float, float]:
    """Find a broad transition search interval from a fractional rise.

    The margin keeps the search from being too tied to noisy individual points,
    while avoiding endpoint derivative artifacts.
    """
    temperature, resistance = _sort_xy(temperature, resistance)
    n_edge = max(5, len(temperature) // 10)
    low = float(np.nanmedian(resistance[:n_edge]))
    high = float(np.nanmedian(resistance[-n_edge:]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(temperature.min()), float(temperature.max())

    normalized = (resistance - low) / (high - low)
    in_transition = (normalized >= low_fraction) & (normalized <= high_fraction)
    if np.count_nonzero(in_transition) < 5:
        return float(temperature.min()), float(temperature.max())

    left = max(float(temperature.min()), float(temperature[in_transition][0]) - margin_K)
    right = min(float(temperature.max()), float(temperature[in_transition][-1]) + margin_K)
    if right <= left:
        return float(temperature.min()), float(temperature.max())
    return left, right


def _sort_xy(
    temperature: np.ndarray,
    resistance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    temperature = np.asarray(temperature, dtype=float)
    resistance = np.asarray(resistance, dtype=float)
    order = np.argsort(temperature)
    return temperature[order], resistance[order]


def _peak_on_grid(
    grid: np.ndarray,
    derivative: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    ok = np.isfinite(derivative) & (grid >= bounds[0]) & (grid <= bounds[1])
    if not np.any(ok):
        return float("nan")
    indexes = np.flatnonzero(ok)
    i_peak = int(indexes[np.argmax(derivative[ok])])
    return float(grid[i_peak])


def smoothing_spline_tc(
    temperature: np.ndarray,
    resistance: np.ndarray,
    *,
    degree: int,
    target_mohm: float,
    bounds: tuple[float, float] | None = None,
) -> float:
    temperature, resistance = _sort_xy(temperature, resistance)
    smoothing = len(temperature) * (target_mohm * 1e-3) ** 2
    spline = UnivariateSpline(temperature, resistance, k=degree, s=smoothing)
    derivative = spline.derivative()
    grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
    values = np.asarray(derivative(grid), dtype=float)
    bounds = transition_bounds(temperature, resistance) if bounds is None else bounds

    ok = np.isfinite(values) & (grid >= bounds[0]) & (grid <= bounds[1])
    if not np.any(ok):
        return float("nan")
    indexes = np.flatnonzero(ok)
    i_peak = int(indexes[np.argmax(values[ok])])
    lo = float(grid[max(0, i_peak - 12)])
    hi = float(grid[min(len(grid) - 1, i_peak + 12)])
    if hi <= lo:
        return float(grid[i_peak])

    result = minimize_scalar(
        lambda temp: -float(derivative(temp)),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x) if result.success and np.isfinite(result.x) else float(grid[i_peak])


def weighted_spline_tc(
    df: pd.DataFrame,
    *,
    degree: int = 3,
    s_scale: float = 1.0,
) -> float:
    temperature, resistance = ordered_arrays(df)
    sigma_r = sigma_resistance(
        df["voltage_V"].to_numpy(dtype=float),
        df["current_A"].to_numpy(dtype=float),
    )
    sigma_r = sigma_r[np.argsort(df["temperature_K"].to_numpy(dtype=float))]
    weights = 1.0 / np.maximum(sigma_r, np.nanmedian(sigma_r) * 0.05)
    spline = UnivariateSpline(
        temperature,
        resistance,
        w=weights,
        k=degree,
        s=len(temperature) * s_scale,
    )
    derivative = spline.derivative()
    grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
    return _peak_on_grid(grid, np.asarray(derivative(grid), dtype=float), transition_bounds(temperature, resistance))


def robust_weighted_spline_tc(
    df: pd.DataFrame,
    *,
    degree: int = 3,
    s_scale: float = 1.0,
    iterations: int = 4,
) -> float:
    temperature, resistance = ordered_arrays(df)
    order = np.argsort(df["temperature_K"].to_numpy(dtype=float))
    sigma_r = sigma_resistance(
        df["voltage_V"].to_numpy(dtype=float),
        df["current_A"].to_numpy(dtype=float),
    )[order]
    sigma_r = np.maximum(sigma_r, np.nanmedian(sigma_r) * 0.05)

    robust = np.ones_like(resistance)
    spline = None
    for _ in range(iterations):
        weights = robust / sigma_r
        spline = UnivariateSpline(
            temperature,
            resistance,
            w=weights,
            k=degree,
            s=len(temperature) * s_scale,
        )
        standardized = (resistance - spline(temperature)) / sigma_r
        median = float(np.nanmedian(standardized))
        mad = float(np.nanmedian(np.abs(standardized - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            break

        u = standardized / (6.0 * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        robust = np.maximum(robust, 0.05)

    if spline is None:
        return float("nan")
    grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
    return _peak_on_grid(
        grid,
        np.asarray(spline.derivative()(grid), dtype=float),
        transition_bounds(temperature, resistance),
    )


def gcv_spline_tc(temperature: np.ndarray, resistance: np.ndarray) -> float:
    temperature, resistance = _sort_xy(temperature, resistance)
    spline = make_smoothing_spline(temperature, resistance, lam=None)
    grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
    derivative = np.asarray(spline.derivative(1)(grid), dtype=float)
    return _peak_on_grid(grid, derivative, transition_bounds(temperature, resistance))


def savgol_tc(
    temperature: np.ndarray,
    resistance: np.ndarray,
    *,
    window_K: float,
    polyorder: int = 3,
    grid_step_K: float = 0.02,
) -> float:
    temperature, resistance = _sort_xy(temperature, resistance)
    grid = np.arange(float(temperature.min()), float(temperature.max()) + grid_step_K / 2.0, grid_step_K)
    uniform_resistance = np.interp(grid, temperature, resistance)

    window_points = int(round(window_K / grid_step_K))
    if window_points % 2 == 0:
        window_points += 1
    minimum = polyorder + 3
    if minimum % 2 == 0:
        minimum += 1
    window_points = max(window_points, minimum)
    if window_points >= len(grid):
        window_points = len(grid) - 1 if len(grid) % 2 == 0 else len(grid)

    derivative = savgol_filter(
        uniform_resistance,
        window_points,
        polyorder,
        deriv=1,
        delta=grid_step_K,
        mode="interp",
    )
    return _peak_on_grid(grid, np.asarray(derivative, dtype=float), transition_bounds(temperature, resistance))


def local_quadratic_tc(
    temperature: np.ndarray,
    resistance: np.ndarray,
    *,
    window_K: float,
) -> float:
    temperature, resistance = _sort_xy(temperature, resistance)
    grid = np.linspace(float(temperature.min()), float(temperature.max()), 1200)
    derivative = np.full_like(grid, np.nan, dtype=float)
    half_window = window_K / 2.0
    min_points = 6

    for i, center in enumerate(grid):
        distance = np.abs(temperature - center)
        use = distance <= half_window
        if np.count_nonzero(use) < min_points:
            radius = np.sort(distance)[min_points - 1]
            use = distance <= radius

        x = temperature[use] - center
        y = resistance[use]
        radius = max(float(np.max(np.abs(x))), half_window, 1e-12)
        u = np.abs(x) / radius
        weights = (1.0 - u**3) ** 3
        weights[u >= 1.0] = 0.0
        if np.count_nonzero(weights > 0.0) < min_points:
            weights = np.ones_like(y)

        design = np.column_stack([np.ones_like(x), x, x**2])
        beta = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
        derivative[i] = beta[1]

    return _peak_on_grid(grid, derivative, transition_bounds(temperature, resistance))


def sigmoid_midpoint_tc(temperature: np.ndarray, resistance: np.ndarray) -> float:
    """Diagnostic only: this estimates a model midpoint, not max dR/dT."""
    temperature, resistance = _sort_xy(temperature, resistance)
    y_mohm = resistance * 1e3
    n_edge = max(5, len(temperature) // 10)
    low = float(np.median(y_mohm[:n_edge]))
    high = float(np.median(y_mohm[-n_edge:]))
    amplitude = max(high - low, 0.1)
    center0 = smoothing_spline_tc(temperature, resistance, degree=3, target_mohm=0.06)

    def model(temp: np.ndarray, base: float, slope: float, amp: float, center: float, width: float) -> np.ndarray:
        z = np.clip(-(temp - center) / width, -60.0, 60.0)
        return base + slope * (temp - center) + amp / (1.0 + np.exp(z))

    bounds = (
        [low - 5.0, -2.0, 0.0, float(temperature.min()), 0.05],
        [high + 5.0, 2.0, 50.0, float(temperature.max()), 10.0],
    )
    popt, _ = curve_fit(
        model,
        temperature,
        y_mohm,
        p0=[low, 0.0, amplitude, center0, 1.0],
        bounds=bounds,
        maxfev=50000,
    )
    return float(popt[3])


def method_specs() -> list[MethodSpec]:
    specs: list[MethodSpec] = []

    for degree in (3, 5):
        for target in (0.03, 0.04, 0.06, 0.08, 0.12, 0.16):
            accepted = target in (0.04, 0.06, 0.08, 0.12)
            use_for_ranking = (
                accepted
                or (degree == 3 and target == 0.03)
            )

            def estimate(
                df: pd.DataFrame,
                degree: int = degree,
                target: float = target,
            ) -> float:
                temperature, resistance = ordered_arrays(df)
                return smoothing_spline_tc(
                    temperature,
                    resistance,
                    degree=degree,
                    target_mohm=target,
                )

            specs.append(
                MethodSpec(
                    f"spline_k{degree}_target_{target:.2f}mohm",
                    "smoothing_spline",
                    estimate,
                    accepted=accepted,
                    use_for_ranking=use_for_ranking,
                    note="" if accepted else "Smoothing sensitivity endpoint.",
                )
            )

    for window in (1.0, 1.2, 1.6, 2.0, 2.5, 3.0):
        accepted = window in (1.6, 2.0, 2.5)

        def estimate(df: pd.DataFrame, window: float = window) -> float:
            temperature, resistance = ordered_arrays(df)
            return savgol_tc(temperature, resistance, window_K=window)

        specs.append(
            MethodSpec(
                f"savgol_poly3_window_{window:.1f}K",
                "savitzky_golay",
                estimate,
                accepted=accepted,
                use_for_ranking=window in (1.2, 1.6, 2.0, 2.5),
                note="" if accepted else "Window sensitivity endpoint.",
            )
        )

    for window in (1.2, 1.8, 2.4, 3.0, 3.6):
        accepted = window in (1.8, 2.4, 3.0)

        def estimate(df: pd.DataFrame, window: float = window) -> float:
            temperature, resistance = ordered_arrays(df)
            return local_quadratic_tc(temperature, resistance, window_K=window)

        specs.append(
            MethodSpec(
                f"local_quadratic_window_{window:.1f}K",
                "local_polynomial",
                estimate,
                accepted=accepted,
                use_for_ranking=window == 2.4,
                note="" if accepted else "Window sensitivity endpoint.",
            )
        )

    accepted_specs = tuple(spec for spec in specs if spec.accepted)

    def accepted_ensemble_median(df: pd.DataFrame) -> float:
        values = []
        for spec in accepted_specs:
            try:
                value = float(spec.estimate(df))
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                values.append(value)
        return float(np.median(values)) if values else float("nan")

    specs.append(
        MethodSpec(
            "accepted_derivative_ensemble_median",
            "ensemble",
            accepted_ensemble_median,
            accepted=False,
            use_for_ranking=True,
            note="Median of the pre-declared accepted derivative estimators.",
        )
    )

    for s_scale in (0.7, 1.0, 1.3):
        specs.append(
            MethodSpec(
                f"weighted_cubic_spline_s={s_scale:.1f}N",
                "weighted_spline",
                lambda df, s_scale=s_scale: weighted_spline_tc(
                    df,
                    degree=3,
                    s_scale=s_scale,
                ),
                accepted=False,
                use_for_ranking=True,
                note=(
                    "Noise-weighted spline using the chi-square rule "
                    "s = scale * N."
                ),
            )
        )
        specs.append(
            MethodSpec(
                f"robust_weighted_cubic_spline_s={s_scale:.1f}N",
                "robust_weighted_spline",
                lambda df, s_scale=s_scale: robust_weighted_spline_tc(
                    df,
                    degree=3,
                    s_scale=s_scale,
                ),
                accepted=False,
                use_for_ranking=True,
                note=(
                    "Noise-weighted spline with iterative bisquare-style "
                    "residual downweighting."
                ),
            )
        )

    specs.extend(
        [
            MethodSpec(
                "gcv_cubic_smoothing_spline",
                "diagnostic",
                lambda df: gcv_spline_tc(*ordered_arrays(df)),
                accepted=False,
                use_for_ranking=False,
                note="GCV can undersmooth derivative peaks on this data.",
            ),
            MethodSpec(
                "weighted_cubic_spline_s=N",
                "diagnostic",
                lambda df: weighted_spline_tc(df, degree=3, s_scale=1.0),
                accepted=False,
                use_for_ranking=False,
                note="Uses DMM uncertainties as independent point weights.",
            ),
            MethodSpec(
                "sigmoid_midpoint",
                "diagnostic",
                lambda df: sigmoid_midpoint_tc(*ordered_arrays(df)),
                accepted=False,
                use_for_ranking=False,
                note="Model midpoint; not the max-derivative definition.",
            ),
        ]
    )
    return specs


def estimate_all_methods(measurements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    estimates: list[TcEstimate] = []
    specs = method_specs()

    for measurement_id, df in measurements.items():
        for spec in specs:
            try:
                tc = float(spec.estimate(df))
            except Exception:
                tc = float("nan")
            estimates.append(
                TcEstimate(
                    measurement_id,
                    spec.name,
                    tc,
                    spec.family,
                    spec.accepted,
                    spec.note,
                )
            )

    return pd.DataFrame([estimate.__dict__ for estimate in estimates])


def _mc_stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "draws_used": 0,
            "mean_K": float("nan"),
            "std_K": float("nan"),
            "p16_K": float("nan"),
            "p50_K": float("nan"),
            "p84_K": float("nan"),
            "robust_sigma_K": float("nan"),
        }

    p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
    return {
        "draws_used": int(len(values)),
        "mean_K": float(np.mean(values)),
        "std_K": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p16_K": float(p16),
        "p50_K": float(p50),
        "p84_K": float(p84),
        "robust_sigma_K": float((p84 - p16) / 2.0),
    }


def monte_carlo_primary(
    measurement_id: str,
    df: pd.DataFrame,
    *,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    temperature = df["temperature_K"].to_numpy(dtype=float)
    voltage = df["voltage_V"].to_numpy(dtype=float)
    current = df["current_A"].to_numpy(dtype=float)

    sigma_t = sigma_temperature_local(temperature)
    sigma_v = sigma_voltage(voltage)
    sigma_i = sigma_current(current)
    nominal_resistance = voltage / current
    nominal_tc = smoothing_spline_tc(
        temperature,
        nominal_resistance,
        degree=3,
        target_mohm=0.06,
    )
    sorted_temperature, sorted_resistance = _sort_xy(temperature, nominal_resistance)
    nominal_spline = UnivariateSpline(
        sorted_temperature,
        sorted_resistance,
        k=3,
        s=len(sorted_temperature) * (0.06e-3) ** 2,
    )
    residual = sorted_resistance - nominal_spline(sorted_temperature)

    temperature_samples: list[float] = []
    residual_wild_samples: list[float] = []
    residual_resample_samples: list[float] = []
    independent_instrument_samples: list[float] = []
    for _ in range(draws):
        temperature_star = temperature + rng.normal(0.0, sigma_t)
        try:
            temperature_samples.append(
                smoothing_spline_tc(
                    temperature_star,
                    nominal_resistance,
                    degree=3,
                    target_mohm=0.06,
                )
            )
        except Exception:
            pass

        try:
            signs = rng.choice([-1.0, 1.0], size=len(residual))
            residual_wild_samples.append(
                smoothing_spline_tc(
                    sorted_temperature,
                    nominal_spline(sorted_temperature) + signs * residual,
                    degree=3,
                    target_mohm=0.06,
                )
            )
        except Exception:
            pass

        try:
            residual_resample_samples.append(
                smoothing_spline_tc(
                    sorted_temperature,
                    nominal_spline(sorted_temperature)
                    + rng.choice(residual, size=len(residual), replace=True),
                    degree=3,
                    target_mohm=0.06,
                )
            )
        except Exception:
            pass

        voltage_star = voltage + rng.normal(0.0, sigma_v)
        current_star = current + rng.normal(0.0, sigma_i)
        resistance_star = voltage_star / current_star
        try:
            independent_instrument_samples.append(
                smoothing_spline_tc(
                    temperature,
                    resistance_star,
                    degree=3,
                    target_mohm=0.06,
                )
            )
        except Exception:
            pass

    temperature_stats = _mc_stats(temperature_samples)
    wild_stats = _mc_stats(residual_wild_samples)
    resample_stats = _mc_stats(residual_resample_samples)
    independent_stats = _mc_stats(independent_instrument_samples)

    residual_sigma = float(
        np.nanmax(
            [
                wild_stats["robust_sigma_K"],
                resample_stats["robust_sigma_K"],
            ]
        )
    )
    mc_sigma = float(
        np.sqrt(float(temperature_stats["robust_sigma_K"]) ** 2 + residual_sigma**2)
    )
    return {
        "measurement_id": measurement_id,
        "mc_draws_requested": draws,
        "primary_nominal_tc_K": float(nominal_tc),
        "temperature_draws_used": int(temperature_stats["draws_used"]),
        "temperature_robust_sigma_K": float(temperature_stats["robust_sigma_K"]),
        "temperature_p16_K": float(temperature_stats["p16_K"]),
        "temperature_p50_K": float(temperature_stats["p50_K"]),
        "temperature_p84_K": float(temperature_stats["p84_K"]),
        "residual_wild_draws_used": int(wild_stats["draws_used"]),
        "residual_wild_robust_sigma_K": float(wild_stats["robust_sigma_K"]),
        "residual_resample_draws_used": int(resample_stats["draws_used"]),
        "residual_resample_robust_sigma_K": float(resample_stats["robust_sigma_K"]),
        "residual_robust_sigma_K": residual_sigma,
        "mc_sigma_K": mc_sigma,
        "independent_instrument_draws_used": int(independent_stats["draws_used"]),
        "independent_instrument_robust_sigma_K": float(independent_stats["robust_sigma_K"]),
        "independent_instrument_p16_K": float(independent_stats["p16_K"]),
        "independent_instrument_p50_K": float(independent_stats["p50_K"]),
        "independent_instrument_p84_K": float(independent_stats["p84_K"]),
    }


def summarize(
    measurements: dict[str, pd.DataFrame],
    method_results: pd.DataFrame,
    mc_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    current_path = OUT_DIR / "tc_summary.csv"
    current = pd.read_csv(current_path) if current_path.exists() else pd.DataFrame()

    for measurement_id, df in measurements.items():
        meta = df.iloc[0]
        accepted = method_results[
            (method_results["measurement_id"] == measurement_id)
            & (method_results["accepted"])
            & method_results["tc_K"].notna()
        ].copy()
        primary = method_results[
            (method_results["measurement_id"] == measurement_id)
            & (method_results["method"] == PRIMARY_METHOD)
        ]["tc_K"].iloc[0]
        mc = mc_results[mc_results["measurement_id"] == measurement_id].iloc[0]

        method_sigma = float(accepted["tc_K"].std(ddof=1))
        method_p16, method_p50, method_p84 = np.percentile(
            accepted["tc_K"].to_numpy(dtype=float),
            [16.0, 50.0, 84.0],
        )
        current_row = current[current["measurement_id"] == measurement_id]
        current_tc = float(current_row["tc_K"].iloc[0]) if len(current_row) else float("nan")
        current_sigma = (
            float(current_row["tc_sigma_K"].iloc[0]) if len(current_row) else float("nan")
        )
        mc_sigma = float(mc["mc_sigma_K"])
        total_sigma = float(np.sqrt(method_sigma**2 + mc_sigma**2))

        rows.append(
            {
                "measurement_id": measurement_id,
                "sample_current_mA_nominal": float(meta["sample_current_mA_nominal"]),
                "series_resistor": meta["series_resistor"],
                "direction": meta["direction"],
                "field_condition": meta["field_condition"],
                "current_notebook_tc_K": current_tc,
                "current_notebook_sigma_K": current_sigma,
                "primary_method": PRIMARY_METHOD,
                "primary_tc_K": float(primary),
                "accepted_method_median_tc_K": float(method_p50),
                "accepted_method_sigma_K": method_sigma,
                "accepted_method_p16_K": float(method_p16),
                "accepted_method_p84_K": float(method_p84),
                "accepted_method_min_K": float(accepted["tc_K"].min()),
                "accepted_method_max_K": float(accepted["tc_K"].max()),
                "mc_sigma_K": mc_sigma,
                "temperature_robust_sigma_K": float(mc["temperature_robust_sigma_K"]),
                "residual_robust_sigma_K": float(mc["residual_robust_sigma_K"]),
                "independent_instrument_robust_sigma_K": float(
                    mc["independent_instrument_robust_sigma_K"]
                ),
                "combined_mc_method_sigma_K": total_sigma,
            }
        )

    return pd.DataFrame(rows).sort_values("primary_tc_K").reset_index(drop=True)


def final_low_current_pair(summary: pd.DataFrame) -> pd.DataFrame:
    anchor = summary[
        np.isclose(summary["sample_current_mA_nominal"], 30.0)
        & (summary["field_condition"] == "no_magnet")
    ].copy()
    if len(anchor) == 0:
        return pd.DataFrame()

    sigma = anchor["combined_mc_method_sigma_K"].to_numpy(dtype=float)
    values = anchor["accepted_method_median_tc_K"].to_numpy(dtype=float)
    ok = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(values)
    if not np.any(ok):
        return pd.DataFrame()

    # Heat/cool splitting is physical hysteresis, not repeated random sampling.
    # Use a plain center for the pair, and report the half-split separately.
    tc = float(np.mean(values[ok]))
    stat_sigma = float(np.sqrt(np.sum(sigma[ok] ** 2)) / np.count_nonzero(ok))
    half_hysteresis = (
        float((anchor["accepted_method_median_tc_K"].max() - anchor["accepted_method_median_tc_K"].min()) / 2.0)
        if len(anchor) >= 2
        else float("nan")
    )
    return pd.DataFrame(
        [
            {
                "basis": "30mA zero-field heat/cool accepted-method median",
                "tc_K": tc,
                "method_mc_sigma_of_pair_K": stat_sigma,
                "half_heat_cool_split_K": half_hysteresis,
                "recommended_report": (
                    f"{tc:.2f} +/- {stat_sigma:.2f} K method+MC; "
                    f"heat/cool half-split {half_hysteresis:.2f} K"
                ),
            }
        ]
    )


def plot_method_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = np.arange(len(summary))

    ax.hlines(
        y,
        summary["accepted_method_min_K"],
        summary["accepted_method_max_K"],
        color="#8c8c8c",
        lw=4.0,
        alpha=0.28,
        label="accepted method range",
    )
    ax.errorbar(
        summary["accepted_method_median_tc_K"],
        y,
        xerr=[
            summary["accepted_method_median_tc_K"] - summary["accepted_method_p16_K"],
            summary["accepted_method_p84_K"] - summary["accepted_method_median_tc_K"],
        ],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.0,
        capsize=0,
        label="accepted method median, 16-84%",
    )
    ax.plot(
        summary["current_notebook_tc_K"],
        y,
        "x",
        color="#d62728",
        ms=7,
        mew=1.4,
        label="current notebook k=5, 0.04 mOhm",
    )

    labels = [
        f"{int(row.sample_current_mA_nominal)} mA {row.direction}, "
        f"{'B=0' if row.field_condition == 'no_magnet' else 'B!=0'}"
        for row in summary.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$T_c$ from max $dR/dT$ (K)")
    ax.set_title("Part A derivative-method sensitivity")
    ax.grid(True, axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.14, top=0.90)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def block_delete_stability(
    measurements: dict[str, pd.DataFrame],
    specs: list[MethodSpec],
    *,
    blocks: int = 12,
) -> pd.DataFrame:
    rows = []
    ranking_specs = [spec for spec in specs if spec.use_for_ranking]
    for measurement_id, df in measurements.items():
        n = len(df)
        edges = np.linspace(0, n, blocks + 1, dtype=int)
        for spec in ranking_specs:
            try:
                nominal = float(spec.estimate(df))
            except Exception:
                nominal = float("nan")

            values: list[float] = []
            for start, stop in zip(edges[:-1], edges[1:], strict=True):
                if stop <= start:
                    continue
                reduced = pd.concat([df.iloc[:start], df.iloc[stop:]]).reset_index(drop=True)
                if len(reduced) < 12:
                    continue
                try:
                    value = float(spec.estimate(reduced))
                except Exception:
                    value = float("nan")
                if np.isfinite(value):
                    values.append(value)

            values_array = np.asarray(values, dtype=float)
            deltas = values_array - nominal
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "method": spec.name,
                    "family": spec.family,
                    "nominal_tc_K": nominal,
                    "blocks_requested": blocks,
                    "blocks_used": int(len(values_array)),
                    "block_delete_std_K": (
                        float(np.std(values_array, ddof=1))
                        if len(values_array) > 1
                        else float("nan")
                    ),
                    "block_delete_max_abs_shift_K": (
                        float(np.nanmax(np.abs(deltas)))
                        if len(values_array)
                        else float("nan")
                    ),
                    "block_delete_p16_K": (
                        float(np.percentile(values_array, 16.0))
                        if len(values_array)
                        else float("nan")
                    ),
                    "block_delete_p84_K": (
                        float(np.percentile(values_array, 84.0))
                        if len(values_array)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def search_bounds_stability(measurements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    methods = [
        ("primary_spline_k3_0.06mohm", 3, 0.06),
        ("current_notebook_spline_k5_0.04mohm", 5, 0.04),
    ]
    low_fractions = (0.03, 0.05, 0.10)
    high_fractions = (0.90, 0.95, 0.97)
    margins = (0.5, 1.0, 1.5)

    for measurement_id, df in measurements.items():
        temperature, resistance = ordered_arrays(df)
        for method, degree, target in methods:
            values = []
            for low_fraction in low_fractions:
                for high_fraction in high_fractions:
                    if low_fraction >= high_fraction:
                        continue
                    for margin in margins:
                        bounds = transition_bounds(
                            temperature,
                            resistance,
                            margin_K=margin,
                            low_fraction=low_fraction,
                            high_fraction=high_fraction,
                        )
                        try:
                            tc = smoothing_spline_tc(
                                temperature,
                                resistance,
                                degree=degree,
                                target_mohm=target,
                                bounds=bounds,
                            )
                        except Exception:
                            tc = float("nan")
                        values.append(
                            {
                                "low_fraction": low_fraction,
                                "high_fraction": high_fraction,
                                "margin_K": margin,
                                "tc_K": tc,
                            }
                        )

            values_df = pd.DataFrame(values)
            finite = values_df["tc_K"].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "method": method,
                    "bounds_trials": len(values_df),
                    "bounds_trials_finite": int(len(finite)),
                    "bounds_median_tc_K": (
                        float(np.median(finite)) if len(finite) else float("nan")
                    ),
                    "bounds_std_K": (
                        float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
                    ),
                    "bounds_min_K": float(np.min(finite)) if len(finite) else float("nan"),
                    "bounds_max_K": float(np.max(finite)) if len(finite) else float("nan"),
                    "bounds_range_K": (
                        float(np.max(finite) - np.min(finite))
                        if len(finite)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def synthetic_validation(
    measurements: dict[str, pd.DataFrame],
    specs: list[MethodSpec],
    *,
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    ranking_specs = [spec for spec in specs if spec.use_for_ranking]
    truth_models = [
        ("truth_cubic_spline_0.09mohm", 3, 0.09),
        ("truth_quintic_spline_0.09mohm", 5, 0.09),
        ("truth_cubic_spline_0.14mohm", 3, 0.14),
        ("truth_quintic_spline_0.14mohm", 5, 0.14),
    ]

    for measurement_id, df in measurements.items():
        temperature, resistance = ordered_arrays(df)
        order = np.argsort(df["temperature_K"].to_numpy(dtype=float))
        current = df["current_A"].to_numpy(dtype=float)[order]
        sigma_t = sigma_temperature_local(temperature)
        for truth_name, truth_degree, truth_target in truth_models:
            truth_smoothing = len(temperature) * (truth_target * 1e-3) ** 2
            truth_spline = UnivariateSpline(
                temperature,
                resistance,
                k=truth_degree,
                s=truth_smoothing,
            )
            truth_resistance = truth_spline(temperature)
            residual = resistance - truth_resistance
            grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
            true_tc = _peak_on_grid(
                grid,
                np.asarray(truth_spline.derivative()(grid), dtype=float),
                transition_bounds(temperature, truth_resistance),
            )

            synthetic_frames = []
            for _ in range(draws):
                temperature_star = temperature + rng.normal(0.0, sigma_t)
                resistance_star = truth_spline(temperature_star) + rng.choice(
                    residual,
                    size=len(residual),
                    replace=True,
                )
                voltage_star = resistance_star * current
                synthetic_frames.append(
                    pd.DataFrame(
                        {
                            "temperature_K": temperature_star,
                            "voltage_V": voltage_star,
                            "current_A": current,
                            "resistance_ohm": resistance_star,
                        }
                    )
                )

            for spec in ranking_specs:
                errors: list[float] = []
                estimates: list[float] = []
                for synthetic_df in synthetic_frames:
                    try:
                        estimate = float(spec.estimate(synthetic_df))
                    except Exception:
                        estimate = float("nan")
                    if np.isfinite(estimate):
                        estimates.append(estimate)
                        errors.append(estimate - true_tc)

                error_stats = _mc_stats(errors)
                estimate_stats = _mc_stats(estimates)
                errors_array = np.asarray(errors, dtype=float)
                rmse = (
                    float(np.sqrt(np.mean(errors_array**2)))
                    if len(errors_array)
                    else float("nan")
                )
                rows.append(
                    {
                        "measurement_id": measurement_id,
                        "truth_model": truth_name,
                        "true_tc_K": true_tc,
                        "method": spec.name,
                        "family": spec.family,
                        "draws_requested": draws,
                        "draws_used": int(error_stats["draws_used"]),
                        "fail_fraction": 1.0 - float(error_stats["draws_used"]) / draws,
                        "mean_bias_K": float(error_stats["mean_K"]),
                        "median_bias_K": float(error_stats["p50_K"]),
                        "abs_median_bias_K": abs(float(error_stats["p50_K"])),
                        "robust_sigma_K": float(error_stats["robust_sigma_K"]),
                        "rmse_K": rmse,
                        "estimate_p16_K": float(estimate_stats["p16_K"]),
                        "estimate_p50_K": float(estimate_stats["p50_K"]),
                        "estimate_p84_K": float(estimate_stats["p84_K"]),
                    }
                )
    return pd.DataFrame(rows)


def rank_estimators(
    synthetic_results: pd.DataFrame,
    block_stability: pd.DataFrame,
) -> pd.DataFrame:
    synth = (
        synthetic_results.groupby(["method", "family"], as_index=False)
        .agg(
            median_abs_bias_K=("abs_median_bias_K", "median"),
            median_synthetic_sigma_K=("robust_sigma_K", "median"),
            median_synthetic_rmse_K=("rmse_K", "median"),
            worst_synthetic_rmse_K=("rmse_K", "max"),
            median_fail_fraction=("fail_fraction", "median"),
        )
    )
    block = (
        block_stability.groupby(["method", "family"], as_index=False)
        .agg(
            median_block_delete_std_K=("block_delete_std_K", "median"),
            worst_block_delete_shift_K=("block_delete_max_abs_shift_K", "max"),
        )
    )
    ranking = synth.merge(block, on=["method", "family"], how="left")
    ranking["score_K"] = np.sqrt(
        ranking["median_synthetic_rmse_K"] ** 2
        + ranking["median_block_delete_std_K"] ** 2
    )
    return ranking.sort_values("score_K").reset_index(drop=True)


def plot_estimator_ranking(ranking: pd.DataFrame, output: Path) -> None:
    top = ranking.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(top))
    ax.barh(y, top["score_K"], color="#4c78a8", alpha=0.82, label="combined score")
    ax.scatter(
        top["median_abs_bias_K"],
        y,
        color="#f58518",
        s=30,
        zorder=3,
        label="median synthetic |bias|",
    )
    ax.scatter(
        top["median_block_delete_std_K"],
        y,
        color="#54a24b",
        s=30,
        zorder=3,
        label="median block-delete std",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top["method"])
    ax.set_xlabel("K")
    ax.set_title("Estimator ranking on synthetic and real-data stress tests")
    ax.grid(True, axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.subplots_adjust(left=0.39, right=0.97, bottom=0.11, top=0.91)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(draws: int = 1000, synthetic_draws: int = 100, seed: int = 20260530) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements()
    specs = method_specs()
    method_results = estimate_all_methods(measurements)

    rng = np.random.default_rng(seed)
    mc_results = pd.DataFrame(
        [
            monte_carlo_primary(measurement_id, df, draws=draws, rng=rng)
            for measurement_id, df in measurements.items()
        ]
    )
    summary = summarize(measurements, method_results, mc_results)
    final_pair = final_low_current_pair(summary)
    block_stability = block_delete_stability(measurements, specs)
    bounds_stability = search_bounds_stability(measurements)
    synthetic_results = synthetic_validation(
        measurements,
        specs,
        draws=synthetic_draws,
        rng=rng,
    )
    estimator_ranking = rank_estimators(synthetic_results, block_stability)

    method_path = OUT_DIR / "tc_method_comparison.csv"
    mc_path = OUT_DIR / "tc_primary_monte_carlo.csv"
    summary_path = OUT_DIR / "tc_method_summary.csv"
    final_path = OUT_DIR / "tc_recommended_low_current_pair.csv"
    plot_path = OUT_DIR / "tc_method_sensitivity.png"
    block_path = OUT_DIR / "tc_block_delete_stability.csv"
    bounds_path = OUT_DIR / "tc_search_bounds_stability.csv"
    synthetic_path = OUT_DIR / "tc_synthetic_validation.csv"
    ranking_path = OUT_DIR / "tc_estimator_ranking.csv"
    ranking_plot_path = OUT_DIR / "tc_estimator_ranking.png"

    method_results.to_csv(method_path, index=False)
    mc_results.to_csv(mc_path, index=False)
    summary.to_csv(summary_path, index=False)
    final_pair.to_csv(final_path, index=False)
    block_stability.to_csv(block_path, index=False)
    bounds_stability.to_csv(bounds_path, index=False)
    synthetic_results.to_csv(synthetic_path, index=False)
    estimator_ranking.to_csv(ranking_path, index=False)
    plot_method_summary(summary, plot_path)
    plot_estimator_ranking(estimator_ranking, ranking_plot_path)

    print(f"Wrote {method_path.relative_to(ROOT)}")
    print(f"Wrote {mc_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {final_path.relative_to(ROOT)}")
    print(f"Wrote {plot_path.relative_to(ROOT)}")
    print(f"Wrote {block_path.relative_to(ROOT)}")
    print(f"Wrote {bounds_path.relative_to(ROOT)}")
    print(f"Wrote {synthetic_path.relative_to(ROOT)}")
    print(f"Wrote {ranking_path.relative_to(ROOT)}")
    print(f"Wrote {ranking_plot_path.relative_to(ROOT)}")
    print("Top estimators by stress-test score:")
    print(estimator_ranking[["method", "score_K"]].head(5).to_string(index=False))
    if len(final_pair):
        print(final_pair["recommended_report"].iloc[0])


if __name__ == "__main__":
    main()
