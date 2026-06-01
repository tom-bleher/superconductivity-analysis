# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///
"""Focused Savitzky-Golay parameter study for Part A.

This script answers a narrow question: if Part A must use one Savitzky-Golay
local-polynomial derivative method, which polynomial order and temperature
window are most defensible for these data?

The ranking combines:
  1. synthetic validation on smooth truth curves shaped like each measured run,
  2. leave-temperature-block-out stability on the real data, and
  3. rejection of settings that produce high failure/outlier rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

from part_a_tc_method_study import (
    GRID_POINTS,
    ROOT,
    _mc_stats,
    _peak_on_grid,
    load_measurements,
    ordered_arrays,
    sigma_temperature_local,
    transition_bounds,
)


OUT_DIR = ROOT / "results" / "part_a"
GRID_STEP_K = 0.02
POLYORDERS = (2, 3, 4, 5)
WINDOWS_K = tuple(np.round(np.arange(1.0, 6.01, 0.25), 2))
SYNTHETIC_DRAWS = 80
SEED = 20260531
SELECTED_POLYORDER = 3
SELECTED_WINDOW_K = 3.25


@dataclass(frozen=True)
class SGSpec:
    polyorder: int
    window_K: float

    @property
    def method(self) -> str:
        return f"sg_p{self.polyorder}_w{self.window_K:.2f}K"


def _sort_xy(temperature: np.ndarray, resistance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(temperature)
    return np.asarray(temperature, dtype=float)[order], np.asarray(resistance, dtype=float)[order]


def _window_points(window_K: float, polyorder: int, n_grid: int) -> int:
    points = int(round(window_K / GRID_STEP_K))
    if points % 2 == 0:
        points += 1
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1
    points = max(points, minimum)
    if points > n_grid:
        points = n_grid if n_grid % 2 == 1 else n_grid - 1
    return points


def savgol_tc(
    temperature: np.ndarray,
    resistance: np.ndarray,
    *,
    polyorder: int,
    window_K: float,
    grid_step_K: float = GRID_STEP_K,
) -> float:
    temperature, resistance = _sort_xy(temperature, resistance)
    grid = np.arange(float(temperature.min()), float(temperature.max()) + grid_step_K / 2.0, grid_step_K)
    uniform_resistance = np.interp(grid, temperature, resistance)
    window_points = _window_points(window_K, polyorder, len(grid))
    if window_points <= polyorder:
        return float("nan")

    derivative = savgol_filter(
        uniform_resistance,
        window_points,
        polyorder,
        deriv=1,
        delta=grid_step_K,
        mode="interp",
    )
    return _peak_on_grid(
        grid,
        np.asarray(derivative, dtype=float),
        transition_bounds(temperature, resistance),
    )


def savgol_curve(
    temperature: np.ndarray,
    resistance: np.ndarray,
    *,
    polyorder: int,
    window_K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature, resistance = _sort_xy(temperature, resistance)
    grid = np.arange(
        float(temperature.min()),
        float(temperature.max()) + GRID_STEP_K / 2.0,
        GRID_STEP_K,
    )
    uniform_resistance = np.interp(grid, temperature, resistance)
    window_points = _window_points(window_K, polyorder, len(grid))
    smooth = savgol_filter(
        uniform_resistance,
        window_points,
        polyorder,
        deriv=0,
        mode="interp",
    )
    derivative = savgol_filter(
        uniform_resistance,
        window_points,
        polyorder,
        deriv=1,
        delta=GRID_STEP_K,
        mode="interp",
    )
    return grid, np.asarray(smooth, dtype=float), np.asarray(derivative, dtype=float)


def derivative_fwhm(
    grid: np.ndarray,
    derivative: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    in_bounds = (
        np.isfinite(derivative)
        & (grid >= bounds[0])
        & (grid <= bounds[1])
    )
    if not np.any(in_bounds):
        return float("nan")

    indexes = np.flatnonzero(in_bounds)
    local = derivative[indexes]
    i_peak = int(indexes[np.nanargmax(local)])
    baseline = max(float(np.nanpercentile(local, 5.0)), 0.0)
    half_height = baseline + 0.5 * (float(derivative[i_peak]) - baseline)

    left = float("nan")
    for i in range(i_peak, indexes[0], -1):
        if derivative[i] >= half_height and derivative[i - 1] < half_height:
            left = linear_crossing(
                grid[i - 1],
                derivative[i - 1],
                grid[i],
                derivative[i],
                half_height,
            )
            break

    right = float("nan")
    for i in range(i_peak, indexes[-1]):
        if derivative[i] >= half_height and derivative[i + 1] < half_height:
            right = linear_crossing(
                grid[i],
                derivative[i],
                grid[i + 1],
                derivative[i + 1],
                half_height,
            )
            break

    if np.isfinite(left) and np.isfinite(right):
        return float(right - left)
    return float("nan")


def linear_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    level: float,
) -> float:
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def synthetic_truths(
    temperature: np.ndarray,
    resistance: np.ndarray,
) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    truths = []
    for degree, target_mohm in ((3, 0.09), (5, 0.09), (3, 0.14), (5, 0.14)):
        spline = UnivariateSpline(
            temperature,
            resistance,
            k=degree,
            s=len(temperature) * (target_mohm * 1e-3) ** 2,
        )
        truth_resistance = spline(temperature)
        residual = resistance - truth_resistance
        grid = np.linspace(float(temperature.min()), float(temperature.max()), GRID_POINTS)
        true_tc = _peak_on_grid(
            grid,
            np.asarray(spline.derivative()(grid), dtype=float),
            transition_bounds(temperature, truth_resistance),
        )
        truths.append((f"k{degree}_{target_mohm:.2f}mohm", true_tc, truth_resistance, residual))
    return truths


def synthetic_validation(
    measurements: dict[str, pd.DataFrame],
    specs: list[SGSpec],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for measurement_id, df in measurements.items():
        temperature, resistance = ordered_arrays(df)
        sigma_t = sigma_temperature_local(temperature)
        for truth_name, true_tc, _truth_resistance, residual in synthetic_truths(temperature, resistance):
            # Use the smooth truth as a callable so temperature perturbations move along the curve.
            truth_degree = int(truth_name[1])
            truth_target = float(truth_name.split("_")[1].replace("mohm", ""))
            truth_spline = UnivariateSpline(
                temperature,
                resistance,
                k=truth_degree,
                s=len(temperature) * (truth_target * 1e-3) ** 2,
            )

            synthetic = []
            for _ in range(SYNTHETIC_DRAWS):
                t_star = temperature + rng.normal(0.0, sigma_t)
                r_star = truth_spline(t_star) + rng.choice(
                    residual,
                    size=len(residual),
                    replace=True,
                )
                synthetic.append((t_star, r_star))

            for spec in specs:
                errors = []
                for t_star, r_star in synthetic:
                    tc = savgol_tc(
                        t_star,
                        r_star,
                        polyorder=spec.polyorder,
                        window_K=spec.window_K,
                    )
                    if np.isfinite(tc):
                        errors.append(tc - true_tc)

                stats = _mc_stats(errors)
                errors_array = np.asarray(errors, dtype=float)
                rows.append(
                    {
                        "measurement_id": measurement_id,
                        "truth_model": truth_name,
                        "method": spec.method,
                        "polyorder": spec.polyorder,
                        "window_K": spec.window_K,
                        "true_tc_K": true_tc,
                        "draws_requested": SYNTHETIC_DRAWS,
                        "draws_used": int(stats["draws_used"]),
                        "fail_fraction": 1.0 - int(stats["draws_used"]) / SYNTHETIC_DRAWS,
                        "median_bias_K": float(stats["p50_K"]),
                        "abs_median_bias_K": abs(float(stats["p50_K"])),
                        "robust_sigma_K": float(stats["robust_sigma_K"]),
                        "rmse_K": (
                            float(np.sqrt(np.mean(errors_array**2)))
                            if len(errors_array)
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def block_delete_stability(
    measurements: dict[str, pd.DataFrame],
    specs: list[SGSpec],
    *,
    blocks: int = 12,
) -> pd.DataFrame:
    rows = []
    for measurement_id, df in measurements.items():
        temperature, resistance = ordered_arrays(df)
        edges = np.linspace(0, len(temperature), blocks + 1, dtype=int)
        for spec in specs:
            nominal = savgol_tc(
                temperature,
                resistance,
                polyorder=spec.polyorder,
                window_K=spec.window_K,
            )
            values = []
            for start, stop in zip(edges[:-1], edges[1:], strict=True):
                use = np.ones(len(temperature), dtype=bool)
                use[start:stop] = False
                if np.count_nonzero(use) < 12:
                    continue
                tc = savgol_tc(
                    temperature[use],
                    resistance[use],
                    polyorder=spec.polyorder,
                    window_K=spec.window_K,
                )
                if np.isfinite(tc):
                    values.append(tc)
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "method": spec.method,
                    "polyorder": spec.polyorder,
                    "window_K": spec.window_K,
                    "nominal_tc_K": nominal,
                    "blocks_used": int(len(values)),
                    "block_delete_std_K": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                    ),
                    "block_delete_max_abs_shift_K": (
                        float(np.nanmax(np.abs(values - nominal)))
                        if len(values)
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def nominal_values(
    measurements: dict[str, pd.DataFrame],
    specs: list[SGSpec],
) -> pd.DataFrame:
    rows = []
    for measurement_id, df in measurements.items():
        meta = df.iloc[0]
        temperature, resistance = ordered_arrays(df)
        for spec in specs:
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "sample_current_mA_nominal": float(meta["sample_current_mA_nominal"]),
                    "series_resistor": meta["series_resistor"],
                    "direction": meta["direction"],
                    "field_condition": meta["field_condition"],
                    "method": spec.method,
                    "polyorder": spec.polyorder,
                    "window_K": spec.window_K,
                    "tc_K": savgol_tc(
                        temperature,
                        resistance,
                        polyorder=spec.polyorder,
                        window_K=spec.window_K,
                    ),
                }
            )
    return pd.DataFrame(rows)


def shape_metrics(
    measurements: dict[str, pd.DataFrame],
    specs: list[SGSpec],
) -> pd.DataFrame:
    rows = []
    reference = SGSpec(polyorder=3, window_K=2.50)
    for measurement_id, df in measurements.items():
        temperature, resistance = ordered_arrays(df)
        bounds = transition_bounds(temperature, resistance)
        ref_grid, _ref_smooth, ref_derivative = savgol_curve(
            temperature,
            resistance,
            polyorder=reference.polyorder,
            window_K=reference.window_K,
        )
        reference_fwhm = derivative_fwhm(ref_grid, ref_derivative, bounds)

        for spec in specs:
            grid, smooth, derivative = savgol_curve(
                temperature,
                resistance,
                polyorder=spec.polyorder,
                window_K=spec.window_K,
            )
            residual = (resistance - np.interp(temperature, grid, smooth)) * 1e3
            fwhm = derivative_fwhm(grid, derivative, bounds)
            rows.append(
                {
                    "measurement_id": measurement_id,
                    "method": spec.method,
                    "polyorder": spec.polyorder,
                    "window_K": spec.window_K,
                    "fwhm_K": fwhm,
                    "reference_fwhm_K": reference_fwhm,
                    "fwhm_ratio_to_p3_w2p5": (
                        float(fwhm / reference_fwhm)
                        if np.isfinite(fwhm)
                        and np.isfinite(reference_fwhm)
                        and reference_fwhm > 0
                        else float("nan")
                    ),
                    "fwhm_increase_K": (
                        float(fwhm - reference_fwhm)
                        if np.isfinite(fwhm) and np.isfinite(reference_fwhm)
                        else float("nan")
                    ),
                    "rmse_mohm": float(np.sqrt(np.mean(residual**2))),
                    "max_abs_residual_mohm": float(np.max(np.abs(residual))),
                    "peak_mohm_per_K": float(
                        np.nanmax(
                            derivative[
                                (grid >= bounds[0])
                                & (grid <= bounds[1])
                                & np.isfinite(derivative)
                            ]
                        )
                        * 1e3
                    ),
                }
            )
    return pd.DataFrame(rows)


def rank_parameters(
    synthetic: pd.DataFrame,
    block: pd.DataFrame,
    nominal: pd.DataFrame,
    shape: pd.DataFrame,
) -> pd.DataFrame:
    synth_summary = (
        synthetic.groupby(["method", "polyorder", "window_K"], as_index=False)
        .agg(
            median_abs_bias_K=("abs_median_bias_K", "median"),
            median_synthetic_sigma_K=("robust_sigma_K", "median"),
            median_synthetic_rmse_K=("rmse_K", "median"),
            worst_synthetic_rmse_K=("rmse_K", "max"),
            median_fail_fraction=("fail_fraction", "median"),
        )
    )
    block_summary = (
        block.groupby(["method", "polyorder", "window_K"], as_index=False)
        .agg(
            median_block_delete_std_K=("block_delete_std_K", "median"),
            worst_block_delete_shift_K=("block_delete_max_abs_shift_K", "max"),
        )
    )
    ranking = synth_summary.merge(
        block_summary,
        on=["method", "polyorder", "window_K"],
        how="left",
    )
    ranking["score_K"] = np.sqrt(
        ranking["median_synthetic_rmse_K"] ** 2
        + ranking["median_block_delete_std_K"] ** 2
    )
    consensus = load_consensus_tc()
    if consensus is not None:
        ranking = ranking.merge(
            consensus_deviation(nominal, consensus),
            on=["method", "polyorder", "window_K"],
            how="left",
        )
    else:
        ranking["median_consensus_dev_K"] = np.nan
        ranking["worst_consensus_dev_K"] = np.nan

    ranking = ranking.merge(
        window_stability(nominal),
        on=["method", "polyorder", "window_K"],
        how="left",
    )
    ranking = ranking.merge(
        shape_summary(shape),
        on=["method", "polyorder", "window_K"],
        how="left",
    )
    ranking["conservative_score_K"] = np.sqrt(
        ranking["score_K"] ** 2
        + ranking["median_consensus_dev_K"].fillna(0.0) ** 2
        + (0.5 * ranking["worst_consensus_dev_K"].fillna(0.0)) ** 2
        + (0.25 * ranking["worst_abs_window_slope_K_per_K"].fillna(0.0)) ** 2
    )
    ranking["passes_selection_screen"] = (
        (ranking["polyorder"] == SELECTED_POLYORDER)
        & (ranking["score_K"] < 0.60)
        & (ranking["median_block_delete_std_K"] < 0.15)
        & (ranking["worst_block_delete_shift_K"] < 0.80)
        & (ranking["worst_consensus_dev_K"] < 0.30)
        & (ranking["median_fwhm_ratio_to_p3_w2p5"] < 1.10)
        & (ranking["worst_fwhm_ratio_to_p3_w2p5"] < 1.20)
        & (ranking["worst_rmse_mohm"] < 0.08)
    )
    return ranking.sort_values("conservative_score_K").reset_index(drop=True)


def shape_summary(shape: pd.DataFrame) -> pd.DataFrame:
    return (
        shape.groupby(["method", "polyorder", "window_K"], as_index=False)
        .agg(
            median_fwhm_ratio_to_p3_w2p5=(
                "fwhm_ratio_to_p3_w2p5",
                "median",
            ),
            worst_fwhm_ratio_to_p3_w2p5=(
                "fwhm_ratio_to_p3_w2p5",
                "max",
            ),
            median_fwhm_increase_K=("fwhm_increase_K", "median"),
            worst_fwhm_increase_K=("fwhm_increase_K", "max"),
            median_rmse_mohm=("rmse_mohm", "median"),
            worst_rmse_mohm=("rmse_mohm", "max"),
            median_peak_mohm_per_K=("peak_mohm_per_K", "median"),
        )
    )


def load_consensus_tc() -> pd.DataFrame | None:
    comparison_path = ROOT / "results" / "part_a" / "tc_method_comparison.csv"
    if not comparison_path.exists():
        return None
    comparison = pd.read_csv(comparison_path)
    consensus = comparison[
        comparison["method"] == "accepted_derivative_ensemble_median"
    ][["measurement_id", "tc_K"]].copy()
    if consensus.empty:
        return None
    return consensus.rename(columns={"tc_K": "consensus_tc_K"})


def consensus_deviation(
    nominal: pd.DataFrame,
    consensus: pd.DataFrame,
) -> pd.DataFrame:
    merged = nominal.merge(consensus, on="measurement_id", how="inner")
    merged["abs_consensus_dev_K"] = np.abs(
        merged["tc_K"] - merged["consensus_tc_K"]
    )
    return (
        merged.groupby(["method", "polyorder", "window_K"], as_index=False)
        .agg(
            median_consensus_dev_K=("abs_consensus_dev_K", "median"),
            worst_consensus_dev_K=("abs_consensus_dev_K", "max"),
        )
    )


def window_stability(nominal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for polyorder, group in nominal.groupby("polyorder"):
        pivot = group.pivot(
            index="measurement_id",
            columns="window_K",
            values="tc_K",
        ).sort_index(axis=1)
        windows = pivot.columns.to_numpy(dtype=float)
        values = pivot.to_numpy(dtype=float)
        for i, window in enumerate(windows):
            if i == 0:
                slope = (values[:, i + 1] - values[:, i]) / (
                    windows[i + 1] - windows[i]
                )
            elif i == len(windows) - 1:
                slope = (values[:, i] - values[:, i - 1]) / (
                    windows[i] - windows[i - 1]
                )
            else:
                slope = (values[:, i + 1] - values[:, i - 1]) / (
                    windows[i + 1] - windows[i - 1]
                )
            rows.append(
                {
                    "method": f"sg_p{polyorder}_w{window:.2f}K",
                    "polyorder": int(polyorder),
                    "window_K": float(window),
                    "median_abs_window_slope_K_per_K": float(
                        np.nanmedian(np.abs(slope))
                    ),
                    "worst_abs_window_slope_K_per_K": float(
                        np.nanmax(np.abs(slope))
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(ranking: pd.DataFrame, output: Path) -> None:
    pivot = ranking.pivot(index="polyorder", columns="window_K", values="score_K")
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        vmin=float(np.nanpercentile(pivot, 5)),
        vmax=float(np.nanpercentile(pivot, 90)),
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel("Savitzky-Golay window width (K)")
    ax.set_ylabel("Polynomial order")
    ax.set_title("SG parameter stress-test score; lower is better")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("score (K)")
    best = ranking.iloc[0]
    x = list(pivot.columns).index(best["window_K"])
    y = list(pivot.index).index(best["polyorder"])
    ax.plot(x, y, marker="o", ms=10, mfc="none", mec="white", mew=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_conservative_heatmap(ranking: pd.DataFrame, output: Path) -> None:
    pivot = ranking.pivot(
        index="polyorder",
        columns="window_K",
        values="conservative_score_K",
    )
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        vmin=float(np.nanpercentile(pivot, 5)),
        vmax=float(np.nanpercentile(pivot, 90)),
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel("Savitzky-Golay window width (K)")
    ax.set_ylabel("Polynomial order")
    ax.set_title("SG conservative selection score; lower is better")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("score (K)")
    x = list(pivot.columns).index(SELECTED_WINDOW_K)
    y = list(pivot.index).index(SELECTED_POLYORDER)
    ax.plot(x, y, marker="o", ms=10, mfc="none", mec="white", mew=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_candidate_derivatives(
    measurements: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    candidate_specs = [
        SGSpec(3, 2.50),
        SGSpec(2, 6.00),
        SGSpec(3, 3.25),
    ]
    show_ids = [
        "partA_30mA_1kohm_cool_nomagnet",
        "partA_30mA_1kohm_heat_nomagnet",
        "partA_100mA_100ohm_heat_nomagnet",
        "partA_240mA_100ohm_cool_nomagnet",
    ]
    colors = ["tab:red", "tab:blue", "tab:green"]
    labels = [
        "old p=3, w=2.5 K",
        "lowest raw score p=2, w=6.0 K",
        "selected p=3, w=3.25 K",
    ]

    fig, axes = plt.subplots(
        len(show_ids),
        2,
        figsize=(13, 3.0 * len(show_ids)),
        sharex=False,
    )
    for row, measurement_id in enumerate(show_ids):
        df = measurements[measurement_id]
        temperature, resistance = ordered_arrays(df)
        ax_curve, ax_derivative = axes[row]
        ax_curve.scatter(temperature, resistance * 1e3, s=12, c="0.65", alpha=0.75)

        for spec, color, label in zip(candidate_specs, colors, labels, strict=True):
            grid = np.arange(
                float(temperature.min()),
                float(temperature.max()) + GRID_STEP_K / 2.0,
                GRID_STEP_K,
            )
            resistance_uniform = np.interp(grid, temperature, resistance)
            window_points = _window_points(spec.window_K, spec.polyorder, len(grid))
            smooth = savgol_filter(
                resistance_uniform,
                window_points,
                spec.polyorder,
                deriv=0,
                mode="interp",
            )
            derivative = savgol_filter(
                resistance_uniform,
                window_points,
                spec.polyorder,
                deriv=1,
                delta=GRID_STEP_K,
                mode="interp",
            )
            tc = _peak_on_grid(
                grid,
                np.asarray(derivative, dtype=float),
                transition_bounds(temperature, resistance),
            )
            ax_curve.plot(grid, smooth * 1e3, color=color, lw=1.5)
            ax_curve.axvline(tc, color=color, lw=1.0, alpha=0.45)
            ax_derivative.plot(
                grid,
                derivative * 1e3,
                color=color,
                lw=1.5,
                label=f"{label}: {tc:.2f} K",
            )
            ax_derivative.axvline(tc, color=color, lw=1.0, alpha=0.45)

        ax_curve.set_title(measurement_id.replace("partA_", ""))
        ax_curve.set_ylabel("R (mOhm)")
        ax_curve.grid(alpha=0.2)
        ax_derivative.set_title("derivative")
        ax_derivative.set_ylabel("dR/dT (mOhm/K)")
        ax_derivative.grid(alpha=0.2)
        ax_derivative.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("T (K)")
    axes[-1, 1].set_xlabel("T (K)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements()
    specs = [SGSpec(polyorder, window) for polyorder in POLYORDERS for window in WINDOWS_K]
    rng = np.random.default_rng(SEED)

    nominal = nominal_values(measurements, specs)
    synthetic = synthetic_validation(measurements, specs, rng)
    block = block_delete_stability(measurements, specs)
    shape = shape_metrics(measurements, specs)
    ranking = rank_parameters(synthetic, block, nominal, shape)

    nominal.to_csv(OUT_DIR / "sg_parameter_nominal_values.csv", index=False)
    synthetic.to_csv(OUT_DIR / "sg_parameter_synthetic_validation.csv", index=False)
    block.to_csv(OUT_DIR / "sg_parameter_block_delete.csv", index=False)
    shape.to_csv(OUT_DIR / "sg_parameter_shape_metrics.csv", index=False)
    ranking.to_csv(OUT_DIR / "sg_parameter_ranking.csv", index=False)
    ranking[ranking["passes_selection_screen"]].to_csv(
        OUT_DIR / "sg_parameter_selection_shortlist.csv",
        index=False,
    )
    plot_heatmap(ranking, OUT_DIR / "sg_parameter_score_heatmap.png")
    plot_conservative_heatmap(
        ranking,
        OUT_DIR / "sg_parameter_conservative_score_heatmap.png",
    )
    plot_candidate_derivatives(
        measurements,
        OUT_DIR / "sg_candidate_derivatives.png",
    )

    print("Top SG parameter settings by conservative score:")
    print(
        ranking[
            [
                "method",
                "conservative_score_K",
                "score_K",
                "median_abs_bias_K",
                "median_synthetic_rmse_K",
                "median_block_delete_std_K",
                "worst_synthetic_rmse_K",
                "median_consensus_dev_K",
                "worst_consensus_dev_K",
                "worst_abs_window_slope_K_per_K",
                "worst_fwhm_ratio_to_p3_w2p5",
                "worst_rmse_mohm",
                "passes_selection_screen",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )
    print("\nShape-preserving cubic shortlist:")
    print(
        ranking[ranking["passes_selection_screen"]][
            [
                "method",
                "score_K",
                "median_synthetic_rmse_K",
                "worst_synthetic_rmse_K",
                "median_block_delete_std_K",
                "worst_block_delete_shift_K",
                "worst_consensus_dev_K",
                "median_fwhm_ratio_to_p3_w2p5",
                "worst_fwhm_ratio_to_p3_w2p5",
                "worst_rmse_mohm",
            ]
        ]
        .sort_values("score_K")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
