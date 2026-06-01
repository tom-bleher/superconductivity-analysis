# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "openpyxl",
#     "scipy",
# ]
# ///
"""Audit Part A raw Excel trimming against the cleaned measurement CSVs.

This script is intentionally read-only with respect to the analysis inputs. It
checks whether cleaned CSV rows are exact rows from the original Excel files,
summarizes rows excluded by the current trimming, and writes small audit tables
and plots to results/part_a_raw_trim_audit/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_data"
MEAS_DIR = ROOT / "data" / "part_a" / "measurements"
OUT_DIR = ROOT / "results" / "part_a_raw_trim_audit"

T_MIN_K = 80.0
T_MAX_K = 105.0
SG_POLYORDER = 3
SG_WINDOW_K = 3.25
SG_GRID_STEP_K = 0.02


@dataclass(frozen=True)
class RunMeta:
    current_label: str
    current_mA: float
    series_resistor: str
    direction: str
    field_condition: str


def parse_meta(path: Path) -> RunMeta:
    stem = path.stem
    current_match = re.match(r"(?P<current>[0-9.]+)ma_", stem, re.IGNORECASE)
    resistor_match = re.search(r"_(?P<resistor>[0-9.]+k?ohm)_", stem, re.IGNORECASE)
    current_label = current_match.group("current") if current_match else "unknown"
    current_mA = float(current_label) if current_label != "unknown" else float("nan")
    series_resistor = resistor_match.group("resistor") if resistor_match else "unknown"
    direction = "heat" if "_heat" in stem else "cool" if "_cool" in stem else "unknown"
    field_condition = "magnet" if "magnet" in stem else "no_magnet"
    return RunMeta(current_label, current_mA, series_resistor, direction, field_condition)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "t[sec]": "time_s",
            "T[K]": "temperature_K",
            "V[V]": "voltage_V",
            "I[A]": "current_A",
        }
    )
    df = df[["time_s", "temperature_K", "voltage_V", "current_A"]].copy()
    df["resistance_ohm"] = df["voltage_V"] / df["current_A"]
    df["raw_index"] = np.arange(len(df))
    return df


def measurement_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["time_s"].round(9).astype(str)
        + "|"
        + df["temperature_K"].round(9).astype(str)
    )


def split_contiguous(indexes: np.ndarray) -> list[tuple[int, int]]:
    if len(indexes) == 0:
        return []
    spans = []
    start = previous = int(indexes[0])
    for value in indexes[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        spans.append((start, previous))
        start = previous = value
    spans.append((start, previous))
    return spans


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


def transition_bounds(temperature: np.ndarray, resistance: np.ndarray) -> tuple[float, float]:
    order = np.argsort(temperature)
    temperature = np.asarray(temperature, dtype=float)[order]
    resistance = np.asarray(resistance, dtype=float)[order]
    n_edge = max(5, len(temperature) // 10)
    low = float(np.nanmedian(resistance[:n_edge]))
    high = float(np.nanmedian(resistance[-n_edge:]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float(temperature.min()), float(temperature.max())
    normalized = (resistance - low) / (high - low)
    in_transition = (normalized >= 0.05) & (normalized <= 0.95)
    if np.count_nonzero(in_transition) < 5:
        return float(temperature.min()), float(temperature.max())
    left = max(float(temperature.min()), float(temperature[in_transition][0]) - 1.0)
    right = min(float(temperature.max()), float(temperature[in_transition][-1]) + 1.0)
    return (left, right) if right > left else (float(temperature.min()), float(temperature.max()))


def window_points(n_grid: int) -> int:
    points = int(round(SG_WINDOW_K / SG_GRID_STEP_K))
    if points % 2 == 0:
        points += 1
    minimum = SG_POLYORDER + 2
    if minimum % 2 == 0:
        minimum += 1
    points = max(points, minimum)
    if points > n_grid:
        points = n_grid if n_grid % 2 == 1 else n_grid - 1
    return points


def savgol_tc(temperature: pd.Series, resistance: pd.Series) -> float:
    temperature = np.asarray(temperature, dtype=float)
    resistance = np.asarray(resistance, dtype=float)
    ok = np.isfinite(temperature) & np.isfinite(resistance)
    temperature = temperature[ok]
    resistance = resistance[ok]
    if len(temperature) < SG_POLYORDER + 3:
        return float("nan")
    order = np.argsort(temperature)
    temperature = temperature[order]
    resistance = resistance[order]
    grid = np.arange(
        float(temperature.min()),
        float(temperature.max()) + SG_GRID_STEP_K / 2.0,
        SG_GRID_STEP_K,
    )
    uniform_resistance = np.interp(grid, temperature, resistance)
    derivative = savgol_filter(
        uniform_resistance,
        window_points(len(grid)),
        SG_POLYORDER,
        deriv=1,
        delta=SG_GRID_STEP_K,
        mode="interp",
    )
    left, right = transition_bounds(temperature, resistance)
    valid = np.isfinite(derivative) & (grid >= left) & (grid <= right)
    if not np.any(valid):
        return float("nan")
    valid_indexes = np.flatnonzero(valid)
    i_peak = int(valid_indexes[np.argmax(derivative[valid])])
    if 0 < i_peak < len(grid) - 1:
        x = grid[i_peak - 1 : i_peak + 2]
        y = derivative[i_peak - 1 : i_peak + 2]
        a, b, c = np.polyfit(x, y, deg=2)
        if a < 0:
            vertex = float(-b / (2.0 * a))
            if x[0] <= vertex <= x[-1]:
                return vertex
    return float(grid[i_peak])


def raw_summary() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.xlsx")):
        meta = parse_meta(path)
        df = load_raw(path)
        finite = (
            np.isfinite(df["time_s"])
            & np.isfinite(df["temperature_K"])
            & np.isfinite(df["voltage_V"])
            & np.isfinite(df["current_A"])
            & (df["current_A"] != 0)
        )
        in_window = finite & df["temperature_K"].between(T_MIN_K, T_MAX_K)
        indexes = df.index[in_window].to_numpy()
        spans = split_contiguous(indexes)
        dT = np.diff(df.loc[finite, "temperature_K"].to_numpy(dtype=float))
        if meta.direction == "heat":
            non_monotonic_steps = int(np.count_nonzero(dT < 0))
        elif meta.direction == "cool":
            non_monotonic_steps = int(np.count_nonzero(dT > 0))
        else:
            non_monotonic_steps = int(np.nan)

        sigma_r = sigma_resistance(
            df.loc[in_window, "voltage_V"].to_numpy(dtype=float),
            df.loc[in_window, "current_A"].to_numpy(dtype=float),
        )
        normal = df.loc[in_window & (df["temperature_K"] >= 100.0), "resistance_ohm"]
        median_normal = float(np.nanmedian(normal)) if len(normal) else float("nan")
        median_sigma = float(np.nanmedian(sigma_r)) if len(sigma_r) else float("nan")

        rows.append(
            {
                "source_file": path.name,
                "current_mA": meta.current_mA,
                "series_resistor": meta.series_resistor,
                "direction": meta.direction,
                "field_condition": meta.field_condition,
                "raw_rows": len(df),
                "finite_rows": int(finite.sum()),
                "temperature_min_K": float(df["temperature_K"].min()),
                "temperature_max_K": float(df["temperature_K"].max()),
                "rows_in_80_105K": int(in_window.sum()),
                "contiguous_80_105K_spans": ";".join(
                    f"{start}-{end}" for start, end in spans
                ),
                "non_monotonic_temperature_steps": non_monotonic_steps,
                "median_abs_dT_K": float(np.nanmedian(np.abs(dT))) if len(dT) else float("nan"),
                "median_sigma_R_mohm_80_105K": 1e3 * median_sigma,
                "median_R_mohm_above_100K": 1e3 * median_normal,
                "sigma_R_over_R_above_100K": median_sigma / median_normal
                if median_normal and np.isfinite(median_normal)
                else float("nan"),
                "objective_80_105K_savgol_tc_K": savgol_tc(
                    df.loc[in_window, "temperature_K"],
                    df.loc[in_window, "resistance_ohm"],
                ),
            }
        )
    return pd.DataFrame(rows)


def processed_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    omitted_rows = []

    for csv_path in sorted(MEAS_DIR.glob("partA_*.csv")):
        cleaned = pd.read_csv(csv_path)
        source_file = str(cleaned["source_file"].iloc[0])
        raw = load_raw(RAW_DIR / source_file)

        raw_keys = measurement_key(raw)
        cleaned_keys = set(measurement_key(cleaned))
        matched = raw_keys.isin(cleaned_keys).to_numpy()
        matched_indexes = np.flatnonzero(matched)

        in_window = (
            np.isfinite(raw["time_s"])
            & np.isfinite(raw["temperature_K"])
            & np.isfinite(raw["voltage_V"])
            & np.isfinite(raw["current_A"])
            & (raw["current_A"] != 0)
            & raw["temperature_K"].between(T_MIN_K, T_MAX_K)
        ).to_numpy()
        in_window_indexes = np.flatnonzero(in_window)
        omitted_in_window = [int(i) for i in in_window_indexes if not matched[i]]
        omitted_inside_bounds: list[int] = []
        if len(matched_indexes):
            lo = int(matched_indexes.min())
            hi = int(matched_indexes.max())
            omitted_inside_bounds = [i for i in range(lo, hi + 1) if not matched[i]]
        else:
            lo = hi = None

        for raw_index in omitted_in_window:
            row = raw.iloc[raw_index]
            omitted_rows.append(
                {
                    "measurement_id": csv_path.stem,
                    "source_file": source_file,
                    "raw_index": raw_index,
                    "inside_cleaned_index_span": raw_index in omitted_inside_bounds,
                    "time_s": row["time_s"],
                    "temperature_K": row["temperature_K"],
                    "voltage_V": row["voltage_V"],
                    "current_A": row["current_A"],
                    "resistance_ohm": row["resistance_ohm"],
                }
            )

        summary_rows.append(
            {
                "measurement_id": csv_path.stem,
                "source_file": source_file,
                "raw_rows": len(raw),
                "cleaned_rows": len(cleaned),
                "matched_cleaned_rows": int(matched.sum()),
                "cleaned_rows_not_matched_to_raw": int(len(cleaned) - matched.sum()),
                "raw_rows_in_80_105K": int(in_window.sum()),
                "raw_80_105K_rows_omitted_from_cleaned": len(omitted_in_window),
                "omitted_inside_cleaned_index_span": len(omitted_inside_bounds),
                "first_cleaned_raw_index": lo,
                "last_cleaned_raw_index": hi,
                "cleaned_raw_index_contiguous": bool(
                    len(matched_indexes)
                    and len(matched_indexes) == matched_indexes.max() - matched_indexes.min() + 1
                ),
                "cleaned_temperature_min_K": float(cleaned["temperature_K"].min()),
                "cleaned_temperature_max_K": float(cleaned["temperature_K"].max()),
                "cleaned_savgol_tc_K": savgol_tc(
                    cleaned["temperature_K"],
                    cleaned["resistance_ohm"],
                ),
                "objective_raw_80_105K_savgol_tc_K": savgol_tc(
                    raw.loc[in_window, "temperature_K"],
                    raw.loc[in_window, "resistance_ohm"],
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["objective_minus_cleaned_tc_K"] = (
        summary["objective_raw_80_105K_savgol_tc_K"] - summary["cleaned_savgol_tc_K"]
    )
    return summary, pd.DataFrame(omitted_rows)


def plot_overlays(raw_info: pd.DataFrame) -> None:
    processed_by_source = {
        pd.read_csv(path)["source_file"].iloc[0]: path
        for path in sorted(MEAS_DIR.glob("partA_*.csv"))
    }

    selected_sources = list(processed_by_source)
    excluded_sources = [
        name
        for name in raw_info["source_file"].tolist()
        if name not in processed_by_source
    ]

    def draw(paths: list[str], output: Path, title: str) -> None:
        if not paths:
            return
        ncols = 2
        nrows = int(np.ceil(len(paths) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
        for ax, source_file in zip(axes.ravel(), paths):
            raw = load_raw(RAW_DIR / source_file)
            ax.plot(
                raw["temperature_K"],
                1e3 * raw["resistance_ohm"],
                ".",
                ms=2.0,
                alpha=0.35,
                label="raw",
            )
            if source_file in processed_by_source:
                cleaned = pd.read_csv(processed_by_source[source_file])
                ax.plot(
                    cleaned["temperature_K"],
                    1e3 * cleaned["resistance_ohm"],
                    "o",
                    ms=2.8,
                    alpha=0.85,
                    label="cleaned",
                )
            ax.axvspan(T_MIN_K, T_MAX_K, color="#dddddd", alpha=0.2, lw=0)
            ax.set_title(source_file, fontsize=8)
            ax.set_xlabel("T (K)")
            ax.set_ylabel("R (mOhm)")
            ax.set_xlim(76, 106)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7)
        for ax in axes.ravel()[len(paths) :]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output, dpi=180)
        plt.close(fig)

    draw(selected_sources, OUT_DIR / "raw_vs_cleaned_part_a_sources.png", "Cleaned Part A sources")
    draw(excluded_sources, OUT_DIR / "excluded_raw_part_a_sources.png", "Raw Part A files not in cleaned CSV set")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = raw_summary()
    trim_summary, omitted = processed_audit()
    info.to_csv(OUT_DIR / "raw_part_a_file_summary.csv", index=False)
    trim_summary.to_csv(OUT_DIR / "processed_trim_audit.csv", index=False)
    omitted.to_csv(OUT_DIR / "omitted_raw_80_105K_rows.csv", index=False)
    plot_overlays(info)
    print(f"Wrote audit outputs to {OUT_DIR}")
    print(f"Raw Part A Excel files: {len(info)}")
    print(f"Cleaned Part A CSV files audited: {len(trim_summary)}")
    print(
        "Raw 80-105 K rows omitted from cleaned files:",
        int(trim_summary["raw_80_105K_rows_omitted_from_cleaned"].sum()),
    )


if __name__ == "__main__":
    main()
