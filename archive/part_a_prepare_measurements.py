# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "openpyxl",
# ]
# ///
"""Rebuild canonical Part A measurement CSVs from raw Excel files.

The rule is deliberately objective:
  1. read the source files listed in data/part_a/manifest.csv,
  2. compute four-probe resistance as R = V / I,
  3. keep finite rows with nonzero current in 80 <= T <= 105 K,
  4. keep the longest contiguous span matching the sweep,
  5. sort by temperature for downstream R(T) analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw_data"
PART_A_DIR = ROOT / "data" / "part_a"
MEAS_DIR = PART_A_DIR / "measurements"
MANIFEST = PART_A_DIR / "manifest.csv"

T_MIN_K = 80.0
T_MAX_K = 105.0


def contiguous_spans(indexes: np.ndarray) -> list[np.ndarray]:
    if len(indexes) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indexes) != 1) + 1
    return [span for span in np.split(indexes, breaks) if len(span)]


def timestamp_from_source(source_file: str) -> str:
    stem = Path(source_file).stem
    return stem.split("_")[-1]


def load_raw(source_file: str) -> pd.DataFrame:
    raw = pd.read_excel(RAW_DIR / source_file)
    df = raw.rename(
        columns={
            "t[sec]": "time_s",
            "T[K]": "temperature_K",
            "V[V]": "voltage_V",
            "I[A]": "current_A",
        }
    )[["time_s", "temperature_K", "voltage_V", "current_A"]].copy()
    df["resistance_ohm"] = df["voltage_V"] / df["current_A"]
    return df


def trimmed_measurement(manifest_row: pd.Series) -> pd.DataFrame:
    source_file = str(manifest_row["source_file"])
    raw = load_raw(source_file)
    finite = (
        np.isfinite(raw["time_s"])
        & np.isfinite(raw["temperature_K"])
        & np.isfinite(raw["voltage_V"])
        & np.isfinite(raw["current_A"])
        & np.isfinite(raw["resistance_ohm"])
        & (raw["current_A"] != 0)
    )
    in_window = finite & raw["temperature_K"].between(T_MIN_K, T_MAX_K)
    spans = contiguous_spans(np.flatnonzero(in_window.to_numpy()))
    if not spans:
        raise ValueError(f"{source_file} has no finite rows in {T_MIN_K}-{T_MAX_K} K")

    span = max(spans, key=len)
    df = raw.iloc[span].copy()
    df = df.sort_values("temperature_K").reset_index(drop=True)
    df["source_file"] = source_file
    df["sample_current_mA_nominal"] = float(manifest_row["sample_current_mA_nominal"])
    df["series_resistor"] = manifest_row["series_resistor"]
    df["direction"] = manifest_row["direction"]
    df["field_condition"] = manifest_row["field_condition"]
    df["timestamp"] = timestamp_from_source(source_file)
    return df[
        [
            "time_s",
            "temperature_K",
            "voltage_V",
            "current_A",
            "resistance_ohm",
            "source_file",
            "sample_current_mA_nominal",
            "series_resistor",
            "direction",
            "field_condition",
            "timestamp",
        ]
    ]


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    rebuilt_rows = []
    for _, row in manifest.sort_values("measurement_id").iterrows():
        df = trimmed_measurement(row)
        output = PART_A_DIR / str(row["data_file"])
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        rebuilt_rows.append(
            {
                **row.to_dict(),
                "temperature_min_K": float(df["temperature_K"].min()),
                "temperature_max_K": float(df["temperature_K"].max()),
                "points": len(df),
            }
        )

    rebuilt_manifest = pd.DataFrame(rebuilt_rows)[manifest.columns]
    rebuilt_manifest.to_csv(MANIFEST, index=False)
    print(f"Rebuilt {len(rebuilt_manifest)} Part A measurement files")


if __name__ == "__main__":
    main()
