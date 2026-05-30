# Superconductivity Data

This directory contains the processed input data used by the analysis notebooks.

- `raw_data/*.xlsx`: original lab-export workbooks for the Part A resistance
  runs.
- `raw_data/part_b/*.xlsx`: original lab-export workbooks for the Part B
  Hall-probe measurements.
- `part_a/measurements/*.csv`: accepted resistance-vs-temperature runs, sorted by `temperature_K`, deduplicated, and clipped to the common `80-105 K` transition window.
- `part_a/manifest.csv`: compact index of the Part A measurement files.
- `part_b/coil_calibration.csv`: cleaned empty-coil Hall calibration measurements.
- `part_b/disk_BH.csv`: cleaned superconducting-disk Hall measurements.

Regenerated analysis outputs are written under `analysis/results/`, not here.
