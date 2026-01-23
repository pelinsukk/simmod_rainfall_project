# SimMod Rainfall Prediction (Radar → Ground Precip) — Demo Baseline

Quick working demo for our SimMod project: we use **radar reflectivity (DBZH)** above Cologne and try to predict **10-minute ground precipitation** at a station.

This repo is mainly meant to prove the pipeline runs end-to-end (load data → match timestamps → extract features → fit a baseline model → save plots).

---

## What the demo does

- Scans `radar/` recursively and collects radar files (supports filenames ending with `-hd5`)
- Reads 10-minute precipitation labels from `data/precip_10min_long_balanced.csv`
- Matches each label timestamp to the nearest radar scan (within ±10 minutes)
- Extracts a few simple global radar features:
  - mean DBZH
  - max DBZH
  - 90th percentile DBZH
- Trains a baseline `LinearRegression`
- Saves outputs into `outputs/`

---

## Data layout

Radar files are stored locally under:

```text
radar/
  07/
  08/
  09/
  10/
  11/
  12/
