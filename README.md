# SimMod Rainfall Prediction (Radar -> Ground Precip)

This project is a small end-to-end baseline that connects weather radar reflectivity (DBZH) to ground precipitation measurements.

The goal is to build a clean, runnable pipeline:
- load radar files
- pair them with precipitation timestamps
- extract simple radar features
- train a first regression model
- generate quick diagnostic plots

---

## What this project does

- Reads **DBZH radar `.hd5` files** (reflectivity field)
- Matches radar timestamps to **10-minute precipitation labels**
- Extracts simple radar features (mean / max / percentile reflectivity)
- Trains a baseline **Linear Regression** model
- Prints basic metrics (MAE / RMSE)
- Saves diagnostic plots into `outputs/`

---

## Project structure

- `data/`
  - `precip_10min_long_balanced.csv` (precipitation labels)
  - `Station_Features_Lookup.csv` (station metadata / lookup)
- `radar/`
  - radar `.hd5` files (**NOT committed**)
- `src/`
  - `run_demo.py` (main runnable demo script)
- `outputs/`
  - generated plots (**NOT committed**)

---

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt