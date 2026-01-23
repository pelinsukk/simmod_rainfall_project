# SimMod Rainfall Prediction (Radar Reflectivity DBZH → 10-min Ground Precipitation) — Group Project Baseline Demo

## Project Overview

This is a baseline demo pipeline for our SimMod group project. The goal is end-to-end validation: load radar files → match timestamps → extract simple features → train baseline regression → save plots/logs. It is not meant to be a final model, just a reproducible baseline.

## Project Goal

- **Input**: Weather radar reflectivity DBZH snapshots over Cologne region, stored as HDF5 `*.hd5` files under `radar/`
- **Target**: 10-minute ground precipitation measured at stations, provided as CSV labels
- **Task**: Supervised regression (predict precipitation intensity from radar information)

## What is Implemented (Current Baseline)

- Recursively scans `radar/` for files ending with `*.hd5`
- Parses timestamps from radar filenames
- Matches each precipitation timestamp to the nearest radar scan (within ±10 minutes)
- Reads DBZH values from the radar HDF5 structure and converts them to dBZ using gain/offset metadata
- Extracts simple global radar features from each radar frame:
  - mean DBZH
  - max DBZH
  - 90th percentile DBZH
- Trains a baseline regression model (`LinearRegression`)
- Saves a run log and diagnostic plots into `outputs/`

## Data Layout (Local)

Raw radar data is large and therefore not meant to be committed to GitHub. Users should keep `radar/` locally and the repo should stay lightweight.

Expected local folder structure:

```text
radar/
  2021/
    07/
      07/
        .../*.hd5
    08/
      .../*.hd5
    ...
  2022/
    01/
      .../*.hd5
    ...
data/
  precip_10min_long_balanced.csv
  Station_Features_Lookup.csv
outputs/
  [created at runtime]
```

## CSV Files
- `data/precip_10min_long_balanced.csv`: Label dataset with 10-minute precipitation values and timestamps (used as target `y`)
- `data/Station_Features_Lookup.csv`: Station lookup / metadata mapping table used by the pipeline


## How to Run the Demo
From the repo root:

```bash
python --version
pip install -r requirements.txt
mkdir -p outputs
python src/run_demo.py > outputs/run_log.txt 2>&1
tail -n 60 outputs/run_log.txt
```

## Output Files

The demo saves outputs into `outputs/`:

- `outputs/run_log.txt`: Run summary (radar file count, paired sample count, MAE/RMSE, etc.)
- `outputs/fig_radar_example.png`: Example radar DBZH image (sanity check)
- `outputs/fig_true_vs_pred.png`: Scatter plot of true vs predicted precipitation (baseline model)
- `outputs/fig_error_hist.png`: Error histogram (pred - true)


## Baseline Results
Metrics are sanity-check values, not final performance.
The baseline uses only global summary statistics, so it is expected to be limited.

Success at this stage means:
- the pipeline runs end-to-end
- timestamp matching works reliably
- feature extraction is consistent
- outputs are reproducible


## Notes / Next Steps
Potential improvements for the next iteration:

- Better spatial features around station locations (pooling/aggregation, not global stats)
- Multi-elevation / vertical structure features
- Stronger baselines (Ridge/Lasso, tree-based models, neural networks)
- Proper time-based train/val/test split to avoid leakage
- Scale evaluation to longer spans (full year / multi-year)


## Contributors
- Beshoy Hanna Ayaad Labib (blabib@smail.uni-koeln.de)
- Fariba Yazdanjooei (fyazdanj@smail.uni-koeln.de)
- Mohammed Fawaz Nawaz (mnawaz@smail.uni-koeln.de)
- Pelin Su Kaplan (pkaplan@smail.uni-koeln.de)
- Raj Nandini Singh (rsingh8@smail.uni-koeln.de)
- Udayan Mishra (umishra@smail.uni-koeln.de)

## License
Educational / university course project, no commercial use intended.

## Citation / References
Radar reflectivity (DBZH) and station precipitation data are course-provided datasets.
If the instructors provide an official dataset citation/link, it will be added here.


## Known Limitations / Assumptions
- Only global radar summary statistics (mean/max/p90), ignores spatial structure
- Radar → precipitation mapping is highly non-linear and depends on physical factors not represented in baseline
- Nearest-neighbor timestamp matching with fixed ±10 minute window may introduce mismatch
- Results depend on locally available radar files and paired sample count


## Reproducibility Notes
- Dependencies listed in `requirements.txt`
- Keep `radar/` out of GitHub
- Store run outputs in `outputs/` and keep `outputs/run_log.txt` for exact run record



