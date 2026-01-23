# SimMod Rainfall Prediction (Radar -> Ground Precip)

## What this project does
Loads radar reflectivity (DBZH) files and predicts ground precipitation as a baseline demo.

## Project structure
- data/: input CSVs (labels + station metadata)
- radar/: radar .hd5 files (NOT committed)
- src/: code
- outputs/: generated figures (NOT committed)

## Inputs
Put these into `data/`:
- precip_10min_long_balanced.csv
- Station_Features_Lookup.csv

Put radar `.hd5` files into `radar/`.

## Run
```bash
python src/run_demo.py
