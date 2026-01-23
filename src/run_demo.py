import os
import re
import math
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


PRECIP_CSV = "data/precip_10min_long_balanced.csv"
STATION_CSV = "data/Station_Features_Lookup.csv"
RADAR_DIR = "radar"


def list_radar_files(radar_root: str = RADAR_DIR) -> list[str]:
    radar_root_path = Path(radar_root)
    if not radar_root_path.exists():
        raise RuntimeError(f"Radar root folder does not exist: {radar_root_path.resolve()}")

    # Your files end with "-hd5" (no dot), and live in nested folders (e.g., radar/07/07/...).
    patterns = ["*hd5", "*.h5", "*.HDF5", "*.hdf5"]

    files: list[Path] = []
    for pat in patterns:
        files.extend(radar_root_path.rglob(pat))

    files = sorted(set(files))
    return [str(p) for p in files]


def parse_radar_time_from_filename(path_str: str) -> datetime:
    # Example: ...-20210707000005700-...-hd5  -> we want the 14 digits: 20210707000005
    m = re.search(r"-(\d{14})", path_str)
    if not m:
        raise ValueError(f"Cannot parse timestamp from filename/path: {path_str}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def read_dbzh_dbz(radar_path: str) -> np.ndarray:
    with h5py.File(radar_path, "r") as f:
        raw = f["dataset1/data1/data"][...].astype(np.float32)

        what_attrs = f["dataset1/data1/what"].attrs
        gain = float(what_attrs["gain"])
        offset = float(what_attrs["offset"])
        nodata = float(what_attrs["nodata"])

        raw[raw == nodata] = np.nan
        dbz = offset + gain * raw
        return dbz


def radar_features_from_dbz(dbz: np.ndarray) -> dict:
    finite = dbz[np.isfinite(dbz)]
    if finite.size == 0:
        return {"dbz_mean": np.nan, "dbz_max": np.nan, "dbz_p90": np.nan}
    return {
        "dbz_mean": float(np.nanmean(dbz)),
        "dbz_max": float(np.nanmax(dbz)),
        "dbz_p90": float(np.nanpercentile(finite, 90)),
    }


def match_nearest_radar(label_time: datetime, radar_times: list[datetime], max_minutes: float = 10.0):
    best_idx = None
    best_dt = float("inf")

    for i, t in enumerate(radar_times):
        dt_min = abs((t - label_time).total_seconds()) / 60.0
        if dt_min < best_dt:
            best_dt = dt_min
            best_idx = i

    if best_idx is not None and best_dt <= max_minutes:
        return best_idx, best_dt
    return None, None


def main() -> None:
    precip = pd.read_csv(PRECIP_CSV)
    precip["Timestamp_dt"] = pd.to_datetime(precip["Timestamp"])

    _stations = pd.read_csv(STATION_CSV)

    station_id = "NMS0101"
    precip_s = precip[precip["Station"] == station_id].copy()

    radar_files = list_radar_files(RADAR_DIR)
    print(f"Found radar files: {len(radar_files)}")
    print("Example radar file:", radar_files[0] if radar_files else "NONE")

    if len(radar_files) == 0:
        raise RuntimeError("No radar files found in radar/ folder.")

    radar_times = [parse_radar_time_from_filename(f) for f in radar_files]

    rows = []
    max_minutes = 10.0

    for _, r in precip_s.iterrows():
        t_label = r["Timestamp_dt"].to_pydatetime()
        idx, dt_min = match_nearest_radar(t_label, radar_times, max_minutes=max_minutes)
        if idx is None:
            continue

        radar_path = radar_files[idx]
        dbz = read_dbzh_dbz(radar_path)
        feats = radar_features_from_dbz(dbz)

        rows.append(
            {
                "timestamp": t_label,
                "radar_file": os.path.basename(radar_path),
                "dt_minutes": float(dt_min),
                "y_precip": float(r["Precipitation"]),
                **feats,
            }
        )

    df = pd.DataFrame(rows)

    print("Paired samples:", len(df))
    if len(df) == 0:
        raise RuntimeError("No paired samples found. Check station_id, timestamps, and max_minutes.")

    X = df[["dbz_mean", "dbz_max", "dbz_p90"]].values
    y = df["y_precip"].values

    if len(df) < 5:
        raise RuntimeError("Not enough paired samples. Add more radar files around the label times.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    print("MAE:", mae)
    print("RMSE:", rmse)

    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("True precipitation")
    plt.ylabel("Predicted precipitation")
    plt.title("True vs Pred (Linear Regression)")
    plt.savefig("outputs/fig_true_vs_pred.png", dpi=200, bbox_inches="tight")
    plt.close()

    errors = y_pred - y_test
    plt.figure()
    plt.hist(errors, bins=10)
    plt.xlabel("Prediction error (pred - true)")
    plt.ylabel("Count")
    plt.title("Error distribution")
    plt.savefig("outputs/fig_error_hist.png", dpi=200, bbox_inches="tight")
    plt.close()

    sample_path = radar_files[0]
    sample_dbz = read_dbzh_dbz(sample_path)

    plt.figure()
    plt.imshow(sample_dbz, aspect="auto")
    plt.colorbar(label="dBZ")
    plt.title(f"Example DBZH (dBZ): {os.path.basename(sample_path)}")
    plt.savefig("outputs/fig_radar_example.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved figures in outputs/:")
    print(" - outputs/fig_radar_example.png")
    print(" - outputs/fig_true_vs_pred.png")
    print(" - outputs/fig_error_hist.png")


if __name__ == "__main__":
    main()