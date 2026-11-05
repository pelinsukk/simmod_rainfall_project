"""Estimate station rainfall from preceding radar reflectivity scans."""
import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = ["mean_dbz", "max_dbz", "p90_dbz"]


def radar_index(root):
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.lower().endswith(("-hd5", ".hd5", ".h5", ".hdf5"))
    )
    rows = []
    for path in paths:
        match = re.search(r"-(\d{14})", path.name)
        if not match:
            raise ValueError(f"No radar timestamp in filename: {path.name}")
        rows.append((pd.to_datetime(match[1], format="%Y%m%d%H%M%S", utc=True), path))
    if not rows:
        raise ValueError(f"No radar files in {root}; see README for the data layout.")
    return pd.DataFrame(rows, columns=["scan_time", "path"]).sort_values("scan_time", kind="stable")


def read_dbzh(path):
    with h5py.File(path, "r") as f:
        group = f["dataset1/data1"]
        attrs = group["what"].attrs
        quantity = attrs.get("quantity", "DBZH")
        if isinstance(quantity, bytes):
            quantity = quantity.decode()
        if quantity != "DBZH":
            raise ValueError(f"{path.name}: expected DBZH, got {quantity}")
        raw = group["data"][...].astype(float)
        for key in ("nodata", "undetect"):
            if key in attrs:
                raw[raw == float(attrs[key])] = np.nan
        return raw * float(attrs["gain"]) + float(attrs["offset"])


def extract(path):
    values = read_dbzh(path)
    valid = values[np.isfinite(values)]
    return (valid.mean(), valid.max(), np.percentile(valid, 90)) if valid.size else (np.nan,) * 3


def pair(labels, scans, minutes):
    return pd.merge_asof(
        labels.sort_values("timestamp"), scans, left_on="timestamp", right_on="scan_time",
        direction="backward", tolerance=pd.Timedelta(minutes=minutes),
    ).dropna(subset=["path"])


def chronological_split(pairs):
    if len(pairs) < 8:
        raise ValueError("Need at least eight usable pairs for a chronological holdout.")
    split = int(len(pairs) * .75)
    test = pairs.iloc[split:]
    # Exclude shared scans and overlapping 10-minute target intervals at the boundary.
    cutoff = test["timestamp"].iloc[0] - pd.Timedelta(minutes=10)
    train = pairs.iloc[:split]
    train = train.loc[(train["timestamp"] <= cutoff) & (train["scan_time"] < test["scan_time"].min())]
    if len(train) < 4 or len(test) < 2:
        raise ValueError("Too few independent pairs after the holdout boundary purge.")
    return train, test


def synthetic_inputs(root):
    """Generate deterministic toy HDF5 scans and gauge targets; no observations."""
    rng = np.random.default_rng(19)
    radar = root / "radar"
    radar.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2025-01-01", periods=120, freq="10min", tz="UTC")
    signal = 18 + 11 * np.sin(np.arange(len(times)) / 5) + rng.normal(0, 2, len(times))
    rain = np.maximum(0, .09 * (signal - 12) + rng.normal(0, .12, len(times)))
    for time, level in zip(times, signal):
        stamp = (time - pd.Timedelta(minutes=2)).strftime("%Y%m%d%H%M%S")
        raw = np.clip(np.rint((level + rng.normal(0, 3, (16, 16)) + 32) * 2), 1, 254).astype("uint8")
        raw[0, 0] = 255
        with h5py.File(radar / f"synthetic-{stamp}-hd5", "w") as f:
            group = f.create_group("dataset1/data1")
            group.create_dataset("data", data=raw)
            group.create_group("what").attrs.update(
                {"quantity": "DBZH", "gain": .5, "offset": -32., "nodata": 255, "undetect": 0}
            )
    labels = root / "labels.csv"
    pd.DataFrame({"Timestamp": times, "Station": "DEMO", "Precipitation": rain}).to_csv(labels, index=False)
    return radar, labels, "DEMO"


def run(radar, csv, station, out, max_minutes, synthetic=False):
    labels = pd.read_csv(csv, dtype={"Station": "string"})
    missing = {"Timestamp", "Station", "Precipitation"} - set(labels)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    labels = labels.loc[labels["Station"].str.strip().eq(station)].copy()
    labels["timestamp"] = pd.to_datetime(labels["Timestamp"], utc=True, format="mixed", errors="coerce")
    labels["rain"] = pd.to_numeric(labels["Precipitation"], errors="coerce")
    labels = labels.loc[labels["timestamp"].notna() & np.isfinite(labels["rain"]) & labels["rain"].ge(0)]
    if labels["timestamp"].duplicated().any():
        raise ValueError("Duplicate target timestamps for the selected station.")
    if labels.empty:
        raise ValueError(f"No valid rainfall labels for station {station}.")
    scans = radar_index(radar)
    pairs = pair(labels, scans, max_minutes)
    if pairs.empty:
        raise ValueError("No targets matched a preceding radar scan within the tolerance.")
    cache = {path: extract(path) for path in pairs["path"].unique()}
    pairs[FEATURES] = np.array([cache[p] for p in pairs["path"]])
    pairs = pairs.dropna(subset=FEATURES)
    train, test = chronological_split(pairs)
    model = LinearRegression().fit(train[FEATURES], train["rain"])
    actual = test["rain"].to_numpy()
    predictions = np.maximum(0, model.predict(test[FEATURES]))
    baseline = np.full(len(test), train["rain"].median())

    def score(values):
        return {"mae_mm": float(mean_absolute_error(actual, values)),
                "rmse_mm": float(np.sqrt(mean_squared_error(actual, values)))}

    def period(frame):
        return {"rows": len(frame), "start_utc": frame["timestamp"].min().isoformat(),
                "end_utc": frame["timestamp"].max().isoformat()}

    report = {"dataset": "synthetic" if synthetic else "local observations",
              "station": station, "scans": len(scans), "valid_targets": len(labels),
              "usable_pairs": len(pairs), "max_lag_minutes": max_minutes,
              "train": period(train), "test": period(test),
              "purged_rows": len(pairs) - len(train) - len(test),
              "linear_regression": score(predictions), "training_median": score(baseline)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(test["timestamp"], actual, label="Target")
    ax.plot(test["timestamp"], predictions, label="Linear regression")
    ax.plot(test["timestamp"], baseline, "--", label="Training median")
    ax.set(ylabel="10-minute rainfall [mm]", xlabel="UTC", title=f"{report['dataset'].capitalize()} holdout")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "holdout.png", dpi=150)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--radar-dir", type=Path, default=Path("radar"))
    parser.add_argument("--precip-csv", type=Path, default=Path("data/precip_10min_long_balanced.csv"))
    parser.add_argument("--station", default="NMS0101")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-lag-minutes", type=float, default=10)
    args = parser.parse_args()
    if not np.isfinite(args.max_lag_minutes) or args.max_lag_minutes < 0:
        parser.error("--max-lag-minutes must be finite and nonnegative")
    try:
        if args.synthetic:
            radar, csv, station = synthetic_inputs(args.out_dir / "synthetic_inputs")
        else:
            radar, csv, station = args.radar_dir, args.precip_csv, args.station
        print(json.dumps(run(radar, csv, station, args.out_dir, args.max_lag_minutes, args.synthetic), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
