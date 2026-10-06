import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import stft

print("=" * 80)
print("PADERBORN DATASET — STEP 60 ALL-BEARING STFT FEATURES")
print("=" * 80)

ROOT = Path(__file__).resolve().parent

bearing_folders = [
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
    "KA01",
    "KA03",
    "KA04",
    "KI05"
]

records = []

for bearing in bearing_folders:

    files = sorted(
        glob.glob(os.path.join(ROOT, bearing, "*.mat"))
    )

    print(f"\nProcessing {bearing}: {len(files)} files")

    for file_path in files:

        filename = os.path.basename(file_path)

        mat = loadmat(
            file_path,
            struct_as_record=False,
            squeeze_me=True
        )

        variable_name = [
            key for key in mat.keys()
            if not key.startswith("__")
        ][0]

        data = mat[variable_name]

        # ----------------------------------------------------------
        # Vibration signal
        # ----------------------------------------------------------

        vibration_channel = data.Y[6]

        signal = np.asarray(
            vibration_channel.Data,
            dtype=float
        ).squeeze()

        # ----------------------------------------------------------
        # Actual sampling frequency
        # ----------------------------------------------------------

        vibration_time = np.asarray(
            data.X[1].Data,
            dtype=float
        ).squeeze()

        dt = np.median(
            np.diff(vibration_time)
        )

        fs = 1.0 / dt

        # ----------------------------------------------------------
        # STFT
        # ----------------------------------------------------------

        f, t, Zxx = stft(
            signal,
            fs=fs,
            nperseg=4096,
            noverlap=3072,
            window="hann"
        )

        power = np.abs(Zxx) ** 2

        # ----------------------------------------------------------
        # Frequency bands
        # ----------------------------------------------------------

        bands = {
            "Band_0_1000Hz": (0, 1000),
            "Band_1000_3000Hz": (1000, 3000),
            "Band_3000_5000Hz": (3000, 5000),
            "Band_5000_10000Hz": (5000, 10000),
            "Band_10000_20000Hz": (10000, 20000)
        }

        features = {}

        for band_name, (low, high) in bands.items():

            mask = (
                (f >= low) &
                (f < high)
            )

            band_power = power[mask, :]

            features[
                f"{band_name}_Mean"
            ] = np.mean(band_power)

            features[
                f"{band_name}_Max"
            ] = np.max(band_power)

        features["STFT_Mean"] = np.mean(power)
        features["STFT_Max"] = np.max(power)

        # ----------------------------------------------------------
        # Label
        # ----------------------------------------------------------

        if bearing.startswith("K") and not (
            bearing.startswith("KA") or
            bearing.startswith("KI")
        ):
            label = 0
        else:
            label = 1

        # ----------------------------------------------------------
        # Operating condition
        # ----------------------------------------------------------

        if filename.startswith("N09_M07_F10"):
            condition = "N09_M07_F10"

        elif filename.startswith("N15_M01_F10"):
            condition = "N15_M01_F10"

        elif filename.startswith("N15_M07_F04"):
            condition = "N15_M07_F04"

        elif filename.startswith("N15_M07_F10"):
            condition = "N15_M07_F10"

        else:
            condition = "Unknown"

        # ----------------------------------------------------------
        # Save record
        # ----------------------------------------------------------

        features["Bearing"] = bearing
        features["Condition"] = condition
        features["Label"] = label
        features["File"] = filename

        records.append(features)

# ----------------------------------------------------------
# DataFrame
# ----------------------------------------------------------

df = pd.DataFrame(records)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("\nShape:", df.shape)

print("\nBearings:")
print(
    df["Bearing"]
    .value_counts()
    .sort_index()
)

print("\nLabels:")
print(
    df["Label"]
    .value_counts()
    .sort_index()
)

print("\nConditions:")
print(
    df["Condition"]
    .value_counts()
    .sort_index()
)

print("\nFeature columns:")
print(df.columns.tolist())

df.to_csv(
    "all_real_stft_features.csv",
    index=False
)

print("\nSaved:")
print("all_real_stft_features.csv")
