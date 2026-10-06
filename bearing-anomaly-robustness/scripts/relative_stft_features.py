import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import stft

print("=" * 80)
print("PADERBORN DATASET — STEP 62 RELATIVE-FREQUENCY STFT FEATURES")
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
        # Sampling frequency
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
        # Operating condition
        # ----------------------------------------------------------

        if filename.startswith("N09_M07_F10"):
            condition = "N09_M07_F10"
            rpm = 900.0

        elif filename.startswith("N15_M01_F10"):
            condition = "N15_M01_F10"
            rpm = 1500.0

        elif filename.startswith("N15_M07_F04"):
            condition = "N15_M07_F04"
            rpm = 1500.0

        elif filename.startswith("N15_M07_F10"):
            condition = "N15_M07_F10"
            rpm = 1500.0

        else:
            condition = "Unknown"
            rpm = np.nan

        # Rotational frequency in Hz
        rotation_hz = rpm / 60.0

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
        # Convert frequency to orders
        #
        # order = frequency / rotational frequency
        # ----------------------------------------------------------

        order = f / rotation_hz

        # ----------------------------------------------------------
        # Relative/order bands
        # ----------------------------------------------------------

        bands = {
            "Order_0_20": (0, 20),
            "Order_20_50": (20, 50),
            "Order_50_100": (50, 100),
            "Order_100_200": (100, 200),
            "Order_200_400": (200, 400)
        }

        features = {}

        for band_name, (low, high) in bands.items():

            mask = (
                (order >= low) &
                (order < high)
            )

            band_power = power[mask, :]

            features[
                f"{band_name}_Mean"
            ] = np.mean(band_power)

            features[
                f"{band_name}_Max"
            ] = np.max(band_power)

        # ----------------------------------------------------------
        # Overall normalized STFT statistics
        # ----------------------------------------------------------

        features["Relative_STFT_Mean"] = np.mean(power)
        features["Relative_STFT_Max"] = np.max(power)

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
        # Save metadata
        # ----------------------------------------------------------

        features["Bearing"] = bearing
        features["Condition"] = condition
        features["RPM"] = rpm
        features["Rotation_Hz"] = rotation_hz
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

print("\nOperating conditions:")
print(
    df["Condition"]
    .value_counts()
    .sort_index()
)

print("\nRotation frequencies:")
print(
    df.groupby("Condition")["Rotation_Hz"]
    .first()
)

print("\nFeature columns:")
print(df.columns.tolist())

df.to_csv(
    "relative_stft_features.csv",
    index=False
)

print("\nSaved:")
print("relative_stft_features.csv")