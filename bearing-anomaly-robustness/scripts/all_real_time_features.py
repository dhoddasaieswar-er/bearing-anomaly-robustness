import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis

print("=" * 80)
print("PADERBORN DATASET — STEP 58 ALL-BEARING TIME-DOMAIN FEATURES")
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
        # Vibration channel
        # ----------------------------------------------------------

        vibration_channel = data.Y[6]

        signal = np.asarray(
            vibration_channel.Data,
            dtype=float
        ).squeeze()

        # ----------------------------------------------------------
        # Time-domain features
        # ----------------------------------------------------------

        rms = np.sqrt(np.mean(signal ** 2))

        std = np.std(signal)

        peak = np.max(np.abs(signal))

        peak_to_peak = np.ptp(signal)

        kurt = kurtosis(
            signal,
            fisher=False
        )

        crest_factor = peak / rms

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
        # Label
        # ----------------------------------------------------------

        # K = healthy
        # KA / KI = damaged

        if bearing.startswith("K") and not (
            bearing.startswith("KA") or
            bearing.startswith("KI")
        ):
            label = 0
        else:
            label = 1

        records.append({
            "Bearing": bearing,
            "Condition": condition,
            "Label": label,
            "RMS": rms,
            "STD": std,
            "Peak": peak,
            "Peak_to_Peak": peak_to_peak,
            "Kurtosis": kurt,
            "Crest_Factor": crest_factor,
            "File": filename
        })

# ----------------------------------------------------------
# Create dataframe
# ----------------------------------------------------------

df = pd.DataFrame(records)

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("\nShape:", df.shape)

print("\nBearings:")
print(df["Bearing"].value_counts().sort_index())

print("\nLabels:")
print(df["Label"].value_counts().sort_index())

print("\nConditions:")
print(df["Condition"].value_counts().sort_index())

print("\nLabel by bearing:")
print(
    df.groupby("Bearing")["Label"]
    .first()
)

print("\nSaved:")
print("all_real_time_features.csv")

df.to_csv(
    "all_real_time_features.csv",
    index=False
)