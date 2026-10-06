import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import stft

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# STEP 84
# CORRECTED STFT ANALYSIS USING NOMINAL fs = 64000 Hz
# ============================================================

ROOT = Path(os.environ.get("BEARING_DATA_ROOT", Path(__file__).resolve().parent))

BEARINGS = [
    "K001", "K002", "K003", "K004", "K005", "K006",
    "KA01", "KA03", "KA04", "KI05"
]

HEALTHY = {
    "K001", "K002", "K003",
    "K004", "K005", "K006"
}

DAMAGED = {
    "KA01", "KA03", "KA04", "KI05"
}

# ------------------------------------------------------------
# Fixed experimental settings
# ------------------------------------------------------------

FS = 64000.0

NPERSEG = 4096
NOVERLAP = 3072
WINDOW = "hann"

N_ESTIMATORS = 300
RANDOM_STATE = 42

# Frequency bands in Hz
FREQUENCY_BANDS = [
    ("Band_0_1000Hz", 0, 1000),
    ("Band_1000_3000Hz", 1000, 3000),
    ("Band_3000_5000Hz", 3000, 5000),
    ("Band_5000_10000Hz", 5000, 10000),
    ("Band_10000_20000Hz", 10000, 20000),
]


# ============================================================
# Helper functions
# ============================================================

def get_condition(filename):
    """
    Extract operating condition from the Paderborn filename.
    Example:
    N09_M07_F10_K001_1.mat
    -> N09_M07_F10
    """

    parts = os.path.basename(filename).split("_")

    return "_".join(parts[:3])


def get_rpm(condition):
    """
    Extract rotational speed from the operating condition.
    """

    if condition.startswith("N09"):
        return 900.0

    if condition.startswith("N15"):
        return 1500.0

    raise ValueError(
        f"Unknown rotational-speed condition: {condition}"
    )


def load_vibration(filepath):
    """
    Load the actual vibration signal.

    IMPORTANT:
    The vibration channel is Y[6].
    The time vector is NOT used to estimate fs here.
    """

    mat = loadmat(
        filepath,
        struct_as_record=False,
        squeeze_me=True
    )

    variable_names = [
        key for key in mat.keys()
        if not key.startswith("__")
    ]

    if len(variable_names) != 1:
        raise RuntimeError(
            f"Unexpected MAT structure in {filepath}"
        )

    data = mat[variable_names[0]]

    vibration_channel = data.Y[6]

    signal = np.asarray(
        vibration_channel.Data,
        dtype=float
    ).squeeze()

    if signal.ndim != 1:
        raise RuntimeError(
            f"Unexpected signal shape: {signal.shape}"
        )

    return signal


def extract_absolute_stft_features(signal):
    """
    Absolute-frequency STFT features.
    """

    f, t, Zxx = stft(
        signal,
        fs=FS,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        window=WINDOW
    )

    power = np.abs(Zxx) ** 2

    features = {}

    for name, low, high in FREQUENCY_BANDS:

        mask = (f >= low) & (f < high)

        if not np.any(mask):
            raise RuntimeError(
                f"No STFT bins found for {name}"
            )

        band_power = power[mask, :]

        features[f"{name}_Mean"] = np.mean(band_power)
        features[f"{name}_Max"] = np.max(band_power)

    features["STFT_Mean"] = np.mean(power)
    features["STFT_Max"] = np.max(power)

    return features


def extract_order_stft_features(signal, rpm):
    """
    Order-normalized STFT features.

    Order = frequency / rotational frequency.

    900 rpm  -> 15 Hz
    1500 rpm -> 25 Hz
    """

    rotation_hz = rpm / 60.0

    f, t, Zxx = stft(
        signal,
        fs=FS,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        window=WINDOW
    )

    power = np.abs(Zxx) ** 2

    order = f / rotation_hz

    order_bands = [
        ("Order_0_20", 0, 20),
        ("Order_20_50", 20, 50),
        ("Order_50_100", 50, 100),
        ("Order_100_200", 100, 200),
        ("Order_200_400", 200, 400),
    ]

    features = {}

    for name, low, high in order_bands:

        mask = (order >= low) & (order < high)

        if not np.any(mask):
            raise RuntimeError(
                f"No STFT bins found for {name}"
            )

        band_power = power[mask, :]

        features[f"{name}_Mean"] = np.mean(band_power)
        features[f"{name}_Max"] = np.max(band_power)

    features["Relative_STFT_Mean"] = np.mean(power)
    features["Relative_STFT_Max"] = np.max(power)

    return features


def extract_rms_normalized_stft_features(signal):
    """
    True per-signal RMS normalization.

    signal_norm = signal / RMS(signal)
    """

    rms = np.sqrt(
        np.mean(signal ** 2)
    )

    if rms <= 0:
        raise RuntimeError(
            "RMS is zero or negative."
        )

    normalized_signal = signal / rms

    return extract_absolute_stft_features(
        normalized_signal
    )


# ============================================================
# 1. Extract all features
# ============================================================

print("=" * 75)
print("STEP 84 — CORRECTED 64 kHz STFT EXPERIMENT")
print("=" * 75)

print("\nExperimental settings:")
print(f"Sampling rate: {FS} Hz")
print(f"nperseg:       {NPERSEG}")
print(f"noverlap:      {NOVERLAP}")
print(f"window:        {WINDOW}")
print(f"RF trees:      {N_ESTIMATORS}")
print(f"random_state:  {RANDOM_STATE}")

all_absolute = []
all_order = []
all_rms = []

file_count = 0

for bearing in BEARINGS:

    folder = os.path.join(ROOT, bearing)

    files = sorted(
        glob.glob(
            os.path.join(folder, "*.mat")
        )
    )

    print(
        f"\nProcessing {bearing}: "
        f"{len(files)} files"
    )

    label = 0 if bearing in HEALTHY else 1

    for filepath in files:

        filename = os.path.basename(filepath)

        condition = get_condition(filename)
        rpm = get_rpm(condition)
        rotation_hz = rpm / 60.0

        signal = load_vibration(filepath)

        # --------------------------------------------
        # Absolute STFT
        # --------------------------------------------

        absolute_features = extract_absolute_stft_features(
            signal
        )

        absolute_row = {
            "Bearing": bearing,
            "Condition": condition,
            "RPM": rpm,
            "Rotation_Hz": rotation_hz,
            "Label": label,
            "File": filename,
        }

        absolute_row.update(
            absolute_features
        )

        all_absolute.append(
            absolute_row
        )

        # --------------------------------------------
        # Order-normalized STFT
        # --------------------------------------------

        order_features = extract_order_stft_features(
            signal,
            rpm
        )

        order_row = {
            "Bearing": bearing,
            "Condition": condition,
            "RPM": rpm,
            "Rotation_Hz": rotation_hz,
            "Label": label,
            "File": filename,
        }

        order_row.update(
            order_features
        )

        all_order.append(
            order_row
        )

        # --------------------------------------------
        # True RMS-normalized STFT
        # --------------------------------------------

        rms_features = extract_rms_normalized_stft_features(
            signal
        )

        rms_row = {
            "Bearing": bearing,
            "Condition": condition,
            "RPM": rpm,
            "Rotation_Hz": rotation_hz,
            "Label": label,
            "File": filename,
        }

        rms_row.update(
            rms_features
        )

        all_rms.append(
            rms_row
        )

        file_count += 1

        if file_count % 100 == 0:
            print(
                f"  Processed {file_count}/800 files"
            )


# ============================================================
# 2. Convert to DataFrames
# ============================================================

absolute_df = pd.DataFrame(all_absolute)
order_df = pd.DataFrame(all_order)
rms_df = pd.DataFrame(all_rms)

print("\n" + "=" * 75)
print("FEATURE DATASET CHECK")
print("=" * 75)

print("Absolute STFT:", absolute_df.shape)
print("Order STFT:   ", order_df.shape)
print("RMS STFT:     ", rms_df.shape)

print("\nAbsolute labels:")
print(absolute_df["Label"].value_counts().sort_index())

print("\nOrder labels:")
print(order_df["Label"].value_counts().sort_index())

print("\nRMS labels:")
print(rms_df["Label"].value_counts().sort_index())


# ============================================================
# 3. Save corrected feature files
# ============================================================

absolute_path = os.path.join(
    ROOT,
    "corrected_64khz_absolute_stft.csv"
)

order_path = os.path.join(
    ROOT,
    "corrected_64khz_order_stft.csv"
)

rms_path = os.path.join(
    ROOT,
    "corrected_64khz_rms_stft.csv"
)

absolute_df.to_csv(
    absolute_path,
    index=False
)

order_df.to_csv(
    order_path,
    index=False
)

rms_df.to_csv(
    rms_path,
    index=False
)

print("\nSaved:")
print(os.path.basename(absolute_path))
print(os.path.basename(order_path))
print(os.path.basename(rms_path))


# ============================================================
# 4. Strict Leave-One-Bearing-Out evaluation
# ============================================================

def evaluate_lobo(df, method_name):

    print("\n" + "=" * 75)
    print(f"LOBO RESULTS — {method_name}")
    print("=" * 75)

    excluded_columns = {
        "Bearing",
        "Condition",
        "RPM",
        "Rotation_Hz",
        "Label",
        "File",
    }

    feature_columns = [
        c for c in df.columns
        if c not in excluded_columns
    ]

    print(f"Number of numerical features: {len(feature_columns)}")

    all_true = []
    all_pred = []

    bearing_rows = []

    for test_bearing in BEARINGS:

        train_mask = (
            df["Bearing"] != test_bearing
        )

        test_mask = (
            df["Bearing"] == test_bearing
        )

        X_train = df.loc[
            train_mask,
            feature_columns
        ]

        y_train = df.loc[
            train_mask,
            "Label"
        ]

        X_test = df.loc[
            test_mask,
            feature_columns
        ]

        y_test = df.loc[
            test_mask,
            "Label"
        ]

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            y_pred.tolist()
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        if test_bearing in HEALTHY:

            fpr = fp / (tn + fp)

            bearing_rows.append({
                "Bearing": test_bearing,
                "Type": "Healthy",
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Healthy_FPR": fpr,
                "Damaged_Recall": np.nan
            })

            print(
                f"{test_bearing}: "
                f"FPR = {fpr:.4f}"
            )

        else:

            damaged_recall = tp / (tp + fn)

            bearing_rows.append({
                "Bearing": test_bearing,
                "Type": "Damaged",
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "Healthy_FPR": np.nan,
                "Damaged_Recall": damaged_recall
            })

            print(
                f"{test_bearing}: "
                f"Recall = {damaged_recall:.4f}"
            )

    # --------------------------------------------------------
    # Pooled metrics
    # --------------------------------------------------------

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    precision = precision_score(
        all_true,
        all_pred,
        zero_division=0
    )

    recall = recall_score(
        all_true,
        all_pred,
        zero_division=0
    )

    f1 = f1_score(
        all_true,
        all_pred,
        zero_division=0
    )

    healthy_fpr = fp / (tn + fp)

    specificity = tn / (tn + fp)

    print("\nPooled confusion matrix:")
    print(cm)

    print("\nPooled metrics:")
    print(f"Accuracy:       {accuracy:.6f}")
    print(f"Precision:      {precision:.6f}")
    print(f"Recall:         {recall:.6f}")
    print(f"F1:             {f1:.6f}")
    print(f"Healthy FPR:    {healthy_fpr:.6f}")
    print(f"Specificity:    {specificity:.6f}")

    result = {
        "Method": method_name,
        "Sampling_Rate_Hz": FS,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Healthy_FPR": healthy_fpr,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    bearing_df = pd.DataFrame(
        bearing_rows
    )

    return result, bearing_df


# ============================================================
# 5. Evaluate all three corrected representations
# ============================================================

absolute_result, absolute_bearing = evaluate_lobo(
    absolute_df,
    "Corrected Absolute STFT"
)

order_result, order_bearing = evaluate_lobo(
    order_df,
    "Corrected Order-Normalized STFT"
)

rms_result, rms_bearing = evaluate_lobo(
    rms_df,
    "Corrected True RMS-Normalized STFT"
)


# ============================================================
# 6. Final comparison
# ============================================================

comparison = pd.DataFrame([
    absolute_result,
    order_result,
    rms_result
])

comparison_path = os.path.join(
    ROOT,
    "corrected_final_stft_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 75)
print("FINAL CORRECTED STFT COMPARISON")
print("=" * 75)

print(
    comparison[
        [
            "Method",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Healthy_FPR",
            "Specificity"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(os.path.basename(comparison_path))


# ============================================================
# 7. Save bearing-level results
# ============================================================

absolute_bearing["Method"] = (
    "Corrected Absolute STFT"
)

order_bearing["Method"] = (
    "Corrected Order-Normalized STFT"
)

rms_bearing["Method"] = (
    "Corrected True RMS-Normalized STFT"
)

bearing_comparison = pd.concat(
    [
        absolute_bearing,
        order_bearing,
        rms_bearing
    ],
    ignore_index=True
)

bearing_path = os.path.join(
    ROOT,
    "corrected_bearing_results.csv"
)

bearing_comparison.to_csv(
    bearing_path,
    index=False
)

print(os.path.basename(bearing_path))


# ============================================================
# 8. Final sanity checks
# ============================================================

print("\n" + "=" * 75)
print("FINAL SANITY CHECKS")
print("=" * 75)

assert len(absolute_df) == 800
assert len(order_df) == 800
assert len(rms_df) == 800

assert absolute_df["Label"].sum() == 320
assert order_df["Label"].sum() == 320
assert rms_df["Label"].sum() == 320

assert absolute_df["Bearing"].nunique() == 10
assert order_df["Bearing"].nunique() == 10
assert rms_df["Bearing"].nunique() == 10

assert all(
    absolute_df.groupby("Bearing").size() == 80
)

assert all(
    order_df.groupby("Bearing").size() == 80
)

assert all(
    rms_df.groupby("Bearing").size() == 80
)

print("PASS: 800 samples in every feature dataset.")
print("PASS: 10 bearings.")
print("PASS: 80 samples per bearing.")
print("PASS: 480 healthy / 320 damaged.")
print("PASS: Fixed fs = 64000 Hz.")
print("PASS: Same LOBO protocol for all three methods.")
print("PASS: No new dataset or classifier introduced.")

print("\n" + "=" * 75)
print("STEP 84 COMPLETE")
print("=" * 75)

print("\nIMPORTANT:")
print("These corrected 64 kHz results should replace the earlier")
print("STFT-based results in the final paper.")
print("Do NOT use Step 78.")