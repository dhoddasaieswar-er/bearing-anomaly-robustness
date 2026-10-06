import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# STEP 153
# ONE-CLASS SVM BASELINE
# STRICT 10-BEARING LOBO
#
# Input:
#   relative_stft_features.csv
#
# Training:
#   HEALTHY recordings only
#
# Testing:
#   Complete held-out bearing
#
# IMPORTANT:
#   Labels are NEVER used during OC-SVM fitting.
#   Labels are used only for evaluation.
# ============================================================

INPUT_FILE = "relative_stft_features.csv"

# One-Class SVM parameters
NU = 0.10
KERNEL = "rbf"
GAMMA = "scale"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 153 - ONE-CLASS SVM LOBO")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nInput file:")
print(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)

# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_cols = [
    "Order_0_20_Mean",
    "Order_0_20_Max",
    "Order_20_50_Mean",
    "Order_20_50_Max",
    "Order_50_100_Mean",
    "Order_50_100_Max",
    "Order_100_200_Mean",
    "Order_100_200_Max",
    "Order_200_400_Mean",
    "Order_200_400_Max",
    "Relative_STFT_Mean",
    "Relative_STFT_Max"
]

required_cols = feature_cols + [
    "Bearing",
    "Label",
    "File"
]

missing = [
    col for col in required_cols
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=feature_cols + ["Bearing", "Label"]
).copy()

df["Label"] = df["Label"].astype(int)

# ============================================================
# BEARING ORDER
# ============================================================

bearings = [
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

available_bearings = sorted(
    df["Bearing"].unique().tolist()
)

if set(bearings) != set(available_bearings):
    raise ValueError(
        "Bearing identities do not match expected 10-bearing set.\n"
        f"Expected: {bearings}\n"
        f"Found: {available_bearings}"
    )

# ============================================================
# CHECK DATASET
# ============================================================

print("\nFeatures:")
for i, col in enumerate(feature_cols):
    print(f"{i:2d}. {col}")

print("\nBearing counts:")
print(df["Bearing"].value_counts().sort_index())

print("\nLabel counts:")
print(df["Label"].value_counts().sort_index())

# ============================================================
# STORAGE
# ============================================================

all_predictions = []
bearing_results = []

# ============================================================
# LOBO
# ============================================================

for held_out in bearings:

    print("\n" + "=" * 80)
    print(f"HELD-OUT BEARING: {held_out}")
    print("=" * 80)

    train_df = df[
        df["Bearing"] != held_out
    ].copy()

    test_df = df[
        df["Bearing"] == held_out
    ].copy()

    # --------------------------------------------------------
    # TRAINING DATA = HEALTHY ONLY
    # --------------------------------------------------------

    train_healthy = train_df[
        train_df["Label"] == 0
    ].copy()

    X_train = train_healthy[
        feature_cols
    ].to_numpy(dtype=float)

    X_test = test_df[
        feature_cols
    ].to_numpy(dtype=float)

    y_test = test_df[
        "Label"
    ].to_numpy(dtype=int)

    # --------------------------------------------------------
    # STANDARDIZATION
    #
    # Fit ONLY on healthy training recordings.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # ONE-CLASS SVM
    #
    # sklearn:
    #   +1 = inlier / normal
    #   -1 = outlier / anomaly
    # --------------------------------------------------------

    model = OneClassSVM(
        kernel=KERNEL,
        nu=NU,
        gamma=GAMMA
    )

    model.fit(X_train_scaled)

    raw_prediction = model.predict(
        X_test_scaled
    )

    # Convert:
    # +1 -> 0 = healthy
    # -1 -> 1 = anomaly/failure

    y_pred = np.where(
        raw_prediction == -1,
        1,
        0
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    healthy_count = np.sum(
        y_test == 0
    )

    if healthy_count > 0:
        healthy_fpr = fp / healthy_count
        specificity = 1.0 - healthy_fpr
    else:
        healthy_fpr = np.nan
        specificity = np.nan

    failures = int(
        np.sum(y_pred == 1)
    )

    # --------------------------------------------------------
    # STORE RECORD-LEVEL PREDICTIONS
    # --------------------------------------------------------

    for i in range(len(test_df)):

        row = test_df.iloc[i]

        all_predictions.append({
            "Bearing": held_out,
            "True_Label": int(y_test[i]),
            "Predicted_Label": int(y_pred[i]),
            "File": row["File"],
            "Failed": int(y_pred[i] == 1)
        })

    # --------------------------------------------------------
    # STORE BEARING RESULTS
    # --------------------------------------------------------

    bearing_results.append({
        "Bearing": held_out,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Healthy_FPR": healthy_fpr,
        "Specificity": specificity,
        "Failures": failures,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    print(f"Training healthy samples : {len(train_healthy)}")
    print(f"Test samples             : {len(test_df)}")
    print(f"Actual failures          : {np.sum(y_test == 1)}")
    print(f"Predicted anomalies      : {failures}")

    print(f"\nAccuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1           : {f1:.4f}")
    print(f"Healthy FPR  : {healthy_fpr:.4f}")
    print(f"Specificity  : {specificity:.4f}")

# ============================================================
# DATAFRAMES
# ============================================================

predictions_df = pd.DataFrame(
    all_predictions
)

results_df = pd.DataFrame(
    bearing_results
)

# ============================================================
# POOLED CONFUSION MATRIX
# ============================================================

y_true_all = predictions_df[
    "True_Label"
].to_numpy()

y_pred_all = predictions_df[
    "Predicted_Label"
].to_numpy()

tn, fp, fn, tp = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[0, 1]
).ravel()

pooled_accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)

pooled_precision = precision_score(
    y_true_all,
    y_pred_all,
    zero_division=0
)

pooled_recall = recall_score(
    y_true_all,
    y_pred_all,
    zero_division=0
)

pooled_f1 = f1_score(
    y_true_all,
    y_pred_all,
    zero_division=0
)

healthy_total = np.sum(
    y_true_all == 0
)

pooled_healthy_fpr = (
    fp / healthy_total
)

pooled_specificity = (
    1.0 - pooled_healthy_fpr
)

# ============================================================
# LOBO F1 DISPERSION
# ============================================================

f1_values = results_df[
    "F1"
].to_numpy()

mean_f1 = np.mean(
    f1_values
)

sd_f1 = np.std(
    f1_values,
    ddof=1
)

median_f1 = np.median(
    f1_values
)

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n\n" + "=" * 80)
print("STEP 153 - FINAL ONE-CLASS SVM RESULTS")
print("=" * 80)

print("\nBearing-level results:")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "-" * 80)

print("\nPOOLED RESULTS")
print("-" * 40)

print(f"Accuracy     : {pooled_accuracy:.4f}")
print(f"Precision    : {pooled_precision:.4f}")
print(f"Recall       : {pooled_recall:.4f}")
print(f"F1           : {pooled_f1:.4f}")
print(f"Healthy FPR  : {pooled_healthy_fpr:.4f}")
print(f"Specificity  : {pooled_specificity:.4f}")

print("\nConfusion Matrix")
print(
    f"[[{tn}, {fp}],"
    f" [{fn}, {tp}]]"
)

print("\nLOBO F1 DISPERSION")
print("-" * 40)

print(f"Mean F1      : {mean_f1:.4f}")
print(f"SD F1        : {sd_f1:.4f}")
print(f"Median F1    : {median_f1:.4f}")
print(f"Min F1       : {np.min(f1_values):.4f}")
print(f"Max F1       : {np.max(f1_values):.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "ocsvm_bearing_metrics.csv",
    index=False
)

predictions_df.to_csv(
    "ocsvm_predictions.csv",
    index=False
)

summary_df = pd.DataFrame([{
    "Method": "One-Class SVM",
    "Accuracy": pooled_accuracy,
    "Precision": pooled_precision,
    "Recall": pooled_recall,
    "F1": pooled_f1,
    "Healthy_FPR": pooled_healthy_fpr,
    "Specificity": pooled_specificity,
    "Mean_LOBO_F1": mean_f1,
    "SD_LOBO_F1": sd_f1,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "Nu": NU,
    "Kernel": KERNEL,
    "Gamma": GAMMA
}])

summary_df.to_csv(
    "ocsvm_summary.csv",
    index=False
)

print("\nSaved:")
print("  ocsvm_bearing_metrics.csv")
print("  ocsvm_predictions.csv")
print("  ocsvm_summary.csv")

print("\n" + "=" * 80)
print("STEP 153 COMPLETE")
print("=" * 80)