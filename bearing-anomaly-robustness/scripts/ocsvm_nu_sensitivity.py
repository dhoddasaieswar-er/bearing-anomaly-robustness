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
# STEP 154
# ONE-CLASS SVM NU SENSITIVITY
# STRICT 10-BEARING LOBO
# ============================================================

INPUT_FILE = "relative_stft_features.csv"

NU_VALUES = [0.01, 0.05, 0.10]

KERNEL = "rbf"
GAMMA = "scale"

FEATURE_COLS = [
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

BEARINGS = [
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

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("STEP 154 - ONE-CLASS SVM NU SENSITIVITY")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

# ============================================================
# RESULTS
# ============================================================

all_results = []

# ============================================================
# NU LOOP
# ============================================================

for nu in NU_VALUES:

    print("\n" + "=" * 80)
    print(f"NU = {nu}")
    print("=" * 80)

    all_predictions = []
    bearing_results = []

    # --------------------------------------------------------
    # LOBO
    # --------------------------------------------------------

    for held_out in BEARINGS:

        train_df = df[
            df["Bearing"] != held_out
        ].copy()

        test_df = df[
            df["Bearing"] == held_out
        ].copy()

        # Healthy training recordings only
        train_healthy = train_df[
            train_df["Label"] == 0
        ].copy()

        X_train = train_healthy[
            FEATURE_COLS
        ].to_numpy(dtype=float)

        X_test = test_df[
            FEATURE_COLS
        ].to_numpy(dtype=float)

        y_test = test_df[
            "Label"
        ].to_numpy(dtype=int)

        # ----------------------------------------------------
        # FIT SCALER ONLY ON TRAINING HEALTHY DATA
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # OC-SVM
        # ----------------------------------------------------

        model = OneClassSVM(
            kernel=KERNEL,
            nu=nu,
            gamma=GAMMA
        )

        model.fit(X_train_scaled)

        raw_pred = model.predict(
            X_test_scaled
        )

        # +1 = healthy
        # -1 = anomaly

        y_pred = np.where(
            raw_pred == -1,
            1,
            0
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        bearing_results.append({
            "Nu": nu,
            "Bearing": held_out,
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
        })

        # Store predictions for pooled calculation
        for i in range(len(test_df)):

            all_predictions.append({
                "Nu": nu,
                "Bearing": held_out,
                "True_Label": int(y_test[i]),
                "Predicted_Label": int(y_pred[i]),
                "File": test_df.iloc[i]["File"]
            })

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    bearing_df = pd.DataFrame(
        bearing_results
    )

    pred_df = pd.DataFrame(
        all_predictions
    )

    # --------------------------------------------------------
    # POOLED METRICS
    # --------------------------------------------------------

    y_true = pred_df[
        "True_Label"
    ].to_numpy()

    y_pred = pred_df[
        "Predicted_Label"
    ].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    pooled_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    pooled_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    pooled_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    pooled_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    healthy_total = np.sum(
        y_true == 0
    )

    pooled_fpr = fp / healthy_total
    pooled_specificity = 1.0 - pooled_fpr

    # --------------------------------------------------------
    # LOBO F1 DISPERSION
    # --------------------------------------------------------

    f1_values = bearing_df[
        "F1"
    ].to_numpy()

    mean_f1 = np.mean(f1_values)
    sd_f1 = np.std(
        f1_values,
        ddof=1
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "Nu": nu,
        "Accuracy": pooled_accuracy,
        "Precision": pooled_precision,
        "Recall": pooled_recall,
        "F1": pooled_f1,
        "Healthy_FPR": pooled_fpr,
        "Specificity": pooled_specificity,
        "Mean_LOBO_F1": mean_f1,
        "SD_LOBO_F1": sd_f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    all_results.append(summary)

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nPooled:")
    print(f"Accuracy    : {pooled_accuracy:.4f}")
    print(f"Precision   : {pooled_precision:.4f}")
    print(f"Recall      : {pooled_recall:.4f}")
    print(f"F1          : {pooled_f1:.4f}")
    print(f"Healthy FPR : {pooled_fpr:.4f}")
    print(f"Specificity : {pooled_specificity:.4f}")

    print("\nLOBO F1:")
    print(f"Mean        : {mean_f1:.4f}")
    print(f"SD          : {sd_f1:.4f}")

# ============================================================
# FINAL TABLE
# ============================================================

summary_df = pd.DataFrame(
    all_results
)

print("\n\n" + "=" * 80)
print("STEP 154 - FINAL NU SENSITIVITY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    "ocsvm_nu_sensitivity.csv",
    index=False
)

print("\nSaved:")
print("ocsvm_nu_sensitivity.csv")

print("\n" + "=" * 80)
print("STEP 154 COMPLETE")
print("=" * 80)