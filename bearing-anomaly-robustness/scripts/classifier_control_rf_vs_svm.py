"""
STEP 85 — CLASSIFIER-CONTROL EXPERIMENT
Random Forest vs SVM

Research:
Robust Anomaly Detection in Non-Stationary Signals
Under Variable Noise and Operating Conditions

IMPORTANT:
- Same Paderborn dataset
- Same features
- Same LOBO bearing split
- Same evaluation procedure
- Same random seed
- Only classifier changes
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 300

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. DATASETS
# ============================================================

DATASETS = {

    "Time-domain":
        BASE_DIR / "all_real_time_features.csv",

    "Absolute STFT":
        BASE_DIR / "corrected_64khz_absolute_stft.csv",

    "Order STFT":
        BASE_DIR / "corrected_64khz_order_stft.csv",

    "RMS-normalized STFT":
        BASE_DIR / "corrected_64khz_rms_stft.csv",
}


# ============================================================
# 3. METADATA COLUMNS
# ============================================================

META_COLUMNS = {
    "Bearing",
    "Condition",
    "RPM",
    "Rotation_Hz",
    "Label",
    "File",
}


# ============================================================
# 4. RANDOM FOREST
# ============================================================

def create_random_forest():

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# ============================================================
# 5. SVM
# ============================================================

def create_svm():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            )
        )
    ])

    return model


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    healthy_fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,

        "Healthy_FPR": healthy_fpr,
        "Specificity": specificity,
    }


# ============================================================
# 7. LOBO EXPERIMENT
# ============================================================

def run_lobo(
    df,
    feature_columns,
    classifier_name,
    classifier_function
):

    bearings = sorted(
        df["Bearing"]
        .astype(str)
        .unique()
    )

    all_true = []
    all_pred = []

    bearing_results = []

    print()
    print("-" * 70)
    print(classifier_name)
    print("-" * 70)

    for test_bearing in bearings:

        print(
            f"Testing bearing: {test_bearing}"
        )

        # ----------------------------------------------------
        # LOBO SPLIT
        # ----------------------------------------------------

        train_data = df[
            df["Bearing"].astype(str)
            != test_bearing
        ].copy()

        test_data = df[
            df["Bearing"].astype(str)
            == test_bearing
        ].copy()

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        X_train = train_data[
            feature_columns
        ]

        y_train = train_data[
            "Label"
        ].astype(int)

        X_test = test_data[
            feature_columns
        ]

        y_test = test_data[
            "Label"
        ].astype(int)

        # ----------------------------------------------------
        # Create classifier
        # ----------------------------------------------------

        model = classifier_function()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Store pooled predictions
        # ----------------------------------------------------

        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            y_pred.tolist()
        )

        # ----------------------------------------------------
        # Bearing-level analysis
        # ----------------------------------------------------

        bearing_class = int(
            test_data["Label"].iloc[0]
        )

        if bearing_class == 0:

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            ).ravel()

            fpr = (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0
            )

            bearing_results.append({
                "Classifier": classifier_name,
                "Bearing": test_bearing,
                "Class": "Healthy",
                "Metric": "Healthy_FPR",
                "Value": fpr,
            })

        else:

            fn = np.sum(
                (y_test == 1)
                & (y_pred == 0)
            )

            tp = np.sum(
                (y_test == 1)
                & (y_pred == 1)
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0
            )

            bearing_results.append({
                "Classifier": classifier_name,
                "Bearing": test_bearing,
                "Class": "Damaged",
                "Metric": "Damaged_Recall",
                "Value": recall,
            })

    # ========================================================
    # POOLED RESULTS
    # ========================================================

    results = calculate_metrics(
        all_true,
        all_pred
    )

    results["Classifier"] = classifier_name

    return results, bearing_results


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 85 — CLASSIFIER CONTROL EXPERIMENT")
    print("RANDOM FOREST vs SVM")
    print("=" * 70)

    print()
    print("Experimental controls:")
    print("Dataset      : Paderborn")
    print("Split        : Leave-One-Bearing-Out")
    print("Bearings     : 10")
    print("Samples      : 800")
    print("Random seed  : 42")
    print("RF trees     : 300")
    print("SVM kernel   : RBF")
    print("SVM C        : 1.0")
    print("SVM gamma    : scale")
    print("Class weight : balanced")

    # ========================================================
    # CHECK FILES
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKING DATA FILES")
    print("=" * 70)

    for name, path in DATASETS.items():

        print(
            f"{name:25s}: ",
            end=""
        )

        if path.exists():
            print("FOUND")

        else:
            print("MISSING")

            print()
            print(
                "ERROR: Required CSV file not found:"
            )

            print(path)

            return

    # ========================================================
    # RESULTS STORAGE
    # ========================================================

    pooled_results = []
    bearing_results_all = []

    # ========================================================
    # RUN ALL REPRESENTATIONS
    # ========================================================

    for representation, file_path in DATASETS.items():

        print()
        print()
        print("#" * 70)
        print(
            f"REPRESENTATION: {representation}"
        )
        print("#" * 70)

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        df = pd.read_csv(
            file_path
        )

        print(
            f"Rows: {len(df)}"
        )

        # ----------------------------------------------------
        # Identify numerical features
        # ----------------------------------------------------

        feature_columns = []

        for column in df.columns:

            if column in META_COLUMNS:
                continue

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                feature_columns.append(
                    column
                )

        print(
            f"Number of features: "
            f"{len(feature_columns)}"
        )

        print("Features:")

        for feature in feature_columns:
            print(
                "  ",
                feature
            )

        # ====================================================
        # RANDOM FOREST
        # ====================================================

        rf_result, rf_bearing = run_lobo(
            df=df,
            feature_columns=feature_columns,
            classifier_name="Random Forest",
            classifier_function=create_random_forest,
        )

        rf_result[
            "Representation"
        ] = representation

        pooled_results.append(
            rf_result
        )

        bearing_results_all.extend(
            rf_bearing
        )

        # ====================================================
        # SVM
        # ====================================================

        svm_result, svm_bearing = run_lobo(
            df=df,
            feature_columns=feature_columns,
            classifier_name="SVM",
            classifier_function=create_svm,
        )

        svm_result[
            "Representation"
        ] = representation

        pooled_results.append(
            svm_result
        )

        bearing_results_all.extend(
            svm_bearing
        )

        # ====================================================
        # PRINT COMPARISON
        # ====================================================

        print()
        print(
            "RESULT COMPARISON"
        )

        print()

        print(
            "Random Forest"
        )

        print(
            f"Accuracy  : {rf_result['Accuracy']:.4f}"
        )

        print(
            f"Precision : {rf_result['Precision']:.4f}"
        )

        print(
            f"Recall    : {rf_result['Recall']:.4f}"
        )

        print(
            f"F1        : {rf_result['F1']:.4f}"
        )

        print(
            f"Healthy FPR : "
            f"{rf_result['Healthy_FPR']:.4f}"
        )

        print()

        print(
            "SVM"
        )

        print(
            f"Accuracy  : {svm_result['Accuracy']:.4f}"
        )

        print(
            f"Precision : {svm_result['Precision']:.4f}"
        )

        print(
            f"Recall    : {svm_result['Recall']:.4f}"
        )

        print(
            f"F1        : {svm_result['F1']:.4f}"
        )

        print(
            f"Healthy FPR : "
            f"{svm_result['Healthy_FPR']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    pooled_df = pd.DataFrame(
        pooled_results
    )

    bearing_df = pd.DataFrame(
        bearing_results_all
    )

    pooled_columns = [
        "Representation",
        "Classifier",

        "TN",
        "FP",
        "FN",
        "TP",

        "Accuracy",
        "Precision",
        "Recall",
        "F1",

        "Healthy_FPR",
        "Specificity",
    ]

    pooled_df = pooled_df[
        pooled_columns
    ]

    pooled_output = (
        BASE_DIR
        / "classifier_control_pooled_results.csv"
    )

    bearing_output = (
        BASE_DIR
        / "classifier_control_bearing_results.csv"
    )

    pooled_df.to_csv(
        pooled_output,
        index=False
    )

    bearing_df.to_csv(
        bearing_output,
        index=False
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("FINAL STEP 85 RESULTS")
    print("=" * 70)

    print(
        pooled_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    for _, row in pooled_df.iterrows():

        total = (
            row["TN"]
            + row["FP"]
            + row["FN"]
            + row["TP"]
        )

        assert total == 800

    print(
        "PASS: Every experiment contains exactly 800 predictions."
    )

    print(
        "PASS: Same LOBO split used for RF and SVM."
    )

    print(
        "PASS: Same feature sets used for RF and SVM."
    )

    print(
        "PASS: SVM scaling is performed inside each training fold."
    )

    print()
    print(
        "Results saved:"
    )

    print(
        pooled_output.name
    )

    print(
        bearing_output.name
    )

    print()
    print(
        "=" * 70
    )

    print(
        "STEP 85 COMPLETE"
    )

    print(
        "Do NOT modify the results yet."
    )

    print(
        "Send the complete terminal output for verification."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()