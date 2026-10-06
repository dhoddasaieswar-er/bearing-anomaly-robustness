from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# STEP 93 — FAILURE CASE ANALYSIS
#
# Reconstructs the frozen Step 85 LOBO protocol.
#
# NO NEW DATASET
# NO NEW FEATURES
# NO NEW TRAIN/TEST SPLIT
#
# Purpose:
# Identify signal-level false positives and false negatives.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "failure_case_analysis_step93"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

TIME_FILE = BASE_DIR / "all_real_time_features.csv"
STFT_FILE = BASE_DIR / "all_real_stft_features.csv"
ORDER_FILE = BASE_DIR / "relative_stft_features.csv"
RMS_FILE = BASE_DIR / "normalized_stft_features.csv"

INPUTS = {
    "Time-domain": TIME_FILE,
    "Absolute STFT": STFT_FILE,
    "Order-normalized STFT": ORDER_FILE,
    "RMS-normalized STFT": RMS_FILE,
}


# ============================================================
# BEARING ORDER
# ============================================================

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
    "KI05",
]


HEALTHY = {
    "K001",
    "K002",
    "K003",
    "K004",
    "K005",
    "K006",
}


DAMAGED = {
    "KA01",
    "KA03",
    "KA04",
    "KI05",
}


# ============================================================
# LOAD DATA
# ============================================================

datasets = {}

for representation, path in INPUTS.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{path}"
        )

    df = pd.read_csv(path)

    datasets[representation] = df

    print(
        f"Loaded {representation}: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

META_COLUMNS = {
    "Bearing",
    "Condition",
    "Label",
    "File",
    "RPM",
    "Rotation_Hz",
}


def get_feature_columns(df):

    feature_columns = []

    for col in df.columns:

        if col in META_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    return feature_columns


# ============================================================
# CLASSIFIERS — EXACT STEP 85 SETTINGS
# ============================================================

def make_rf():

    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def make_svm():

    return Pipeline([
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
                class_weight="balanced"
            )
        )
    ])


CLASSIFIERS = {
    "Random Forest": make_rf,
    "RBF-SVM": make_svm,
}


# ============================================================
# RECONSTRUCT LOBO PREDICTIONS
# ============================================================

all_predictions = []

experiment_number = 0


for representation, df in datasets.items():

    feature_columns = get_feature_columns(df)

    X = df[feature_columns].values
    y = df["Label"].astype(int).values
    bearing_values = df["Bearing"].astype(str).values

    print()
    print("=" * 80)
    print(f"REPRESENTATION: {representation}")
    print(f"FEATURES: {len(feature_columns)}")
    print("=" * 80)

    for classifier_name, classifier_factory in CLASSIFIERS.items():

        experiment_number += 1

        print()
        print(
            f"[{experiment_number}/8] "
            f"{representation} × {classifier_name}"
        )

        for test_bearing in BEARINGS:

            test_mask = bearing_values == test_bearing
            train_mask = ~test_mask

            X_train = X[train_mask]
            X_test = X[test_mask]

            y_train = y[train_mask]
            y_test = y[test_mask]

            classifier = classifier_factory()

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            # ------------------------------------------------
            # STORE EVERY TEST SIGNAL
            # ------------------------------------------------

            test_indices = np.where(test_mask)[0]

            for local_index, original_index in enumerate(test_indices):

                true_label = int(y_test[local_index])
                predicted_label = int(y_pred[local_index])

                if true_label == 0 and predicted_label == 1:
                    failure_type = "False Positive"

                elif true_label == 1 and predicted_label == 0:
                    failure_type = "False Negative"

                elif true_label == 0 and predicted_label == 0:
                    failure_type = "Correct Healthy"

                else:
                    failure_type = "Correct Damaged"

                all_predictions.append({
                    "Representation": representation,
                    "Classifier": classifier_name,
                    "Test_Bearing": test_bearing,
                    "File": df.iloc[original_index]["File"],
                    "Condition": df.iloc[original_index]["Condition"],
                    "True_Label": true_label,
                    "Predicted_Label": predicted_label,
                    "Failure_Type": failure_type,
                })


# ============================================================
# CREATE PREDICTION TABLE
# ============================================================

predictions = pd.DataFrame(all_predictions)

prediction_file = (
    OUT_DIR /
    "step93_signal_level_predictions.csv"
)

predictions.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# EXTRACT ONLY FAILURES
# ============================================================

failures = predictions[
    predictions["Failure_Type"].isin(
        ["False Positive", "False Negative"]
    )
].copy()

failure_file = (
    OUT_DIR /
    "step93_all_failure_cases.csv"
)

failures.to_csv(
    failure_file,
    index=False
)


# ============================================================
# FAILURE COUNTS BY REPRESENTATION × CLASSIFIER × BEARING
# ============================================================

failure_summary = (
    failures
    .groupby(
        [
            "Representation",
            "Classifier",
            "Test_Bearing",
            "Failure_Type"
        ]
    )
    .size()
    .reset_index(name="Failure_Count")
)

failure_summary_file = (
    OUT_DIR /
    "step93_failure_summary_by_bearing.csv"
)

failure_summary.to_csv(
    failure_summary_file,
    index=False
)


# ============================================================
# HEALTHY FALSE POSITIVE SUMMARY
# ============================================================

healthy_fp = predictions[
    (predictions["True_Label"] == 0) &
    (predictions["Predicted_Label"] == 1)
]

healthy_summary = (
    healthy_fp
    .groupby(
        [
            "Representation",
            "Classifier",
            "Test_Bearing"
        ]
    )
    .size()
    .reset_index(name="False_Positive_Count")
)

healthy_total = (
    predictions[
        predictions["True_Label"] == 0
    ]
    .groupby(
        [
            "Representation",
            "Classifier",
            "Test_Bearing"
        ]
    )
    .size()
    .reset_index(name="Healthy_Total")
)

healthy_summary = healthy_summary.merge(
    healthy_total,
    on=[
        "Representation",
        "Classifier",
        "Test_Bearing"
    ],
    how="right"
)

healthy_summary["False_Positive_Count"] = (
    healthy_summary["False_Positive_Count"]
    .fillna(0)
    .astype(int)
)

healthy_summary["FPR"] = (
    healthy_summary["False_Positive_Count"]
    /
    healthy_summary["Healthy_Total"]
)

healthy_summary_file = (
    OUT_DIR /
    "step93_healthy_false_positive_analysis.csv"
)

healthy_summary.to_csv(
    healthy_summary_file,
    index=False
)


# ============================================================
# DAMAGED FALSE NEGATIVE SUMMARY
# ============================================================

damaged_fn = predictions[
    (predictions["True_Label"] == 1) &
    (predictions["Predicted_Label"] == 0)
]

damaged_summary = (
    damaged_fn
    .groupby(
        [
            "Representation",
            "Classifier",
            "Test_Bearing"
        ]
    )
    .size()
    .reset_index(name="False_Negative_Count")
)

damaged_total = (
    predictions[
        predictions["True_Label"] == 1
    ]
    .groupby(
        [
            "Representation",
            "Classifier",
            "Test_Bearing"
        ]
    )
    .size()
    .reset_index(name="Damaged_Total")
)

damaged_summary = damaged_summary.merge(
    damaged_total,
    on=[
        "Representation",
        "Classifier",
        "Test_Bearing"
    ],
    how="right"
)

damaged_summary["False_Negative_Count"] = (
    damaged_summary["False_Negative_Count"]
    .fillna(0)
    .astype(int)
)

damaged_summary["Recall"] = (
    1 -
    damaged_summary["False_Negative_Count"]
    /
    damaged_summary["Damaged_Total"]
)

damaged_summary_file = (
    OUT_DIR /
    "step93_damaged_false_negative_analysis.csv"
)

damaged_summary.to_csv(
    damaged_summary_file,
    index=False
)


# ============================================================
# CROSS-METHOD FAILURE CONSISTENCY
#
# For each File, determine:
# - number of configurations failing
# - number of FP
# - number of FN
#
# This identifies failures repeated across methods.
# ============================================================

failure_consistency = (
    failures
    .groupby("File")
    .agg(
        Failure_Configurations=("Failure_Type", "count"),
        Representation_Count=("Representation", "nunique"),
        Classifier_Count=("Classifier", "nunique"),
        Test_Bearing=("Test_Bearing", "first"),
        Condition=("Condition", "first"),
        Failure_Types=("Failure_Type", lambda x: ",".join(sorted(set(x))))
    )
    .reset_index()
)

failure_consistency["Total_Configurations"] = 8

failure_consistency = failure_consistency.sort_values(
    "Failure_Configurations",
    ascending=False
)

consistency_file = (
    OUT_DIR /
    "step93_cross_method_failure_consistency.csv"
)

failure_consistency.to_csv(
    consistency_file,
    index=False
)


# ============================================================
# MOST PERSISTENT FAILURE CASES
# ============================================================

persistent = failure_consistency[
    failure_consistency["Failure_Configurations"] >= 4
].copy()

persistent_file = (
    OUT_DIR /
    "step93_persistent_failure_cases.csv"
)

persistent.to_csv(
    persistent_file,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("STEP 93 — FAILURE CASE ANALYSIS COMPLETE")
print("=" * 80)

print()
print("Total predictions:", len(predictions))
print("Total failure predictions:", len(failures))

print()
print("Failure counts:")
print(
    failures["Failure_Type"]
    .value_counts()
    .to_string()
)

print()
print("Most frequent healthy false-positive bearings:")

fp_bearing = (
    healthy_fp
    .groupby("Test_Bearing")
    .size()
    .sort_values(ascending=False)
)

print(fp_bearing.to_string())


print()
print("Most frequent damaged false-negative bearings:")

fn_bearing = (
    damaged_fn
    .groupby("Test_Bearing")
    .size()
    .sort_values(ascending=False)
)

print(fn_bearing.to_string())


print()
print("Persistent signal-level failures")
print("(failed in at least 4 of the 8 configurations):")

if len(persistent) == 0:

    print("None")

else:

    print(
        persistent[
            [
                "File",
                "Test_Bearing",
                "Condition",
                "Failure_Configurations",
                "Representation_Count",
                "Classifier_Count",
                "Failure_Types"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# SANITY CHECKS
# ============================================================

expected_predictions = 800 * 8

assert len(predictions) == expected_predictions

assert set(predictions["Representation"]) == set(INPUTS.keys())

assert set(predictions["Classifier"]) == set(
    CLASSIFIERS.keys()
)

assert predictions["Test_Bearing"].nunique() == 10

assert (
    predictions
    .groupby(
        ["Representation", "Classifier"]
    )
    .size()
    .eq(800)
    .all()
)

print()
print("=" * 80)
print("SANITY CHECKS: PASS")
print("=" * 80)

print()
print("Expected predictions:", expected_predictions)
print("Actual predictions:", len(predictions))

print()
print("Saved files:")

for path in sorted(OUT_DIR.glob("*.csv")):
    print(" -", path)

print()
print("=" * 80)
print("NO NEW REPRESENTATION OR CLASSIFIER WAS INTRODUCED")
print("STEP 85 PROTOCOL RECONSTRUCTED FOR FAILURE ANALYSIS")
print("=" * 80)