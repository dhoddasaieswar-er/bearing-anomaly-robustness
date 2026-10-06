# ============================================================
# STEP 129
# PAIRED VALIDATION OF CONDITION-RELATIVE CENTROID
#
# Compares:
#   1. Absolute STFT + Nearest Centroid
#   2. Condition-relative STFT + Nearest Centroid
#   3. Absolute + Condition-relative STFT + Nearest Centroid
#
# Strict LOBO:
#   - held-out bearing never enters training
#   - condition references use healthy TRAINING samples only
#   - StandardScaler fitted only on training data
#
# No statsmodels required.
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(os.environ.get("BEARING_FEATURE_ROOT", Path(__file__).resolve().parent))

STFT_FILE = os.path.join(
    BASE_DIR,
    "corrected_64khz_absolute_stft.csv"
)

print("=" * 80)
print("STEP 129 — PAIRED VALIDATION")
print("=" * 80)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(STFT_FILE)

print("\nDataset shape:", df.shape)

# ============================================================
# FEATURES
# ============================================================

exclude = {
    "Bearing",
    "Condition",
    "Label",
    "File",
    "RPM",
    "Rotation_Hz"
}

features = [
    c for c in df.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(df[c])
]

print("\nSTFT features:", len(features))
print(features)

X = df[features].to_numpy(dtype=float)

y = df["Label"].to_numpy(dtype=int)

bearings = df["Bearing"].to_numpy()

conditions = df["Condition"].to_numpy()

unique_bearings = sorted(
    np.unique(bearings)
)

unique_conditions = sorted(
    np.unique(conditions)
)

print("\nBearings:")
print(unique_bearings)

print("\nConditions:")
print(unique_conditions)

# ============================================================
# CONFIGURATIONS
# ============================================================

configs = [
    "Absolute_STFT_Centroid",
    "Condition_Relative_Centroid",
    "Absolute_Plus_Relative_Centroid"
]

predictions = {
    config: np.full(
        len(y),
        -1,
        dtype=int
    )
    for config in configs
}

# ============================================================
# LOBO
# ============================================================

print("\n" + "=" * 80)
print("STRICT LEAVE-ONE-BEARING-OUT EVALUATION")
print("=" * 80)

for held_out in unique_bearings:

    print("\n" + "-" * 80)
    print("HELD-OUT:", held_out)
    print("-" * 80)

    train_idx = bearings != held_out
    test_idx = bearings == held_out

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    cond_train = conditions[train_idx]
    cond_test = conditions[test_idx]

    # ========================================================
    # 1. ABSOLUTE STFT + CENTROID
    # ========================================================

    scaler_abs = StandardScaler()

    Xtr_abs = scaler_abs.fit_transform(
        X_train
    )

    Xte_abs = scaler_abs.transform(
        X_test
    )

    clf_abs = NearestCentroid()

    clf_abs.fit(
        Xtr_abs,
        y_train
    )

    pred_abs = clf_abs.predict(
        Xte_abs
    )

    predictions[
        "Absolute_STFT_Centroid"
    ][test_idx] = pred_abs

    # ========================================================
    # 2. CONDITION-RELATIVE STFT
    #
    # Reference:
    # median of HEALTHY TRAINING samples
    # for each operating condition.
    # ========================================================

    Xtr_rel_raw = np.zeros_like(
        X_train
    )

    Xte_rel_raw = np.zeros_like(
        X_test
    )

    for condition in unique_conditions:

        train_condition = (
            cond_train == condition
        )

        test_condition = (
            cond_test == condition
        )

        healthy_train_condition = (
            train_condition &
            (y_train == 0)
        )

        if np.sum(
            healthy_train_condition
        ) == 0:

            raise ValueError(
                "No healthy training samples "
                f"for condition {condition} "
                f"when holding out {held_out}"
            )

        reference = np.median(
            X_train[
                healthy_train_condition
            ],
            axis=0
        )

        denominator = (
            np.abs(reference) + 1e-12
        )

        Xtr_rel_raw[
            train_condition
        ] = (
            X_train[train_condition]
            - reference
        ) / denominator

        Xte_rel_raw[
            test_condition
        ] = (
            X_test[test_condition]
            - reference
        ) / denominator

    # --------------------------------------------------------
    # Scale relative representation
    # --------------------------------------------------------

    scaler_rel = StandardScaler()

    Xtr_rel = scaler_rel.fit_transform(
        Xtr_rel_raw
    )

    Xte_rel = scaler_rel.transform(
        Xte_rel_raw
    )

    clf_rel = NearestCentroid()

    clf_rel.fit(
        Xtr_rel,
        y_train
    )

    pred_rel = clf_rel.predict(
        Xte_rel
    )

    predictions[
        "Condition_Relative_Centroid"
    ][test_idx] = pred_rel

    # ========================================================
    # 3. ABSOLUTE + RELATIVE
    # ========================================================

    Xtr_both = np.hstack([
        Xtr_abs,
        Xtr_rel
    ])

    Xte_both = np.hstack([
        Xte_abs,
        Xte_rel
    ])

    clf_both = NearestCentroid()

    clf_both.fit(
        Xtr_both,
        y_train
    )

    pred_both = clf_both.predict(
        Xte_both
    )

    predictions[
        "Absolute_Plus_Relative_Centroid"
    ][test_idx] = pred_both

    # ========================================================
    # DISPLAY FOLD RESULT
    # ========================================================

    print(
        "Test samples:",
        np.sum(test_idx)
    )

    print(
        "True class distribution:",
        np.bincount(y_test)
    )

    print(
        "Absolute predictions:",
        np.bincount(pred_abs)
    )

    print(
        "Relative predictions:",
        np.bincount(pred_rel)
    )

    print(
        "Combined predictions:",
        np.bincount(pred_both)
    )

# ============================================================
# SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION SANITY CHECK")
print("=" * 80)

for config in configs:

    missing = np.sum(
        predictions[config] == -1
    )

    print(
        config,
        "missing predictions:",
        missing
    )

    if missing != 0:
        raise RuntimeError(
            f"Missing predictions for {config}"
        )

# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    pred
):

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy_score(
            y_true,
            pred
        ),

        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Healthy_FPR": (
            fp / (fp + tn)
        ),

        "Specificity": (
            tn / (tn + fp)
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

# ============================================================
# POOLED RESULTS
# ============================================================

print("\n" + "=" * 80)
print("POOLED RESULTS")
print("=" * 80)

pooled_rows = []

for config in configs:

    pred = predictions[config]

    m = calculate_metrics(
        y,
        pred
    )

    row = {
        "Configuration": config,
        **m
    }

    pooled_rows.append(row)

    print("\n" + config)

    print(
        f"Accuracy   : {m['Accuracy']:.6f}"
    )

    print(
        f"Precision  : {m['Precision']:.6f}"
    )

    print(
        f"Recall     : {m['Recall']:.6f}"
    )

    print(
        f"F1         : {m['F1']:.6f}"
    )

    print(
        f"Healthy FPR: {m['Healthy_FPR']:.6f}"
    )

    print(
        f"Specificity: {m['Specificity']:.6f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        np.array([
            [m["TN"], m["FP"]],
            [m["FN"], m["TP"]]
        ])
    )

pooled_df = pd.DataFrame(
    pooled_rows
)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Bearing": bearings,
    "Condition": conditions,
    "True_Label": y,

    "Absolute_STFT_Centroid":
        predictions[
            "Absolute_STFT_Centroid"
        ],

    "Condition_Relative_Centroid":
        predictions[
            "Condition_Relative_Centroid"
        ],

    "Absolute_Plus_Relative_Centroid":
        predictions[
            "Absolute_Plus_Relative_Centroid"
        ]
})

prediction_file = os.path.join(
    BASE_DIR,
    "nearest_centroid_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nSaved predictions:")
print(prediction_file)

# ============================================================
# PAIRED COMPARISON
# ============================================================

baseline = predictions[
    "Absolute_STFT_Centroid"
]

candidate = predictions[
    "Absolute_Plus_Relative_Centroid"
]

# ------------------------------------------------------------
# Basic changes
# ------------------------------------------------------------

changed = (
    baseline != candidate
)

base_correct = (
    baseline == y
)

candidate_correct = (
    candidate == y
)

print("\n" + "=" * 80)
print("PAIRED PREDICTION CHANGES")
print("=" * 80)

print(
    "Changed predictions:",
    np.sum(changed),
    "/",
    len(y)
)

print(
    "Change rate:",
    np.mean(changed)
)

print(
    "Baseline 0 -> Candidate 1:",
    np.sum(
        (baseline == 0) &
        (candidate == 1)
    )
)

print(
    "Baseline 1 -> Candidate 0:",
    np.sum(
        (baseline == 1) &
        (candidate == 0)
    )
)

print(
    "Wrong -> Correct:",
    np.sum(
        (~base_correct) &
        candidate_correct
    )
)

print(
    "Correct -> Wrong:",
    np.sum(
        base_correct &
        (~candidate_correct)
    )
)

print(
    "Correct -> Correct:",
    np.sum(
        base_correct &
        candidate_correct
    )
)

print(
    "Wrong -> Wrong:",
    np.sum(
        (~base_correct) &
        (~candidate_correct)
    )
)

# ============================================================
# MCNEMAR TEST
# PACKAGE-FREE
# ============================================================

both_correct = np.sum(
    base_correct &
    candidate_correct
)

baseline_only_correct = np.sum(
    base_correct &
    (~candidate_correct)
)

candidate_only_correct = np.sum(
    (~base_correct) &
    candidate_correct
)

both_wrong = np.sum(
    (~base_correct) &
    (~candidate_correct)
)

print("\n" + "=" * 80)
print("MCNEMAR PAIRED CORRECTNESS TABLE")
print("=" * 80)

print(
    "                     Candidate Correct   Candidate Wrong"
)

print(
    "Baseline Correct     ",
    both_correct,
    "                 ",
    baseline_only_correct
)

print(
    "Baseline Wrong       ",
    candidate_only_correct,
    "                 ",
    both_wrong
)

# ------------------------------------------------------------
# McNemar chi-square with continuity correction
#
# chi2 = (|b-c|-1)^2/(b+c)
#
# For b+c=0, statistic is zero.
# ------------------------------------------------------------

b = baseline_only_correct
c = candidate_only_correct

if (b + c) == 0:

    mcnemar_stat = 0.0
    mcnemar_p = 1.0

else:

    mcnemar_stat = (
        (abs(b - c) - 1) ** 2
        / (b + c)
    )

    # Chi-square survival function for 1 df:
    # p = erfc(sqrt(stat / 2))
    import math

    mcnemar_p = math.erfc(
        math.sqrt(
            mcnemar_stat / 2
        )
    )

print(
    "\nDiscordant baseline-only correct:",
    b
)

print(
    "Discordant candidate-only correct:",
    c
)

print(
    "McNemar chi-square:",
    mcnemar_stat
)

print(
    "McNemar p-value:",
    mcnemar_p
)

# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

print("\n" + "=" * 80)
print("PAIRED BOOTSTRAP — F1 DIFFERENCE")
print("=" * 80)

rng = np.random.default_rng(42)

N_BOOTSTRAP = 10000

n = len(y)

bootstrap_differences = np.empty(
    N_BOOTSTRAP
)

for i in range(
    N_BOOTSTRAP
):

    idx = rng.integers(
        0,
        n,
        size=n
    )

    base_f1 = f1_score(
        y[idx],
        baseline[idx],
        zero_division=0
    )

    cand_f1 = f1_score(
        y[idx],
        candidate[idx],
        zero_division=0
    )

    bootstrap_differences[i] = (
        cand_f1 - base_f1
    )

observed_difference = (
    pooled_df.loc[
        pooled_df["Configuration"]
        == "Absolute_Plus_Relative_Centroid",
        "F1"
    ].iloc[0]
    -
    pooled_df.loc[
        pooled_df["Configuration"]
        == "Absolute_STFT_Centroid",
        "F1"
    ].iloc[0]
)

ci_low = np.percentile(
    bootstrap_differences,
    2.5
)

ci_high = np.percentile(
    bootstrap_differences,
    97.5
)

print(
    "Observed F1 difference:",
    observed_difference
)

print(
    "Bootstrap 95% CI:",
    ci_low,
    "to",
    ci_high
)

# ============================================================
# BEARING-LEVEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("BEARING-LEVEL COMPARISON")
print("=" * 80)

bearing_rows = []

for bearing in unique_bearings:

    idx = (
        bearings == bearing
    )

    y_b = y[idx]

    base_b = baseline[idx]

    cand_b = candidate[idx]

    base_m = calculate_metrics(
        y_b,
        base_b
    )

    cand_m = calculate_metrics(
        y_b,
        cand_b
    )

    bearing_rows.append({
        "Bearing": bearing,

        "Baseline_FPR":
            base_m["Healthy_FPR"]
            if np.sum(y_b == 0) > 0
            else np.nan,

        "Candidate_FPR":
            cand_m["Healthy_FPR"]
            if np.sum(y_b == 0) > 0
            else np.nan,

        "Baseline_Recall":
            base_m["Recall"]
            if np.sum(y_b == 1) > 0
            else np.nan,

        "Candidate_Recall":
            cand_m["Recall"]
            if np.sum(y_b == 1) > 0
            else np.nan
    })

bearing_df = pd.DataFrame(
    bearing_rows
)

print(
    bearing_df.to_string(
        index=False
    )
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    BASE_DIR,
    "paired_validation_results.csv"
)

bearing_file = os.path.join(
    BASE_DIR,
    "bearing_comparison.csv"
)

pooled_df.to_csv(
    results_file,
    index=False
)

bearing_df.to_csv(
    bearing_file,
    index=False
)

# ============================================================
# FINAL SUMMARY FILE
# ============================================================

summary = pd.DataFrame([{
    "Baseline":
        "Absolute_STFT_Centroid",

    "Candidate":
        "Absolute_Plus_Relative_Centroid",

    "Baseline_F1":
        calculate_metrics(
            y,
            baseline
        )["F1"],

    "Candidate_F1":
        calculate_metrics(
            y,
            candidate
        )["F1"],

    "Observed_F1_Difference":
        observed_difference,

    "Bootstrap_CI_Low":
        ci_low,

    "Bootstrap_CI_High":
        ci_high,

    "McNemar_ChiSquare":
        mcnemar_stat,

    "McNemar_p":
        mcnemar_p,

    "Changed_Predictions":
        int(np.sum(changed)),

    "Change_Rate":
        float(np.mean(changed)),

    "Baseline_Only_Correct":
        int(b),

    "Candidate_Only_Correct":
        int(c)
}])

summary_file = os.path.join(
    BASE_DIR,
    "paired_validation_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(prediction_file)
print(results_file)
print(bearing_file)
print(summary_file)

print("\n" + "=" * 80)
print("STEP 129 COMPLETE")
print("=" * 80)