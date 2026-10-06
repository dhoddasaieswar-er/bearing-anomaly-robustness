from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 96 — FINAL RESULTS CONSISTENCY AUDIT
#
# FINAL EXPERIMENTAL AUDIT
#
# No new ML experiment.
# No retraining.
# No modification of experimental results.
# Only consistency checking.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "final_results_audit_step96"
OUT_DIR.mkdir(exist_ok=True)


print("=" * 80)
print("STEP 96 — FINAL RESULTS CONSISTENCY AUDIT")
print("=" * 80)


# ============================================================
# HELPER
# ============================================================

audit_rows = []


def record(test, status, detail):

    audit_rows.append({
        "Test": test,
        "Status": status,
        "Detail": detail
    })

    symbol = "PASS" if status == "PASS" else "FAIL"

    print(
        f"[{symbol}] {test}: {detail}"
    )


def safe_div(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# FILE PATHS
# ============================================================

FILES = {

    # Step 84
    "step84_comparison":
        BASE_DIR /
        "corrected_final_stft_comparison.csv",

    "step84_bearing":
        BASE_DIR /
        "corrected_bearing_results.csv",

    "step84_absolute":
        BASE_DIR /
        "corrected_64khz_absolute_stft.csv",

    "step84_order":
        BASE_DIR /
        "corrected_64khz_order_stft.csv",

    "step84_rms":
        BASE_DIR /
        "corrected_64khz_rms_stft.csv",

    # Step 85
    "step85_pooled":
        BASE_DIR /
        "classifier_control_pooled_results.csv",

    "step85_bearing":
        BASE_DIR /
        "classifier_control_bearing_results.csv",

    # Step 88
    "step88_profile":
        BASE_DIR /
        "robustness_profile_step88" /
        "step88_final_multidimensional_robustness_profile.csv",

    # Step 92
    "step92_representation":
        BASE_DIR /
        "representation_classifier_analysis" /
        "step92_representation_classifier_results.csv",

    "step92_classifier_effect":
        BASE_DIR /
        "representation_classifier_analysis" /
        "step92_classifier_effect_by_representation.csv",

    # Step 93
    "step93_predictions":
        BASE_DIR /
        "failure_case_analysis_step93" /
        "step93_signal_level_predictions.csv",

    "step93_failures":
        BASE_DIR /
        "failure_case_analysis_step93" /
        "step93_all_failure_cases.csv",

    "step93_persistent":
        BASE_DIR /
        "failure_case_analysis_step93" /
        "step93_persistent_failure_cases.csv",

    # Step 94
    "step94_full":
        BASE_DIR /
        "failure_case_characterization_step94" /
        "step94_full_failure_characterization.csv",

    "step94_persistent":
        BASE_DIR /
        "failure_case_characterization_step94" /
        "step94_persistent_failure_characterization.csv",

    # Step 95
    "step95_bearing":
        BASE_DIR /
        "failure_concentration_analysis_step95" /
        "step95_failures_by_bearing.csv",

    "step95_condition":
        BASE_DIR /
        "failure_concentration_analysis_step95" /
        "step95_failures_by_condition.csv",

    "step95_concentration":
        BASE_DIR /
        "failure_concentration_analysis_step95" /
        "step95_error_concentration_summary.csv",

    "step95_persistent_profile":
        BASE_DIR /
        "failure_concentration_analysis_step95" /
        "step95_bearing_persistent_profile.csv",
}


# ============================================================
# 1. REQUIRED FILE CHECK
# ============================================================

print()
print("=" * 80)
print("1. REQUIRED FILE CHECK")
print("=" * 80)


missing_files = []


for key, path in FILES.items():

    if path.exists():

        print(
            f"[PASS] {key}: {path.name}"
        )

    else:

        print(
            f"[FAIL] {key}: MISSING"
        )

        missing_files.append(key)


if len(missing_files) == 0:

    record(
        "Required files",
        "PASS",
        f"All {len(FILES)} required result files found"
    )

else:

    record(
        "Required files",
        "FAIL",
        f"Missing {len(missing_files)} files: {missing_files}"
    )


# ============================================================
# CORE FILE CHECK
# ============================================================

CORE_KEYS = [

    "step85_pooled",
    "step85_bearing",
    "step88_profile",
    "step93_failures",
    "step93_persistent",
    "step95_bearing",
    "step95_condition",
    "step95_concentration",
]


missing_core = [

    key

    for key in CORE_KEYS

    if not FILES[key].exists()

]


if missing_core:

    print()
    print("=" * 80)
    print("AUDIT STOPPED")
    print("=" * 80)

    print(
        "Missing core files:",
        missing_core
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        OUT_DIR /
        "step96_audit_results.csv",
        index=False
    )

    raise SystemExit(
        "Cannot complete final audit because core files are missing."
    )


# ============================================================
# LOAD RESULTS
# ============================================================

step85 = pd.read_csv(
    FILES["step85_pooled"]
)

step85_bearing = pd.read_csv(
    FILES["step85_bearing"]
)

step88 = pd.read_csv(
    FILES["step88_profile"]
)

step93_predictions = pd.read_csv(
    FILES["step93_predictions"]
)

step93_failures = pd.read_csv(
    FILES["step93_failures"]
)

step93_persistent = pd.read_csv(
    FILES["step93_persistent"]
)

step95_bearing = pd.read_csv(
    FILES["step95_bearing"]
)

step95_condition = pd.read_csv(
    FILES["step95_condition"]
)

step95_concentration = pd.read_csv(
    FILES["step95_concentration"]
)

step95_persistent_profile = pd.read_csv(
    FILES["step95_persistent_profile"]
)


# ============================================================
# 2. DATASET SIZE AUDIT
# ============================================================

print()
print("=" * 80)
print("2. DATASET SIZE AUDIT")
print("=" * 80)


if len(step85_bearing) == 80:

    record(
        "Step 85 bearing-level rows",
        "PASS",
        "80 rows = 8 experiments × 10 bearings"
    )

else:

    record(
        "Step 85 bearing-level rows",
        "FAIL",
        f"Expected 80, found {len(step85_bearing)}"
    )


# ============================================================
# 3. REPRESENTATION × CLASSIFIER
# ============================================================

print()
print("=" * 80)
print("3. REPRESENTATION × CLASSIFIER AUDIT")
print("=" * 80)


expected_representations = {

    "Time-domain",

    "Absolute STFT",

    "Order STFT",

    "RMS-normalized STFT",
}


expected_classifiers = {

    "Random Forest",

    "SVM",
}


actual_representations = set(
    step85["Representation"].unique()
)

actual_classifiers = set(
    step85["Classifier"].unique()
)


if actual_representations == expected_representations:

    record(
        "Representation count",
        "PASS",
        "All 4 expected representations found"
    )

else:

    record(
        "Representation count",
        "FAIL",
        f"Found: {sorted(actual_representations)}"
    )


if actual_classifiers == expected_classifiers:

    record(
        "Classifier count",
        "PASS",
        "Random Forest and SVM found"
    )

else:

    record(
        "Classifier count",
        "FAIL",
        f"Found: {sorted(actual_classifiers)}"
    )


actual_combinations = (

    step85[
        [
            "Representation",
            "Classifier"
        ]
    ]
    .drop_duplicates()
    .shape[0]

)


if actual_combinations == 8:

    record(
        "Representation × classifier combinations",
        "PASS",
        "8 combinations"
    )

else:

    record(
        "Representation × classifier combinations",
        "FAIL",
        f"Expected 8, found {actual_combinations}"
    )


# ============================================================
# 4. POOLED RESULTS
# ============================================================

print()
print("=" * 80)
print("4. POOLED RESULT AUDIT")
print("=" * 80)


required_metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1",
]


missing_metrics = [

    c

    for c in required_metrics

    if c not in step85.columns

]


if len(missing_metrics) == 0:

    record(
        "Pooled metric columns",
        "PASS",
        "Accuracy, Precision, Recall and F1 present"
    )

else:

    record(
        "Pooled metric columns",
        "FAIL",
        f"Missing: {missing_metrics}"
    )


if len(step85) == 8:

    record(
        "Pooled result rows",
        "PASS",
        "8 representation-classifier results"
    )

else:

    record(
        "Pooled result rows",
        "FAIL",
        f"Expected 8, found {len(step85)}"
    )


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

print()
print("=" * 80)
print("5. CONFUSION-MATRIX CONSISTENCY")
print("=" * 80)


cm_columns = [

    "TN",

    "FP",

    "FN",

    "TP",
]


missing_cm = [

    c

    for c in cm_columns

    if c not in step85.columns

]


if len(missing_cm) == 0:

    cm_sum = (

        step85["TN"]

        + step85["FP"]

        + step85["FN"]

        + step85["TP"]

    )


    if (cm_sum == 800).all():

        record(
            "Pooled prediction count",
            "PASS",
            "Every experiment contains exactly 800 predictions"
        )

    else:

        record(
            "Pooled prediction count",
            "FAIL",
            f"Invalid totals: "
            f"{cm_sum[cm_sum != 800].tolist()}"
        )

else:

    record(
        "Confusion-matrix columns",
        "FAIL",
        f"Missing: {missing_cm}"
    )


# ============================================================
# 6. METRIC RECOMPUTATION
# ============================================================

print()
print("=" * 80)
print("6. METRIC RECOMPUTATION")
print("=" * 80)


metric_errors = []


for _, row in step85.iterrows():

    tn = row["TN"]

    fp = row["FP"]

    fn = row["FN"]

    tp = row["TP"]


    accuracy = safe_div(

        tn + tp,

        tn + fp + fn + tp

    )


    precision = safe_div(

        tp,

        tp + fp

    )


    recall = safe_div(

        tp,

        tp + fn

    )


    f1 = safe_div(

        2 * precision * recall,

        precision + recall

    )


    calculated = {

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,
    }


    for metric, value in calculated.items():

        reported = row[metric]


        if not np.isclose(

            reported,

            value,

            atol=1e-4

        ):

            metric_errors.append({

                "Representation":
                    row["Representation"],

                "Classifier":
                    row["Classifier"],

                "Metric":
                    metric,

                "Reported":
                    reported,

                "Calculated":
                    value,

            })


if len(metric_errors) == 0:

    record(
        "Metric recomputation",
        "PASS",
        "All pooled metrics match their confusion matrices"
    )

else:

    record(
        "Metric recomputation",
        "FAIL",
        f"{len(metric_errors)} mismatches"
    )


# ============================================================
# 7. FAILURE COUNTS
# ============================================================

print()
print("=" * 80)
print("7. FAILURE COUNT AUDIT")
print("=" * 80)


if len(step93_failures) == 2052:

    record(
        "Step 93 failure records",
        "PASS",
        "2052 failure records"
    )

else:

    record(
        "Step 93 failure records",
        "FAIL",
        f"Expected 2052, found {len(step93_failures)}"
    )


if len(step93_persistent) == 237:

    record(
        "Step 93 persistent failures",
        "PASS",
        "237 persistent failure records"
    )

else:

    record(
        "Step 93 persistent failures",
        "FAIL",
        f"Expected 237, found {len(step93_persistent)}"
    )


# ============================================================
# 8. FAILURE TYPE
# ============================================================

print()
print("=" * 80)
print("8. FAILURE TYPE AUDIT")
print("=" * 80)


failure_types = set(

    step93_failures[
        "Failure_Type"
    ].unique()

)


expected_failure_types = {

    "False Positive",

    "False Negative",
}


if failure_types == expected_failure_types:

    record(
        "Failure types",
        "PASS",
        "False Positive and False Negative only"
    )

else:

    record(
        "Failure types",
        "FAIL",
        f"Found: {sorted(failure_types)}"
    )


# ============================================================
# 9. BEARING FAILURE TOTAL
# ============================================================

print()
print("=" * 80)
print("9. BEARING FAILURE TOTAL AUDIT")
print("=" * 80)


bearing_total = (

    step95_bearing[
        "Failure_Records"
    ]
    .sum()

)


if bearing_total == 2052:

    record(
        "Bearing failure total",
        "PASS",
        "Bearing-level failures sum to 2052"
    )

else:

    record(
        "Bearing failure total",
        "FAIL",
        f"Sum = {bearing_total}"
    )


# ============================================================
# 10. CONDITION FAILURE TOTAL
# ============================================================

condition_total = (

    step95_condition[
        "Failure_Records"
    ]
    .sum()

)


if condition_total == 2052:

    record(
        "Condition failure total",
        "PASS",
        "Condition-level failures sum to 2052"
    )

else:

    record(
        "Condition failure total",
        "FAIL",
        f"Sum = {condition_total}"
    )


# ============================================================
# 11. TOP-3 ERROR CONCENTRATION
# ============================================================

print()
print("=" * 80)
print("10. TOP-3 ERROR CONCENTRATION AUDIT")
print("=" * 80)


reported_total = (

    step95_concentration[
        "Total_Failure_Records"
    ]
    .iloc[0]

)


reported_top3 = (

    step95_concentration[
        "Top_3_Failure_Records"
    ]
    .iloc[0]

)


reported_percentage = (

    step95_concentration[
        "Top_3_Percentage_of_All_Failures"
    ]
    .iloc[0]

)


calculated_percentage = (

    reported_top3

    / reported_total

    * 100

)


if reported_total == 2052:

    record(
        "Top-3 total denominator",
        "PASS",
        "2052"
    )

else:

    record(
        "Top-3 total denominator",
        "FAIL",
        str(reported_total)
    )


if np.isclose(

    reported_percentage,

    calculated_percentage,

    atol=1e-5

):

    record(
        "Top-3 percentage",
        "PASS",
        f"{reported_percentage:.2f}%"
    )

else:

    record(
        "Top-3 percentage",
        "FAIL",
        "Percentage mismatch"
    )


reported_top3_bearings = (

    str(

        step95_concentration[
            "Top_3_Bearings"
        ]
        .iloc[0]

    )
    .replace(" ", "")
    .split(",")

)


expected_top3_bearings = {

    "KI05",

    "K002",

    "KA03",
}


if set(reported_top3_bearings) == expected_top3_bearings:

    record(
        "Top-3 failure bearings",
        "PASS",
        "KI05, K002 and KA03"
    )

else:

    record(
        "Top-3 failure bearings",
        "FAIL",
        str(reported_top3_bearings)
    )


persistent_total = (

    step95_concentration[
        "Persistent_Failure_Records"
    ]
    .iloc[0]

)


if persistent_total == 237:

    record(
        "Persistent failure total",
        "PASS",
        "237"
    )

else:

    record(
        "Persistent failure total",
        "FAIL",
        str(persistent_total)
    )


# ============================================================
# 12. BEARING COVERAGE
# ============================================================

print()
print("=" * 80)
print("11. BEARING-LEVEL STATISTICAL COVERAGE")
print("=" * 80)


expected_bearings = {

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
}


actual_bearings = set(

    step95_bearing[
        "Bearing"
    ].unique()

)


missing_from_failure_table = (

    expected_bearings

    - actual_bearings

)


# K003 has zero failures and is therefore absent
# from the failure-only Step 95 table.

if missing_from_failure_table == {"K003"}:

    record(
        "Bearing coverage",
        "PASS",
        "K003 has zero failure records and is therefore "
        "absent from the failure-only Step 95 table; "
        "all other bearings are represented"
    )

elif len(missing_from_failure_table) == 0:

    record(
        "Bearing coverage",
        "PASS",
        "All 10 bearings represented"
    )

else:

    record(
        "Bearing coverage",
        "FAIL",
        f"Unexpected missing bearings: "
        f"{sorted(missing_from_failure_table)}"
    )


# ============================================================
# 13. OPERATING CONDITION
# ============================================================

print()
print("=" * 80)
print("12. OPERATING-CONDITION COVERAGE")
print("=" * 80)


expected_conditions = {

    "N09_M07_F10",

    "N15_M01_F10",

    "N15_M07_F04",

    "N15_M07_F10",
}


actual_conditions = set(

    step95_condition[
        "Condition"
    ].unique()

)


if actual_conditions == expected_conditions:

    record(
        "Operating-condition coverage",
        "PASS",
        "All 4 conditions represented"
    )

else:

    record(
        "Operating-condition coverage",
        "FAIL",
        f"Found: {sorted(actual_conditions)}"
    )


# ============================================================
# 14. ROBUSTNESS PROFILE
# ============================================================

print()
print("=" * 80)
print("13. ROBUSTNESS PROFILE AUDIT")
print("=" * 80)


if len(step88) == 103:

    record(
        "Step 88 robustness profile",
        "PASS",
        "103 rows"
    )

else:

    record(
        "Step 88 robustness profile",
        "FAIL",
        f"Expected 103, found {len(step88)}"
    )


if "Dimension" in step88.columns:

    dimensions = set(

        step88[
            "Dimension"
        ].unique()

    )


    expected_dimensions = {

        "Overall detection",

        "Damaged-bearing robustness",

        "Healthy-bearing robustness",

        "Synthetic noise robustness",

        "Synthetic anomaly-severity robustness",

        "Synthetic operating-condition robustness",

        "Classifier sensitivity",
    }


    if dimensions == expected_dimensions:

        record(
            "Robustness dimensions",
            "PASS",
            "All 7 dimensions present"
        )

    else:

        record(
            "Robustness dimensions",
            "FAIL",
            f"Found: {sorted(dimensions)}"
        )


# ============================================================
# 15. CLASSIFIER ANALYSIS
# ============================================================

print()
print("=" * 80)
print("14. CLASSIFIER ANALYSIS AUDIT")
print("=" * 80)


if FILES["step92_representation"].exists():

    step92 = pd.read_csv(

        FILES[
            "step92_representation"
        ]

    )


    if len(step92) == 8:

        record(
            "Step 92 representation-classifier results",
            "PASS",
            "8 combinations"
        )

    else:

        record(
            "Step 92 representation-classifier results",
            "FAIL",
            f"Expected 8, found {len(step92)}"
        )

else:

    record(
        "Step 92 representation-classifier results",
        "FAIL",
        "File missing"
    )


# ============================================================
# 16. SUPERSEDED RESULT CHECK
# ============================================================

print()
print("=" * 80)
print("15. SUPERSEDED-RESULT CHECK")
print("=" * 80)


superseded_files = [

    "step61",

    "step63",

    "step67",

    "step72",

    "step73",

]


final_package = (

    BASE_DIR /

    "FINAL_RESEARCH_RESULTS"

)


found_superseded = []


if final_package.exists():

    for path in final_package.rglob("*"):

        if not path.is_file():

            continue


        filename = path.name.lower()


        for token in superseded_files:

            if token in filename:

                found_superseded.append(
                    str(path)
                )


if len(found_superseded) == 0:

    record(
        "Superseded results in final package",
        "PASS",
        "No Step 61/63/67/72/73 result files found"
    )

else:

    record(
        "Superseded results in final package",
        "FAIL",
        f"Found {len(found_superseded)} files"
    )


# ============================================================
# 17. STEP 93 SIGNAL-LEVEL SANITY
# ============================================================

print()
print("=" * 80)
print("16. STEP 93 SIGNAL-LEVEL SANITY CHECK")
print("=" * 80)


expected_predictions = 6400


if len(step93_predictions) == expected_predictions:

    record(
        "Step 93 prediction count",
        "PASS",
        "6400 = 800 signals × 8 configurations"
    )

else:

    record(
        "Step 93 prediction count",
        "FAIL",
        f"Expected 6400, found {len(step93_predictions)}"
    )


# ------------------------------------------------------------
# CORRECTED CHECK
#
# The prediction file contains a Failure_Type value for
# correct predictions as well as failures.
#
# Therefore we MUST NOT use .notna().
#
# Instead count explicitly:
#   False Positive
#   False Negative
# ------------------------------------------------------------

if "Failure_Type" in step93_predictions.columns:

    prediction_failure_count = (

        step93_predictions[
            "Failure_Type"
        ]
        .isin([
            "False Positive",
            "False Negative"
        ])
        .sum()

    )


    if prediction_failure_count == 2052:

        record(
            "Step 93 prediction/failure agreement",
            "PASS",
            "2052 False Positive/False Negative records"
        )

    else:

        record(
            "Step 93 prediction/failure agreement",
            "FAIL",
            f"Expected 2052, found "
            f"{prediction_failure_count}"
        )

else:

    record(
        "Step 93 prediction/failure agreement",
        "FAIL",
        "Failure_Type column missing"
    )


# ============================================================
# 18. FALSE POSITIVE / FALSE NEGATIVE
# ============================================================

print()
print("=" * 80)
print("17. FALSE POSITIVE / FALSE NEGATIVE CHECK")
print("=" * 80)


fp_count = (

    step93_failures[
        "Failure_Type"
    ]
    .eq("False Positive")
    .sum()

)


fn_count = (

    step93_failures[
        "Failure_Type"
    ]
    .eq("False Negative")
    .sum()

)


if fp_count + fn_count == 2052:

    record(
        "FP + FN total",
        "PASS",
        f"FP={fp_count}, FN={fn_count}, total=2052"
    )

else:

    record(
        "FP + FN total",
        "FAIL",
        f"FP={fp_count}, FN={fn_count}, "
        f"total={fp_count + fn_count}"
    )


# ============================================================
# 19. UNIQUE FAILED SIGNALS
# ============================================================

print()
print("=" * 80)
print("18. UNIQUE FAILED SIGNAL CHECK")
print("=" * 80)


if "File" in step93_failures.columns:

    unique_failed_signals = (

        step93_failures[
            "File"
        ]
        .nunique()

    )


    if unique_failed_signals == 439:

        record(
            "Unique failed signals",
            "PASS",
            "439 unique signals"
        )

    else:

        record(
            "Unique failed signals",
            "FAIL",
            f"Expected 439, found "
            f"{unique_failed_signals}"
        )


# ============================================================
# 20. PERSISTENT FAILURE BEARING CHECK
# ============================================================

print()
print("=" * 80)
print("19. PERSISTENT FAILURE BEARING CHECK")
print("=" * 80)


persistent_profile_bearings = set(

    step95_persistent_profile[
        "Bearing"
    ].unique()

)


missing_persistent_bearings = (

    expected_bearings

    - persistent_profile_bearings

)


# K003 has zero failures and therefore zero persistent
# failures. It is legitimately absent from this
# failure-only persistent profile.

if missing_persistent_bearings == {"K003"}:

    record(
        "Persistent-profile bearing coverage",
        "PASS",
        "K003 has zero persistent failures and is "
        "therefore absent from the persistent-failure "
        "profile; all other bearings are represented"
    )

elif len(missing_persistent_bearings) == 0:

    record(
        "Persistent-profile bearing coverage",
        "PASS",
        "All 10 bearings represented"
    )

else:

    record(
        "Persistent-profile bearing coverage",
        "FAIL",
        f"Unexpected missing bearings: "
        f"{sorted(missing_persistent_bearings)}"
    )


# ============================================================
# 21. FINAL AUDIT TABLE
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


audit_file = (

    OUT_DIR /

    "step96_audit_results.csv"

)


audit_df.to_csv(

    audit_file,

    index=False

)


# ============================================================
# 22. FINAL COUNTS
# ============================================================

pass_count = (

    audit_df[
        "Status"
    ]
    .eq("PASS")
    .sum()

)


fail_count = (

    audit_df[
        "Status"
    ]
    .eq("FAIL")
    .sum()

)


print()
print("=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)


print(
    f"PASS: {pass_count}"
)


print(
    f"FAIL: {fail_count}"
)


# ============================================================
# 23. FINAL DECISION
# ============================================================

if fail_count == 0:

    final_status = "PASS"


    print()
    print("=" * 80)
    print("STEP 96 — FINAL EXPERIMENTAL AUDIT: PASS")
    print("=" * 80)

    print()

    print(
        "All frozen experimental results are internally consistent."
    )

    print(
        "No new ML experiment was performed."
    )

    print(
        "The experimental phase can now be frozen."
    )

else:

    final_status = "FAIL"


    print()
    print("=" * 80)
    print("STEP 96 — FINAL EXPERIMENTAL AUDIT: FAIL")
    print("=" * 80)

    print()

    print(
        "Review the failed checks above."
    )


# ============================================================
# 24. SAVE FINAL STATUS
# ============================================================

status_df = pd.DataFrame([

    {

        "Step":
            "Step 96",

        "Audit_Status":
            final_status,

        "Pass_Count":
            pass_count,

        "Fail_Count":
            fail_count,

        "Experimental_Phase_Frozen":
            final_status == "PASS",

    }

])


status_file = (

    OUT_DIR /

    "step96_final_audit_status.csv"

)


status_df.to_csv(

    status_file,

    index=False

)


# ============================================================
# 25. HUMAN-READABLE SUMMARY
# ============================================================

summary_file = (

    OUT_DIR /

    "STEP96_FINAL_AUDIT_SUMMARY.txt"

)


with open(

    summary_file,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "STEP 96 — FINAL RESULTS CONSISTENCY AUDIT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Final status: {final_status}\n"
    )

    f.write(
        f"PASS checks: {pass_count}\n"
    )

    f.write(
        f"FAIL checks: {fail_count}\n\n"
    )

    if final_status == "PASS":

        f.write(
            "Experimental phase is frozen.\n"
        )

        f.write(
            "No new ML experiment was performed during the audit.\n"
        )

        f.write(
            "Validated results from Steps 84–95 are internally consistent.\n"
        )

    else:

        f.write(
            "Experimental phase is NOT frozen.\n"
        )

        f.write(
            "Review failed checks in step96_audit_results.csv.\n"
        )


# ============================================================
# 26. OUTPUT LOCATIONS
# ============================================================

print()
print(
    "Audit files saved to:"
)

print(
    OUT_DIR
)

print()

print(
    "Main audit CSV:"
)

print(
    audit_file
)

print()

print(
    "Final status CSV:"
)

print(
    status_file
)

print()

print(
    "Human-readable summary:"
)

print(
    summary_file
)

print()
print("=" * 80)
print("STEP 96 COMPLETE")
print("=" * 80)