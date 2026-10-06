from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 95 — FAILURE CONCENTRATION ANALYSIS
# CORRECTED VERSION
#
# Uses existing Step 93 results only.
# NO new ML training.
# NO new dataset.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "failure_case_analysis_step93"
OUT_DIR = BASE_DIR / "failure_concentration_analysis_step95"

OUT_DIR.mkdir(exist_ok=True)

FAILURE_FILE = INPUT_DIR / "step93_all_failure_cases.csv"
PERSISTENT_FILE = INPUT_DIR / "step93_persistent_failure_cases.csv"

if not FAILURE_FILE.exists():
    raise FileNotFoundError(f"Missing:\n{FAILURE_FILE}")

if not PERSISTENT_FILE.exists():
    raise FileNotFoundError(f"Missing:\n{PERSISTENT_FILE}")


# ============================================================
# LOAD
# ============================================================

failures = pd.read_csv(FAILURE_FILE)
persistent = pd.read_csv(PERSISTENT_FILE)

print("=" * 80)
print("STEP 95 — FAILURE CONCENTRATION ANALYSIS")
print("=" * 80)

print()
print("Total failure records:", len(failures))
print("Persistent failure records:", len(persistent))

print()
print("Available Step 93 columns:")
print(list(failures.columns))


# ============================================================
# DERIVE BEARING / CONDITION / LABEL
# FROM EXISTING FILENAME
#
# Example:
# N15_M07_F10_K002_1.mat
#
# Bearing = K002
# Condition = N15_M07_F10
# ============================================================

def extract_metadata(df):

    df = df.copy()

    # --------------------------------------------------------
    # Bearing
    # --------------------------------------------------------

    if "Bearing" not in df.columns:

        df["Bearing"] = (
            df["File"]
            .astype(str)
            .str.extract(
                r'_(K\d{3}|KA\d{2}|KI\d{2})_',
                expand=False
            )
        )

    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    if "Condition" not in df.columns:

        filename = df["File"].astype(str)

        df["Condition"] = (
            filename
            .str.extract(
                r'^(N\d+_M\d+_F\d+)_',
                expand=False
            )
        )

    # --------------------------------------------------------
    # Label
    #
    # Healthy:
    # K001-K006
    #
    # Damaged:
    # KA01, KA03, KA04, KI05
    # --------------------------------------------------------

    if "Label" not in df.columns:

        damaged = {
            "KA01",
            "KA03",
            "KA04",
            "KI05",
        }

        df["Label"] = (
            df["Bearing"]
            .isin(damaged)
            .astype(int)
        )

    return df


failures = extract_metadata(failures)
persistent = extract_metadata(persistent)


# ============================================================
# VERIFY EXTRACTION
# ============================================================

print()
print("Derived metadata check:")

print(
    failures[
        [
            "File",
            "Bearing",
            "Condition",
            "Label"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# VALIDATE
# ============================================================

required_columns = {
    "File",
    "Bearing",
    "Condition",
    "Label",
    "Failure_Type",
    "Representation",
    "Classifier",
}

missing = required_columns - set(failures.columns)

if missing:
    raise ValueError(
        f"Still missing required columns: {sorted(missing)}"
    )


# ============================================================
# 1. FAILURES BY BEARING
# ============================================================

total_failures = len(failures)

bearing_failures = (
    failures
    .groupby(
        ["Bearing", "Label"]
    )
    .size()
    .reset_index(
        name="Failure_Records"
    )
)

bearing_failures[
    "Percentage_of_All_Failures"
] = (
    bearing_failures["Failure_Records"]
    / total_failures
    * 100
)

bearing_failures = bearing_failures.sort_values(
    "Failure_Records",
    ascending=False
)

bearing_failures.to_csv(
    OUT_DIR / "step95_failures_by_bearing.csv",
    index=False
)


# ============================================================
# 2. FAILURE TYPE BY BEARING
# ============================================================

bearing_failure_type = (
    failures
    .groupby(
        [
            "Bearing",
            "Label",
            "Failure_Type"
        ]
    )
    .size()
    .reset_index(
        name="Failure_Records"
    )
)

bearing_failure_type.to_csv(
    OUT_DIR / "step95_failure_type_by_bearing.csv",
    index=False
)


# ============================================================
# 3. FAILURES BY OPERATING CONDITION
# ============================================================

condition_failures = (
    failures
    .groupby("Condition")
    .size()
    .reset_index(
        name="Failure_Records"
    )
)

condition_failures[
    "Percentage_of_All_Failures"
] = (
    condition_failures["Failure_Records"]
    / total_failures
    * 100
)

condition_failures = condition_failures.sort_values(
    "Failure_Records",
    ascending=False
)

condition_failures.to_csv(
    OUT_DIR / "step95_failures_by_condition.csv",
    index=False
)


# ============================================================
# 4. BEARING × CONDITION
# ============================================================

bearing_condition = pd.crosstab(
    failures["Bearing"],
    failures["Condition"]
)

bearing_condition.to_csv(
    OUT_DIR / "step95_bearing_condition_failure_matrix.csv"
)


# ============================================================
# 5. BEARING × CONDITION × FAILURE TYPE
# ============================================================

bearing_condition_type = (
    failures
    .groupby(
        [
            "Bearing",
            "Condition",
            "Failure_Type"
        ]
    )
    .size()
    .reset_index(
        name="Failure_Records"
    )
)

bearing_condition_type.to_csv(
    OUT_DIR / "step95_bearing_condition_failure_type.csv",
    index=False
)


# ============================================================
# 6. PERSISTENT FAILURES BY BEARING
# ============================================================

persistent_counts = (
    persistent
    .groupby(
        ["Bearing", "Label"]
    )
    .size()
    .reset_index(
        name="Persistent_Failure_Records"
    )
)

persistent_counts[
    "Percentage_of_Persistent_Failures"
] = (
    persistent_counts[
        "Persistent_Failure_Records"
    ]
    / len(persistent)
    * 100
)

persistent_counts = persistent_counts.sort_values(
    "Persistent_Failure_Records",
    ascending=False
)

persistent_counts.to_csv(
    OUT_DIR / "step95_persistent_failures_by_bearing.csv",
    index=False
)


# ============================================================
# 7. PERSISTENT FAILURES BY CONDITION
# ============================================================

persistent_condition = (
    persistent
    .groupby("Condition")
    .size()
    .reset_index(
        name="Persistent_Failure_Records"
    )
)

persistent_condition[
    "Percentage_of_Persistent_Failures"
] = (
    persistent_condition[
        "Persistent_Failure_Records"
    ]
    / len(persistent)
    * 100
)

persistent_condition = persistent_condition.sort_values(
    "Persistent_Failure_Records",
    ascending=False
)

persistent_condition.to_csv(
    OUT_DIR / "step95_persistent_failures_by_condition.csv",
    index=False
)


# ============================================================
# 8. TARGET BEARINGS
# ============================================================

target_bearings = [
    "K002",
    "KI05",
    "KA03",
]

target_failures = failures[
    failures["Bearing"].isin(target_bearings)
].copy()

target_summary = (
    target_failures
    .groupby(
        [
            "Bearing",
            "Condition",
            "Failure_Type"
        ]
    )
    .size()
    .reset_index(
        name="Failure_Records"
    )
)

target_summary.to_csv(
    OUT_DIR / "step95_target_bearing_condition_analysis.csv",
    index=False
)


# ============================================================
# 9. FAILURE COUNT PER SIGNAL
# ============================================================

signal_failure_count = (
    failures
    .groupby(
        [
            "File",
            "Bearing",
            "Condition",
            "Label"
        ]
    )
    .agg(
        Failure_Configurations=(
            "Failure_Type",
            "count"
        ),
        Representation_Count=(
            "Representation",
            "nunique"
        ),
        Classifier_Count=(
            "Classifier",
            "nunique"
        ),
    )
    .reset_index()
)

signal_failure_count[
    "Failure_Fraction_of_8"
] = (
    signal_failure_count[
        "Failure_Configurations"
    ] / 8
)

signal_failure_count.to_csv(
    OUT_DIR / "step95_signal_failure_concentration.csv",
    index=False
)


# ============================================================
# 10. BEARING PERSISTENT PROFILE
# ============================================================

signal_failure_count["Persistent"] = (
    signal_failure_count[
        "Failure_Configurations"
    ] >= 4
)

bearing_persistent_profile = (
    signal_failure_count
    .groupby(
        ["Bearing", "Label"]
    )
    .agg(
        Signals=("File", "count"),
        Persistent_Signals=(
            "Persistent",
            "sum"
        ),
        Mean_Failure_Configurations=(
            "Failure_Configurations",
            "mean"
        ),
        Median_Failure_Configurations=(
            "Failure_Configurations",
            "median"
        ),
        Maximum_Failure_Configurations=(
            "Failure_Configurations",
            "max"
        ),
    )
    .reset_index()
)

bearing_persistent_profile[
    "Persistent_Signal_Percentage"
] = (
    bearing_persistent_profile[
        "Persistent_Signals"
    ]
    /
    bearing_persistent_profile["Signals"]
    * 100
)

bearing_persistent_profile.to_csv(
    OUT_DIR / "step95_bearing_persistent_profile.csv",
    index=False
)


# ============================================================
# 11. CONDITION PERSISTENT PROFILE
# ============================================================

condition_profile = (
    signal_failure_count
    .groupby("Condition")
    .agg(
        Signals=("File", "count"),
        Persistent_Signals=(
            "Persistent",
            "sum"
        ),
        Mean_Failure_Configurations=(
            "Failure_Configurations",
            "mean"
        ),
    )
    .reset_index()
)

condition_profile[
    "Persistent_Signal_Percentage"
] = (
    condition_profile[
        "Persistent_Signals"
    ]
    /
    condition_profile["Signals"]
    * 100
)

condition_profile.to_csv(
    OUT_DIR / "step95_condition_persistent_profile.csv",
    index=False
)


# ============================================================
# 12. ERROR CONCENTRATION — TOP 3 BEARINGS
# ============================================================

top_three = bearing_failures.head(3)

top_three_failures = (
    top_three["Failure_Records"].sum()
)

top_three_percentage = (
    top_three_failures
    / total_failures
    * 100
)

concentration_summary = pd.DataFrame([
    {
        "Total_Failure_Records": total_failures,
        "Top_3_Bearings": ",".join(
            top_three["Bearing"].tolist()
        ),
        "Top_3_Failure_Records":
            top_three_failures,
        "Top_3_Percentage_of_All_Failures":
            top_three_percentage,
        "Persistent_Failure_Records":
            len(persistent),
    }
])

concentration_summary.to_csv(
    OUT_DIR / "step95_error_concentration_summary.csv",
    index=False
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("FAILURES BY BEARING")
print("=" * 80)

print(
    bearing_failures.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("FAILURES BY OPERATING CONDITION")
print("=" * 80)

print(
    condition_failures.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("BEARING × CONDITION FAILURE MATRIX")
print("=" * 80)

print(
    bearing_condition.to_string()
)


print()
print("=" * 80)
print("TARGET BEARINGS: K002 / KI05 / KA03")
print("=" * 80)

print(
    target_summary.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("BEARING PERSISTENT-FAILURE PROFILE")
print("=" * 80)

print(
    bearing_persistent_profile.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("CONDITION PERSISTENT-FAILURE PROFILE")
print("=" * 80)

print(
    condition_profile.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("ERROR CONCENTRATION")
print("=" * 80)

print(
    concentration_summary.to_string(
        index=False
    )
)


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(failures) == 2052
assert len(persistent) == 237

assert set(
    failures["Condition"].unique()
) == {
    "N09_M07_F10",
    "N15_M01_F10",
    "N15_M07_F04",
    "N15_M07_F10",
}

assert set(
    failures["Bearing"].dropna().unique()
) <= {
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

assert (
    signal_failure_count[
        "Failure_Configurations"
    ].max()
    <= 8
)

print()
print("=" * 80)
print("STEP 95 SANITY CHECKS: PASS")
print("=" * 80)

print()
print("Total failure records:", len(failures))
print("Persistent failure records:", len(persistent))
print(
    "Unique failed signals:",
    failures["File"].nunique()
)

print()
print("Top 3 bearings:")

print(
    top_three[
        [
            "Bearing",
            "Failure_Records",
            "Percentage_of_All_Failures"
        ]
    ].to_string(index=False)
)

print()
print(
    "Top 3 contribution:",
    f"{top_three_percentage:.2f}% of all failure records"
)

print()
print("Saved files:")

for path in sorted(
    OUT_DIR.glob("*.csv")
):
    print(" -", path)

print()
print("=" * 80)
print("STEP 95 COMPLETE")
print("NO NEW ML EXPERIMENT")
print("EXISTING STEP 93 RESULTS ONLY")
print("=" * 80)