from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STEP 94 — FAILURE CASE CHARACTERIZATION
#
# Purpose:
# Characterize persistent failure cases using already-existing
# feature representations and Step 93 predictions.
#
# NO NEW ML TRAINING
# NO NEW DATASET
# NO NEW CLASSIFIER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = BASE_DIR / "failure_case_characterization_step94"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

FAILURE_FILE = (
    BASE_DIR
    / "failure_case_analysis_step93"
    / "step93_all_failure_cases.csv"
)

PERSISTENT_FILE = (
    BASE_DIR
    / "failure_case_analysis_step93"
    / "step93_persistent_failure_cases.csv"
)

TIME_FILE = BASE_DIR / "all_real_time_features.csv"
STFT_FILE = BASE_DIR / "all_real_stft_features.csv"
ORDER_FILE = BASE_DIR / "relative_stft_features.csv"
RMS_FILE = BASE_DIR / "normalized_stft_features.csv"


for path in [
    FAILURE_FILE,
    PERSISTENT_FILE,
    TIME_FILE,
    STFT_FILE,
    ORDER_FILE,
    RMS_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file:\n{path}")


# ============================================================
# LOAD DATA
# ============================================================

failures = pd.read_csv(FAILURE_FILE)
persistent = pd.read_csv(PERSISTENT_FILE)

time_df = pd.read_csv(TIME_FILE)
stft_df = pd.read_csv(STFT_FILE)
order_df = pd.read_csv(ORDER_FILE)
rms_df = pd.read_csv(RMS_FILE)


print("=" * 80)
print("STEP 94 — FAILURE CASE CHARACTERIZATION")
print("=" * 80)

print()
print("Failure records:", len(failures))
print("Persistent failure records:", len(persistent))


# ============================================================
# IDENTIFY PERSISTENT FAILURE FILES
# ============================================================

persistent_files = persistent["File"].unique()

print()
print("Persistent failure signals:", len(persistent_files))


# ============================================================
# MERGE EXISTING FEATURES
#
# File is the common signal identifier.
# ============================================================

META = [
    "Bearing",
    "Condition",
    "Label",
    "File",
]


def prepare_features(df, prefix):

    numeric_cols = []

    for col in df.columns:

        if col in META:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    result = df[META + numeric_cols].copy()

    rename_map = {
        col: f"{prefix}_{col}"
        for col in numeric_cols
    }

    result = result.rename(columns=rename_map)

    return result


time_features = prepare_features(
    time_df,
    "TIME"
)

stft_features = prepare_features(
    stft_df,
    "STFT"
)

order_features = prepare_features(
    order_df,
    "ORDER"
)

rms_features = prepare_features(
    rms_df,
    "RMS"
)


# ============================================================
# COMBINE FEATURES
# ============================================================

combined = time_features.copy()

combined = combined.merge(
    stft_features,
    on=META,
    how="inner",
    suffixes=("", "_duplicate")
)

combined = combined.merge(
    order_features,
    on=META,
    how="inner",
    suffixes=("", "_duplicate")
)

combined = combined.merge(
    rms_features,
    on=META,
    how="inner",
    suffixes=("", "_duplicate")
)


print()
print("Combined feature table:")
print("Rows:", len(combined))
print("Columns:", len(combined.columns))


# ============================================================
# ADD FAILURE INFORMATION
# ============================================================

failure_counts = (
    failures
    .groupby("File")
    .agg(
        Failure_Configurations=("Failure_Type", "count"),
        Failure_Types=(
            "Failure_Type",
            lambda x: ",".join(sorted(set(x)))
        ),
        Representations=("Representation", "nunique"),
        Classifiers=("Classifier", "nunique"),
    )
    .reset_index()
)

analysis = combined.merge(
    failure_counts,
    on="File",
    how="left"
)

analysis["Failure_Configurations"] = (
    analysis["Failure_Configurations"]
    .fillna(0)
)

analysis["Representations"] = (
    analysis["Representations"]
    .fillna(0)
)

analysis["Classifiers"] = (
    analysis["Classifiers"]
    .fillna(0)
)

analysis["Persistent_Failure"] = (
    analysis["Failure_Configurations"] >= 4
)


# ============================================================
# SAVE FULL CHARACTERIZATION TABLE
# ============================================================

full_file = (
    OUT_DIR
    / "step94_full_failure_characterization.csv"
)

analysis.to_csv(
    full_file,
    index=False
)


# ============================================================
# PERSISTENT FAILURE CHARACTERIZATION
# ============================================================

persistent_analysis = analysis[
    analysis["Persistent_Failure"]
].copy()

persistent_analysis_file = (
    OUT_DIR
    / "step94_persistent_failure_characterization.csv"
)

persistent_analysis.to_csv(
    persistent_analysis_file,
    index=False
)


# ============================================================
# BEARING-LEVEL FAILURE CHARACTERIZATION
# ============================================================

bearing_summary = (
    analysis
    .groupby(
        ["Bearing", "Label"]
    )
    .agg(
        Signals=("File", "count"),
        Persistent_Failures=(
            "Persistent_Failure",
            "sum"
        ),
        Mean_Failure_Configurations=(
            "Failure_Configurations",
            "mean"
        ),
        Max_Failure_Configurations=(
            "Failure_Configurations",
            "max"
        ),
    )
    .reset_index()
)

bearing_summary_file = (
    OUT_DIR
    / "step94_bearing_failure_characterization.csv"
)

bearing_summary.to_csv(
    bearing_summary_file,
    index=False
)


# ============================================================
# OPERATING-CONDITION FAILURE SUMMARY
# ============================================================

condition_summary = (
    analysis
    .groupby("Condition")
    .agg(
        Signals=("File", "count"),
        Persistent_Failures=(
            "Persistent_Failure",
            "sum"
        ),
        Mean_Failure_Configurations=(
            "Failure_Configurations",
            "mean"
        ),
    )
    .reset_index()
)

condition_file = (
    OUT_DIR
    / "step94_condition_failure_characterization.csv"
)

condition_summary.to_csv(
    condition_file,
    index=False
)


# ============================================================
# FEATURE STATISTICS:
# PERSISTENT FAILURES VS NON-PERSISTENT SIGNALS
# ============================================================

numeric_columns = analysis.select_dtypes(
    include=np.number
).columns.tolist()

# Remove metadata-like numeric variables
exclude = {
    "Label",
    "Failure_Configurations",
    "Representations",
    "Classifiers",
}

numeric_columns = [
    col
    for col in numeric_columns
    if col not in exclude
]


feature_rows = []

for feature in numeric_columns:

    persistent_values = analysis.loc[
        analysis["Persistent_Failure"],
        feature
    ].dropna()

    nonpersistent_values = analysis.loc[
        ~analysis["Persistent_Failure"],
        feature
    ].dropna()

    if len(persistent_values) == 0:
        continue

    if len(nonpersistent_values) == 0:
        continue

    p_mean = persistent_values.mean()
    np_mean = nonpersistent_values.mean()

    p_median = persistent_values.median()
    np_median = nonpersistent_values.median()

    # Ratio of persistent mean to non-persistent mean.
    # Only meaningful when denominator is non-zero.
    if abs(np_mean) > 1e-12:
        mean_ratio = p_mean / np_mean
    else:
        mean_ratio = np.nan

    feature_rows.append({
        "Feature": feature,
        "Persistent_Mean": p_mean,
        "Nonpersistent_Mean": np_mean,
        "Persistent_Median": p_median,
        "Nonpersistent_Median": np_median,
        "Persistent_Mean_to_Nonpersistent_Mean": mean_ratio,
        "Persistent_Count": len(persistent_values),
        "Nonpersistent_Count": len(nonpersistent_values),
    })


feature_comparison = pd.DataFrame(feature_rows)

feature_comparison_file = (
    OUT_DIR
    / "step94_persistent_vs_nonpersistent_feature_statistics.csv"
)

feature_comparison.to_csv(
    feature_comparison_file,
    index=False
)


# ============================================================
# LARGE DIFFERENCES — DESCRIPTIVE ONLY
#
# Sort by absolute log-like ratio where possible.
# This is NOT a statistical significance test.
# ============================================================

fc = feature_comparison.copy()

fc["Absolute_Mean_Difference"] = (
    abs(
        fc["Persistent_Mean"]
        -
        fc["Nonpersistent_Mean"]
    )
)

fc = fc.sort_values(
    "Absolute_Mean_Difference",
    ascending=False
)

large_difference_file = (
    OUT_DIR
    / "step94_largest_descriptive_feature_differences.csv"
)

fc.to_csv(
    large_difference_file,
    index=False
)


# ============================================================
# CONDITION × PERSISTENT FAILURE
# ============================================================

persistent_condition = (
    persistent_analysis
    .groupby("Condition")
    .size()
    .reset_index(
        name="Persistent_Failure_Signals"
    )
)

persistent_condition_file = (
    OUT_DIR
    / "step94_persistent_failures_by_condition.csv"
)

persistent_condition.to_csv(
    persistent_condition_file,
    index=False
)


# ============================================================
# PRINT IMPORTANT RESULTS
# ============================================================

print()
print("=" * 80)
print("PERSISTENT FAILURE SIGNALS")
print("=" * 80)

if len(persistent_analysis) == 0:

    print("No persistent failures found.")

else:

    print(
        persistent_analysis[
            [
                "File",
                "Bearing",
                "Condition",
                "Label",
                "Failure_Configurations",
                "Failure_Types",
            ]
        ]
        .sort_values(
            "Failure_Configurations",
            ascending=False
        )
        .head(50)
        .to_string(index=False)
    )


print()
print("=" * 80)
print("BEARING-LEVEL SUMMARY")
print("=" * 80)

print(
    bearing_summary.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("OPERATING-CONDITION SUMMARY")
print("=" * 80)

print(
    condition_summary.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TOP DESCRIPTIVE FEATURE DIFFERENCES")
print("=" * 80)

print(
    fc[
        [
            "Feature",
            "Persistent_Mean",
            "Nonpersistent_Mean",
            "Persistent_Median",
            "Nonpersistent_Median",
            "Absolute_Mean_Difference",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(combined) == 800

assert analysis["File"].nunique() == 800

assert persistent_analysis["File"].nunique() == len(
    persistent_files
)

assert set(
    analysis["Bearing"].unique()
) == {
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


print()
print("=" * 80)
print("STEP 94 SANITY CHECKS: PASS")
print("=" * 80)

print()
print("Combined signals:", len(combined))
print("Persistent signals:", len(persistent_analysis))

print()
print("Saved files:")

for path in sorted(OUT_DIR.glob("*.csv")):
    print(" -", path)

print()
print("=" * 80)
print("STEP 94 COMPLETE")
print("DESCRIPTIVE CHARACTERIZATION ONLY")
print("NO NEW ML EXPERIMENT WAS PERFORMED")
print("=" * 80)