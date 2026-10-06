import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# STEP 140 - CLUSTERED BOOTSTRAP ROBUSTNESS ANALYSIS
# ============================================================

print("=" * 75)
print("STEP 140 - CLUSTERED BOOTSTRAP ROBUSTNESS ANALYSIS")
print("=" * 75)

INPUT_FILE = "secondary_four_configuration_predictions.csv"

N_BOOT = 10000
SEED = 42

rng = np.random.default_rng(SEED)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

required = [
    "Configuration",
    "Bearing",
    "Condition",
    "File",
    "True_Label",
    "Predicted_Label"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# CREATE FAILURE FLAG
# ============================================================

df["Failure"] = (
    df["True_Label"].astype(int)
    != df["Predicted_Label"].astype(int)
).astype(int)

df["True_Label"] = df["True_Label"].astype(int)
df["Predicted_Label"] = df["Predicted_Label"].astype(int)

# ============================================================
# VERIFY FOUR CONFIGURATIONS PER RECORDING
# ============================================================

counts = df.groupby("File")["Configuration"].nunique()

print("\nUnique recordings:", counts.shape[0])

print("\nConfiguration count per recording:")
print(counts.value_counts().sort_index())

if not np.all(counts.values == 4):
    raise ValueError(
        "Not every recording has exactly four configurations. "
        "Check secondary_four_configuration_predictions.csv."
    )

# ============================================================
# RECORDING-LEVEL INFORMATION
# ============================================================

recording_info = (
    df.groupby("File")
    .agg(
        Bearing=("Bearing", "first"),
        Condition=("Condition", "first"),
        True_Label=("True_Label", "first")
    )
    .reset_index()
)

print("\nRecording-level dataset:")
print(recording_info.shape)

print("\nClass distribution:")
print(recording_info["True_Label"].value_counts().sort_index())

# ============================================================
# PIVOT CONFIGURATION PREDICTIONS
# ============================================================

pred_pivot = df.pivot(
    index="File",
    columns="Configuration",
    values="Predicted_Label"
)

failure_pivot = df.pivot(
    index="File",
    columns="Configuration",
    values="Failure"
)

configurations = sorted(df["Configuration"].unique())

print("\nConfigurations:")
for c in configurations:
    print(" -", c)

# ============================================================
# COMBINE RECORDING-LEVEL DATA
# ============================================================

record = recording_info.set_index("File").copy()

for c in configurations:
    record[f"Pred_{c}"] = pred_pivot[c]
    record[f"Fail_{c}"] = failure_pivot[c]

record["Failure_Count"] = failure_pivot.sum(axis=1)

record["Persistent_Failure"] = (
    record["Failure_Count"] == len(configurations)
).astype(int)

record["Any_Failure"] = (
    record["Failure_Count"] > 0
).astype(int)

record["Failure_Proportion"] = (
    record["Failure_Count"] / len(configurations)
)

# ============================================================
# BASIC RECORDING-LEVEL SUMMARY
# ============================================================

print("\nRecording-level failure summary:")

print(
    "Overall any-configuration failure rate:",
    f"{record['Any_Failure'].mean():.4f}"
)

print(
    "Overall persistent failure rate:",
    f"{record['Persistent_Failure'].mean():.4f}"
)

print(
    "Mean configuration failure proportion:",
    f"{record['Failure_Proportion'].mean():.4f}"
)

# ============================================================
# BOOTSTRAP METRIC FUNCTION
# ============================================================

def calculate_metrics(sample):

    metrics = {}

    n = len(sample)

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    metrics["Overall_Any_Failure_Rate"] = sample["Any_Failure"].mean()

    metrics["Persistent_Failure_Rate"] = (
        sample["Persistent_Failure"].mean()
    )

    metrics["Mean_Configuration_Failure_Rate"] = (
        sample["Failure_Proportion"].mean()
    )

    # --------------------------------------------------------
    # Healthy false-positive rate
    # --------------------------------------------------------

    healthy = sample[sample["True_Label"] == 0]

    if len(healthy) > 0:

        metrics["Healthy_Any_Failure_Rate"] = (
            healthy["Any_Failure"].mean()
        )

        metrics["Healthy_Persistent_Failure_Rate"] = (
            healthy["Persistent_Failure"].mean()
        )

    else:

        metrics["Healthy_Any_Failure_Rate"] = np.nan
        metrics["Healthy_Persistent_Failure_Rate"] = np.nan

    # --------------------------------------------------------
    # Damaged miss rate
    # --------------------------------------------------------

    damaged = sample[sample["True_Label"] == 1]

    if len(damaged) > 0:

        metrics["Damaged_Any_Failure_Rate"] = (
            damaged["Any_Failure"].mean()
        )

        metrics["Damaged_Persistent_Failure_Rate"] = (
            damaged["Persistent_Failure"].mean()
        )

    else:

        metrics["Damaged_Any_Failure_Rate"] = np.nan
        metrics["Damaged_Persistent_Failure_Rate"] = np.nan

    # --------------------------------------------------------
    # Bearing-specific failure rates
    # --------------------------------------------------------

    for bearing in ["KI05", "KA03", "K002"]:

        subset = sample[sample["Bearing"] == bearing]

        if len(subset) > 0:

            metrics[f"{bearing}_Any_Failure_Rate"] = (
                subset["Any_Failure"].mean()
            )

            metrics[f"{bearing}_Persistent_Failure_Rate"] = (
                subset["Persistent_Failure"].mean()
            )

        else:

            metrics[f"{bearing}_Any_Failure_Rate"] = np.nan
            metrics[f"{bearing}_Persistent_Failure_Rate"] = np.nan

    # --------------------------------------------------------
    # Configuration failure rates
    # --------------------------------------------------------

    for c in configurations:

        metrics[f"{c}_Failure_Rate"] = (
            sample[f"Fail_{c}"].mean()
        )

    return metrics


# ============================================================
# ORIGINAL METRICS
# ============================================================

original_metrics = calculate_metrics(record)

print("\nOriginal recording-level metrics:")

for k, v in original_metrics.items():
    print(f"{k}: {v:.6f}")

# ============================================================
# CLUSTERED BOOTSTRAP
# ============================================================

files = record.index.to_numpy()

bootstrap_results = []

print("\nRunning clustered bootstrap...")
print("Bootstrap replicates:", N_BOOT)
print("Resampling unit: physical recording/file")
print("Random seed:", SEED)

for i in range(N_BOOT):

    sampled_files = rng.choice(
        files,
        size=len(files),
        replace=True
    )

    sample = record.loc[sampled_files]

    metrics = calculate_metrics(sample)

    bootstrap_results.append(metrics)

    if (i + 1) % 1000 == 0:
        print(f"Completed {i + 1}/{N_BOOT}")

bootstrap_df = pd.DataFrame(bootstrap_results)

# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def percentile_ci(values):

    values = np.asarray(values)
    values = values[~np.isnan(values)]

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    return lower, upper


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = []

for metric, original_value in original_metrics.items():

    lower, upper = percentile_ci(
        bootstrap_df[metric].values
    )

    summary_rows.append({
        "Metric": metric,
        "Observed": original_value,
        "CI_2.5%": lower,
        "CI_97.5%": upper,
        "CI_Width": upper - lower
    })

summary = pd.DataFrame(summary_rows)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("95% CLUSTERED BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 75)

for _, row in summary.iterrows():

    print(
        f"{row['Metric']:<55} "
        f"{row['Observed']:.4f} "
        f"[{row['CI_2.5%']:.4f}, "
        f"{row['CI_97.5%']:.4f}]"
    )

# ============================================================
# CONFIGURATION DIFFERENCES
# ============================================================

print("\n" + "=" * 75)
print("CONFIGURATION FAILURE-RATE DIFFERENCES")
print("=" * 75)

config_bootstrap = pd.DataFrame()

for c in configurations:

    config_bootstrap[c] = bootstrap_df[
        f"{c}_Failure_Rate"
    ]

difference_rows = []

for i in range(len(configurations)):

    for j in range(i + 1, len(configurations)):

        c1 = configurations[i]
        c2 = configurations[j]

        diff = (
            config_bootstrap[c1]
            - config_bootstrap[c2]
        )

        observed_diff = (
            original_metrics[f"{c1}_Failure_Rate"]
            - original_metrics[f"{c2}_Failure_Rate"]
        )

        lower, upper = percentile_ci(diff)

        difference_rows.append({
            "Configuration_A": c1,
            "Configuration_B": c2,
            "Observed_Difference_A_minus_B": observed_diff,
            "CI_2.5%": lower,
            "CI_97.5%": upper
        })

        print(
            f"\n{c1} - {c2}"
        )

        print(
            f"Observed difference: {observed_diff:.6f}"
        )

        print(
            f"95% CI: "
            f"[{lower:.6f}, {upper:.6f}]"
        )

difference_df = pd.DataFrame(difference_rows)

# ============================================================
# BEARING-SPECIFIC BOOTSTRAP TABLE
# ============================================================

print("\n" + "=" * 75)
print("BEARING-SPECIFIC FAILURE RATES")
print("=" * 75)

bearing_rows = []

for bearing in sorted(record["Bearing"].unique()):

    subset = record[
        record["Bearing"] == bearing
    ]

    observed = subset["Any_Failure"].mean()

    # bootstrap within the complete recording population
    # and calculate bearing metric for each replicate
    vals = []

    for sampled_files in []:
        pass

    # Use bootstrap records already generated by repeating
    # the same resampling procedure specifically for bearing.
    for _ in range(2000):

        sampled_files = rng.choice(
            files,
            size=len(files),
            replace=True
        )

        sample = record.loc[sampled_files]

        b = sample[
            sample["Bearing"] == bearing
        ]

        if len(b) > 0:
            vals.append(b["Any_Failure"].mean())

    if len(vals) > 0:

        lower, upper = percentile_ci(vals)

    else:

        lower, upper = np.nan, np.nan

    bearing_rows.append({
        "Bearing": bearing,
        "Observed_Any_Failure_Rate": observed,
        "CI_2.5%": lower,
        "CI_97.5%": upper
    })

    print(
        f"{bearing:<6} "
        f"Observed={observed:.4f} "
        f"95% CI=[{lower:.4f}, {upper:.4f}]"
    )

bearing_bootstrap_df = pd.DataFrame(bearing_rows)

# ============================================================
# SAVE RESULTS
# ============================================================

summary.to_csv(
    "clustered_bootstrap_summary.csv",
    index=False
)

difference_df.to_csv(
    "configuration_bootstrap_differences.csv",
    index=False
)

bearing_bootstrap_df.to_csv(
    "bearing_bootstrap_summary.csv",
    index=False
)

bootstrap_df.to_csv(
    "all_bootstrap_replicates.csv",
    index=False
)

record.reset_index().to_csv(
    "recording_level_dataset.csv",
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("STEP 140 COMPLETED")
print("=" * 75)

print("\nFiles saved:")

print("clustered_bootstrap_summary.csv")
print("configuration_bootstrap_differences.csv")
print("bearing_bootstrap_summary.csv")
print("all_bootstrap_replicates.csv")
print("recording_level_dataset.csv")

print("\nImportant:")
print(
    "Bootstrap resampling was performed at the physical recording "
    "level rather than the prediction-row level."
)

print(
    "The analysis estimates uncertainty; it does not establish causation."
)

print(
    "Configuration comparisons remain paired because all configurations "
    "belong to the same physical recording."
)