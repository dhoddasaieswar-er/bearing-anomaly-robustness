import numpy as np
import pandas as pd
# ============================================================
# EXPERIMENT 2
# ENVELOPE-FEATURE ABLATION
# STRICT 10-BEARING LOBO
# ============================================================

# ------------------------------------------------------------
# FROZEN BASELINE
# Order-normalized STFT + RBF-SVM
# ------------------------------------------------------------

baseline = {
    "K001": 0.0000,
    "K002": 0.0000,
    "K003": 0.0000,
    "K004": 0.0000,
    "K005": 0.0000,
    "K006": 0.0000,
    "KA01": 0.8571,
    "KA03": 0.5321,
    "KA04": 1.0000,
    "KI05": 0.0000
}

# ------------------------------------------------------------
# ACTUAL STEP-152 FILE
# ------------------------------------------------------------

ENVELOPE_FILE = "bearing_metrics.csv"

df = pd.read_csv(ENVELOPE_FILE)

print("=" * 80)
print("STEP 152 - ENVELOPE ABLATION")
print("=" * 80)

print("\nLoaded file:")
print(ENVELOPE_FILE)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# IDENTIFY COLUMNS
# ============================================================

possible_bearing_cols = [
    "Bearing",
    "bearing",
    "Bearing_ID",
    "bearing_id",
    "BearingIdentity"
]

bearing_col = None

for col in possible_bearing_cols:
    if col in df.columns:
        bearing_col = col
        break

if bearing_col is None:
    raise ValueError("Bearing column not found.")

possible_f1_cols = [
    "F1",
    "f1",
    "F1_Score",
    "F1_score",
    "F1Score"
]

f1_col = None

for col in possible_f1_cols:
    if col in df.columns:
        f1_col = col
        break

if f1_col is None:
    raise ValueError("F1 column not found.")

# ============================================================
# EXTRACT ENVELOPE RESULTS
# ============================================================

envelope = {}

for _, row in df.iterrows():

    bearing = str(row[bearing_col]).strip()

    if bearing in baseline:
        envelope[bearing] = float(row[f1_col])

# ============================================================
# VERIFY ALL 10 BEARINGS
# ============================================================

bearings = list(baseline.keys())

missing = [b for b in bearings if b not in envelope]

if missing:
    raise ValueError(
        f"Missing LOBO bearings in envelope results: {missing}"
    )

envelope_scores = np.array(
    [envelope[b] for b in bearings],
    dtype=float
)

baseline_scores = np.array(
    [baseline[b] for b in bearings],
    dtype=float
)

# ============================================================
# DISPERSION
# ============================================================

def dispersion(scores):

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)

    return {
        "Mean": np.mean(scores),
        "SD": np.std(scores, ddof=1),
        "Median": np.median(scores),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "Min": np.min(scores),
        "Max": np.max(scores)
    }

base_stats = dispersion(baseline_scores)
env_stats = dispersion(envelope_scores)

# ============================================================
# PRINT DISPERSION
# ============================================================

print("\n" + "=" * 80)
print("LOBO DISPERSION")
print("=" * 80)

print("\nOrder-STFT + RBF-SVM")

for key, value in base_stats.items():
    print(f"{key:<10}: {value:.4f}")

print("\nEnvelope + RBF-SVM")

for key, value in env_stats.items():
    print(f"{key:<10}: {value:.4f}")

# ============================================================
# BEARING-LEVEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("BEARING-LEVEL COMPARISON")
print("=" * 80)

print(
    f"{'Bearing':<10}"
    f"{'Baseline':<15}"
    f"{'Envelope':<15}"
    f"{'Difference':<15}"
)

print("-" * 55)

for i, bearing in enumerate(bearings):

    diff = envelope_scores[i] - baseline_scores[i]

    print(
        f"{bearing:<10}"
        f"{baseline_scores[i]:<15.4f}"
        f"{envelope_scores[i]:<15.4f}"
        f"{diff:+.4f}"
    )

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL EXPERIMENT 2 SUMMARY")
print("=" * 80)

print(
    f"\n{'Method':<30}"
    f"{'Mean F1':<12}"
    f"{'SD':<12}"
)

print("-" * 54)

print(
    f"{'Order-STFT + RBF-SVM':<30}"
    f"{base_stats['Mean']:<12.4f}"
    f"{base_stats['SD']:<12.4f}"
)

print(
    f"{'Envelope + RBF-SVM':<30}"
    f"{env_stats['Mean']:<12.4f}"
    f"{env_stats['SD']:<12.4f}"
)

print(
    f"\nMean F1 difference: "
    f"{mean_difference:+.4f}"
)

print("\n" + "=" * 80)
print("EXPERIMENT 2 COMPLETE")
print("=" * 80)