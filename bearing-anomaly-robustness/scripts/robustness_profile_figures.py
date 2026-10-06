import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ============================================================
# STEP 89 — ROBUSTNESS PROFILE FIGURES
# Uses validated Step 88 results only.
# NO EXPERIMENTS ARE RERUN.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = (
    BASE_DIR
    / "robustness_profile_step88"
    / "step88_final_multidimensional_robustness_profile.csv"
)

OUTPUT_DIR = BASE_DIR / "robustness_profile_step89"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("STEP 89 — ROBUSTNESS PROFILE FIGURES")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE.resolve())

# ============================================================
# 1. POOLED F1
# ============================================================

pooled = df[df["Dimension"] == "Overall detection"].copy()

labels = (
    pooled["Representation"]
    + "\n"
    + pooled["Classifier"]
)

values = pooled["Value"]

plt.figure(figsize=(11, 6))
plt.bar(labels, values)
plt.ylabel("Pooled F1-score")
plt.xlabel("Representation / Classifier")
plt.title("Pooled Detection Performance")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure1_pooled_f1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 2. DAMAGED-BEARING RECALL
# Mean ± SD
# ============================================================

damaged = df[
    (df["Dimension"] == "Damaged-bearing robustness")
    & (df["Metric"] == "Recall mean")
].copy()

damaged_sd = df[
    (df["Dimension"] == "Damaged-bearing robustness")
    & (df["Metric"] == "Recall SD")
].copy()

labels = (
    damaged["Representation"]
    + "\n"
    + damaged["Classifier"]
)

means = damaged["Value"].to_numpy()
sds = damaged_sd["Value"].to_numpy()

x = np.arange(len(labels))

plt.figure(figsize=(11, 6))
plt.errorbar(
    x,
    means,
    yerr=sds,
    fmt="o",
    capsize=5
)

plt.xticks(x, labels, rotation=30, ha="right")
plt.ylabel("Damaged-bearing Recall")
plt.xlabel("Representation / Classifier")
plt.title("Bearing-Level Damaged-Bearing Recall")
plt.ylim(0, 1.05)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure2_damaged_recall_mean_sd.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 3. HEALTHY-BEARING FPR
# Mean ± SD
# ============================================================

healthy = df[
    (df["Dimension"] == "Healthy-bearing robustness")
    & (df["Metric"] == "FPR mean")
].copy()

healthy_sd = df[
    (df["Dimension"] == "Healthy-bearing robustness")
    & (df["Metric"] == "FPR SD")
].copy()

labels = (
    healthy["Representation"]
    + "\n"
    + healthy["Classifier"]
)

means = healthy["Value"].to_numpy()
sds = healthy_sd["Value"].to_numpy()

x = np.arange(len(labels))

plt.figure(figsize=(11, 6))
plt.errorbar(
    x,
    means,
    yerr=sds,
    fmt="o",
    capsize=5
)

plt.xticks(x, labels, rotation=30, ha="right")
plt.ylabel("Healthy-bearing False Positive Rate")
plt.xlabel("Representation / Classifier")
plt.title("Bearing-Level Healthy-Bearing False Positive Rate")
plt.ylim(0, 1.05)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure3_healthy_fpr_mean_sd.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 4. SYNTHETIC NOISE ROBUSTNESS
# F1 vs SNR
# ============================================================

noise = df[
    df["Dimension"] == "Synthetic noise robustness"
].copy()

time_noise = noise[
    noise["Representation"] == "Time-domain"
].copy()

relative_noise = noise[
    noise["Representation"] == "Relative-frequency STFT"
].copy()

def extract_snr(metric):
    return int(metric.split("at")[1].replace("dB", "").strip())

time_noise["SNR"] = time_noise["Metric"].apply(extract_snr)
relative_noise["SNR"] = relative_noise["Metric"].apply(extract_snr)

time_noise = time_noise.sort_values("SNR")
relative_noise = relative_noise.sort_values("SNR")

plt.figure(figsize=(9, 6))

plt.plot(
    time_noise["SNR"],
    time_noise["Value"],
    marker="o",
    label="Time-domain"
)

plt.plot(
    relative_noise["SNR"],
    relative_noise["Value"],
    marker="o",
    label="Relative-frequency STFT"
)

plt.xlabel("SNR (dB)")
plt.ylabel("F1-score")
plt.title("Synthetic Noise Robustness")
plt.ylim(0, 1.05)
plt.xticks([0, 5, 10, 20])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure4_synthetic_noise_robustness.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 5. ANOMALY-SEVERITY ROBUSTNESS
# F1 vs SNR for Mild / Medium / Strong
# ============================================================

severity = df[
    df["Dimension"] == "Synthetic anomaly-severity robustness"
].copy()

def extract_severity_and_snr(metric):
    parts = metric.split("F1 at")
    severity_name = parts[0].strip()
    snr_value = int(
        parts[1]
        .replace("dB", "")
        .strip()
    )
    return severity_name, snr_value

severity[
    ["Severity_Name", "SNR"]
] = severity["Metric"].apply(
    lambda x: pd.Series(
        extract_severity_and_snr(x)
    )
)

plt.figure(figsize=(9, 6))

for severity_name in ["Mild", "Medium", "Strong"]:

    subset = severity[
        severity["Severity_Name"] == severity_name
    ].sort_values("SNR")

    plt.plot(
        subset["SNR"],
        subset["Value"],
        marker="o",
        label=severity_name
    )

plt.xlabel("SNR (dB)")
plt.ylabel("F1-score")
plt.title("Synthetic Anomaly-Severity Robustness")
plt.ylim(0, 1.05)
plt.xticks([0, 5, 10, 20])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure5_anomaly_severity_robustness.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 6. OPERATING-CONDITION ROBUSTNESS
# ============================================================

condition = df[
    df["Dimension"]
    == "Synthetic operating-condition robustness"
].copy()

conditions = []

for metric in condition["Metric"]:
    conditions.append(
        metric.split("condition")[1].strip()
    )

condition["Held_out_Condition"] = conditions

plt.figure(figsize=(8, 6))

plt.bar(
    condition["Held_out_Condition"],
    condition["Value"]
)

plt.xlabel("Held-out Operating Condition")
plt.ylabel("F1-score")
plt.title("Synthetic Operating-Condition Robustness")
plt.ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure6_operating_condition_robustness.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# SUMMARY
# ============================================================

print()
print("Generated figures:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", file.name)

print()
print("=" * 80)
print("STEP 89 COMPLETE")
print("=" * 80)

print()
print("Output folder:")
print(OUTPUT_DIR.resolve())

print()
print("No experiments were rerun.")
print("Figures were generated only from Step 88 validated results.")

print()
print("=" * 80)