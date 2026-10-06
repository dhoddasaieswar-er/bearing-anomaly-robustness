import numpy as np
import pandas as pd
from scipy.stats import kurtosis


# ==========================================
# 1. Load Step 16 dataset
# ==========================================

data = np.load("step16_repeated_severity_dataset.npz")

signals = data["signals"]
labels = data["labels"]
snrs = data["SNR"]
conditions = data["Condition"]
severities = data["Severity"]

print("Dataset loaded")
print("Signals shape:", signals.shape)


# ==========================================
# 2. Feature extraction
# ==========================================

def extract_features(signal):

    rms = np.sqrt(np.mean(signal ** 2))

    std = np.std(signal)

    peak = np.max(np.abs(signal))

    peak_to_peak = np.ptp(signal)

    kurt = kurtosis(signal)

    crest_factor = peak / rms

    return [
        rms,
        std,
        peak,
        peak_to_peak,
        kurt,
        crest_factor
    ]


# ==========================================
# 3. Extract features
# ==========================================

features = []

for i, signal in enumerate(signals):

    if (i + 1) % 500 == 0:
        print(
            f"Processed {i + 1} / {len(signals)} signals"
        )

    features.append(
        extract_features(signal)
    )


# ==========================================
# 4. Create DataFrame
# ==========================================

feature_columns = [
    "RMS",
    "STD",
    "Peak",
    "Peak_to_Peak",
    "Kurtosis",
    "Crest_Factor"
]

df = pd.DataFrame(
    features,
    columns=feature_columns
)


# ==========================================
# 5. Add labels and metadata
# ==========================================

df["Label"] = labels
df["SNR"] = snrs
df["Condition"] = conditions
df["Severity"] = severities


# ==========================================
# 6. Check dataset
# ==========================================

print("\n===================================")
print("STEP 17 FEATURE DATASET")
print("===================================")

print("Shape:")
print(df.shape)

print("\nSeverity distribution:")
print(df["Severity"].value_counts())

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nCondition distribution:")
print(df["Condition"].value_counts())

print("\nSNR distribution:")
print(df["SNR"].value_counts())


# ==========================================
# 7. Save
# ==========================================

df.to_csv(
    "repeated_severity_features.csv",
    index=False
)

print("\nFeature dataset saved as:")
print("repeated_severity_features.csv")
