import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("STEP 66 — PER-SIGNAL NORMALIZED STFT FEATURES")
print("=" * 80)

# ------------------------------------------------------------
# 1. Load absolute STFT features
# ------------------------------------------------------------

df = pd.read_csv("all_real_stft_features.csv")

metadata_cols = [
    "Bearing",
    "Condition",
    "Label",
    "File"
]

feature_cols = [
    col for col in df.columns
    if col not in metadata_cols
    and pd.api.types.is_numeric_dtype(df[col])
]

print("\nOriginal feature columns:")
print(feature_cols)

# ------------------------------------------------------------
# 2. Normalize spectral features
# ------------------------------------------------------------

# STFT_Mean is the reference energy level.
reference = df["STFT_Mean"].replace(0, np.nan)

normalized_features = pd.DataFrame(index=df.index)

for feature in feature_cols:

    # Do not normalize STFT_Mean itself.
    if feature == "STFT_Mean":
        continue

    normalized_features[feature + "_Norm"] = (
        df[feature] / reference
    )

# Keep metadata
result = pd.concat(
    [
        normalized_features,
        df[metadata_cols]
    ],
    axis=1
)

# Remove any possible invalid values
result = result.replace(
    [np.inf, -np.inf],
    np.nan
)

result = result.dropna()

# ------------------------------------------------------------
# 3. Display result
# ------------------------------------------------------------

print("\nNormalized dataset shape:", result.shape)

print("\nNormalized feature columns:")
print(normalized_features.columns.tolist())

print("\nFirst five rows:")
print(result.head().to_string())

# ------------------------------------------------------------
# 4. Save
# ------------------------------------------------------------

result.to_csv(
    "normalized_stft_features.csv",
    index=False
)

print("\nSaved:")
print("normalized_stft_features.csv")

print("\n" + "=" * 80)
print("STEP 66 COMPLETE")
print("=" * 80)