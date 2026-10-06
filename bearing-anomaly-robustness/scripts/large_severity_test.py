import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Load feature dataset
# ==========================================

df = pd.read_csv(
    "repeated_severity_features.csv"
)

print("Dataset loaded")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Features
# ==========================================

feature_columns = [
    "RMS",
    "STD",
    "Peak",
    "Peak_to_Peak",
    "Kurtosis",
    "Crest_Factor"
]


# ==========================================
# 3. Results storage
# ==========================================

all_results = []


# ==========================================
# 4. Test each SNR
# ==========================================

for snr in [0, 5, 10, 20]:

    print("\n===================================")
    print(f"SNR = {snr} dB")
    print("===================================")


    # Select current SNR
    snr_df = df[
        df["SNR"] == snr
    ].copy()


    # --------------------------------------
    # Unseen operating condition
    #
    # Train: A + B
    # Test : C
    # --------------------------------------

    train_df = snr_df[
        snr_df["Condition"].isin(["A", "B"])
    ]

    test_df = snr_df[
        snr_df["Condition"] == "C"
    ]


    X_train = train_df[feature_columns]
    y_train = train_df["Label"]

    X_test = test_df[feature_columns]
    y_test = test_df["Label"]


    print("Training samples:", len(train_df))
    print("Testing samples :", len(test_df))


    # ======================================
    # Train Random Forest
    # ======================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # ======================================
    # Predict
    # ======================================

    y_pred = model.predict(X_test)


    # ======================================
    # Overall metrics
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )


    print("\nOverall:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


    # ======================================
    # Severity-wise detection
    # ======================================

    print("\nSeverity-wise detection:")


    for severity in [
        "Mild",
        "Medium",
        "Strong"
    ]:

        mask = (
            test_df["Severity"] == severity
        )


        # Ground-truth anomalous samples
        true_values = y_test[mask]

        predicted_values = y_pred[mask]


        # Detection rate
        detection_rate = (
            predicted_values == 1
        ).mean()


        # Number detected
        detected = np_count = (
            predicted_values == 1
        ).sum()


        total = len(
            predicted_values
        )


        print(
            f"{severity:>6} | "
            f"Detected = {detected}/{total} | "
            f"Detection Rate = "
            f"{detection_rate:.4f}"
        )


        all_results.append({
            "SNR": snr,
            "Severity": severity,
            "Detected": detected,
            "Total": total,
            "Detection_Rate": detection_rate
        })


# ==========================================
# 5. Results DataFrame
# ==========================================

results_df = pd.DataFrame(
    all_results
)


# ==========================================
# 6. Display
# ==========================================

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 7. Save
# ==========================================

results_df.to_csv(
    "large_severity_results.csv",
    index=False
)

print("\nResults saved as:")
print("large_severity_results.csv")