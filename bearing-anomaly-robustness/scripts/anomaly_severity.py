import numpy as np


# ==========================================
# 1. Basic parameters
# ==========================================

fs = 1000
duration = 10

t = np.arange(0, duration, 1 / fs)

snr_levels = [0, 5, 10, 20]

conditions = {
    "A": (10, 50),
    "B": (15, 45),
    "C": (20, 60)
}

# Anomaly severity levels
severity_levels = {
    "Mild": 0.20,
    "Medium": 0.40,
    "Strong": 0.60
}


# ==========================================
# 2. Storage
# ==========================================

signals = []
labels = []
snrs = []
condition_labels = []
severity_labels = []


# ==========================================
# 3. Reproducible random generator
# ==========================================

rng = np.random.default_rng(42)


# ==========================================
# 4. Generate dataset
# ==========================================

for condition, (f_start, f_end) in conditions.items():

    print(f"Generating condition {condition}...")

    # --------------------------------------
    # Normal signal components
    # --------------------------------------

    f1 = np.linspace(
        f_start,
        f_end,
        len(t)
    )

    phase1 = 2 * np.pi * np.cumsum(f1) / fs

    component1 = np.sin(phase1)


    # Second component
    f2 = np.linspace(
        f_start / 2,
        f_end / 2,
        len(t)
    )

    phase2 = 2 * np.pi * np.cumsum(f2) / fs

    component2 = 0.5 * np.sin(phase2)


    normal_signal = component1 + component2


    # --------------------------------------
    # Generate normal signals
    # --------------------------------------

    for snr in snr_levels:

        for _ in range(20):

            signal_power = np.mean(
                normal_signal ** 2
            )

            noise_power = (
                signal_power /
                (10 ** (snr / 10))
            )

            noise = rng.normal(
                0,
                np.sqrt(noise_power),
                len(t)
            )

            noisy_signal = normal_signal + noise

            signals.append(noisy_signal)

            labels.append(0)

            snrs.append(snr)

            condition_labels.append(condition)

            severity_labels.append("Normal")


    # --------------------------------------
    # Generate anomalous signals
    # --------------------------------------

    for severity, amplitude in severity_levels.items():

        for snr in snr_levels:

            for _ in range(20):

                # Start with normal signal
                anomalous_signal = normal_signal.copy()


                # Anomaly from 4 to 6 seconds
                anomaly_mask = (
                    (t >= 4) &
                    (t <= 6)
                )


                # 70 Hz anomaly
                anomaly = (
                    amplitude *
                    np.sin(2 * np.pi * 70 * t)
                )


                anomalous_signal[
                    anomaly_mask
                ] += anomaly[
                    anomaly_mask
                ]


                # Calculate signal power
                signal_power = np.mean(
                    anomalous_signal ** 2
                )

                noise_power = (
                    signal_power /
                    (10 ** (snr / 10))
                )

                noise = rng.normal(
                    0,
                    np.sqrt(noise_power),
                    len(t)
                )


                noisy_signal = (
                    anomalous_signal + noise
                )


                signals.append(noisy_signal)

                labels.append(1)

                snrs.append(snr)

                condition_labels.append(condition)

                severity_labels.append(severity)


# ==========================================
# 5. Convert to NumPy arrays
# ==========================================

signals = np.array(signals)

labels = np.array(labels)

snrs = np.array(snrs)

condition_labels = np.array(condition_labels)

severity_labels = np.array(severity_labels)


# ==========================================
# 6. Print dataset information
# ==========================================

print("\n===================================")
print("STEP 13 DATASET")
print("===================================")

print("Signals shape:")
print(signals.shape)

print("\nLabels:")
print(labels.shape)

print("\nNumber of normal signals:")
print(np.sum(labels == 0))

print("\nNumber of anomalous signals:")
print(np.sum(labels == 1))

print("\nSeverity distribution:")

unique_severity, counts = np.unique(
    severity_labels,
    return_counts=True
)

for severity, count in zip(
    unique_severity,
    counts
):
    print(f"{severity}: {count}")


print("\nSNR distribution:")

unique_snr, counts = np.unique(
    snrs,
    return_counts=True
)

for snr, count in zip(
    unique_snr,
    counts
):
    print(f"{snr} dB: {count}")


print("\nCondition distribution:")

unique_condition, counts = np.unique(
    condition_labels,
    return_counts=True
)

for condition, count in zip(
    unique_condition,
    counts
):
    print(f"{condition}: {count}")


# ==========================================
# 7. Save dataset
# ==========================================

np.savez(
    "step13_severity_dataset.npz",
    signals=signals,
    labels=labels,
    SNR=snrs,
    Condition=condition_labels,
    Severity=severity_labels
)

print("\nDataset saved as:")
print("step13_severity_dataset.npz")