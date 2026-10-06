# Robust Anomaly Detection in Non-Stationary Vibration Signals

This repository contains the code, experimental results, figures, and analysis developed as part of my independent research interest in machine learning, signal processing, and condition monitoring.

## About

This work investigates the robustness of anomaly detection methods when vibration signals are affected by:

- Non-stationary behavior
- Variable noise conditions
- Different operating conditions
- Bearing-to-bearing distribution shifts

The main focus is on understanding how anomaly detection performance changes when models are evaluated on previously unseen bearings.

## Dataset

The real-data experiments use the Paderborn University Bearing Dataset.

The analysis uses the vibration channel standardized to a nominal sampling frequency of 64 kHz.

A Leave-One-Bearing-Out (LOBO) evaluation strategy is used as the primary real-data evaluation protocol. This ensures that recordings from the test bearing are not included in model training.

## Signal Representations

The study evaluates four representations:

1. Time-domain statistical features
2. Absolute STFT features
3. Order-normalized STFT features
4. True-RMS-normalized STFT features

## Machine Learning Methods

The evaluated classifiers are:

- Random Forest (RF)
- Radial Basis Function Support Vector Machine (RBF-SVM)

## Evaluation

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Healthy false-positive rate
- Specificity

Recording-level analysis is also performed to investigate bearing-specific effects.

A recording-level clustered bootstrap with 10,000 replicates is used for uncertainty estimation.

## Main Observation

The results demonstrate that anomaly-detection performance can decrease substantially when a model is evaluated on bearings that were not represented during training.

Across the evaluated representation-classifier configurations, pooled F1 scores range from approximately 0.375 to 0.616.

These results highlight the importance of evaluating anomaly-detection methods under realistic bearing-to-bearing distribution shifts rather than relying only on random train-test splits.

## Repository Contents

This repository contains:

- Research code
- Feature extraction scripts
- Derived datasets
- Experimental results
- Validation results
- Figures
- Tables
- Statistical analysis outputs
- Model evaluation results

Raw and large local datasets are excluded through `.gitignore`.

## Reproducibility

The repository is organized to support reproduction of the experimental analysis.

The general workflow consists of:

1. Preparing the vibration data
2. Extracting signal representations
3. Generating feature datasets
4. Performing LOBO evaluation
5. Training the machine-learning models
6. Computing evaluation metrics
7. Performing recording-level analysis
8. Generating figures and tables

## Research Status

This is an **independent research study** developed as part of my personal interest in:

- Signal Processing
- Machine Learning
- Anomaly Detection
- Vibration Analysis
- Condition Monitoring
- Distribution Shift and Generalization

It is not presented as an official university project.

## Author

**Dhodda Sai Eswar**

B.Tech Student  
Electronics and Communication Engineering
