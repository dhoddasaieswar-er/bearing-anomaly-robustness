# Reproducibility guide

## 1. Real-data pipeline

The corrected real-data analysis standardizes the vibration input to the nominal 64 kHz sampling rate and uses `Y[6]` / `vibration_1`. The earlier stored time-vector analysis is not part of the evaluated pipeline.

Primary evaluation is strict 10-bearing leave-one-bearing-out (LOBO): no recording from the held-out bearing is included in training.

## 2. Feature/representation generation

The curated feature-generation scripts are:

- `all_real_time_features.py`
- `all_real_stft_features.py`
- `relative_stft_features.py`
- `normalized_stft_features.py`
- `corrected_stft_64khz.py`

The later classifier/analysis scripts consume generated CSV artifacts. Their exact filenames are documented in the source code and should be preserved when reproducing the pipeline.

## 3. Classical evaluation

`classifier_control_rf_vs_svm.py` performs the classifier comparison over the prepared representations. The downstream failure-analysis scripts are:

- `failure_case_analysis.py`
- `failure_case_characterization.py`
- `failure_concentration_analysis.py`
- `final_results_consistency_audit.py`

## 4. Robustness and secondary analyses

The curated robustness scripts cover paired validation, clustered bootstrap, envelope ablation, one-class SVM sensitivity, and the KI05 physical-signature analysis.

## 5. Deep-learning baselines

Use only the corrected public implementations:

- `1d_cnn_lobo_stratified_validation.py`
- `cnn_gru_lobo_stratified_validation.py`

Older CNN/CNN-GRU scripts from the historical archive are intentionally excluded because they represent superseded validation/results.

## 6. Intermediate artifacts

Some analyses consume intermediate CSV files produced by earlier stages. A clean reproduction should generate those artifacts in the same logical sequence rather than replacing them with unrelated historical files.

## 7. Privacy/portability

The public scripts have been scanned for the author's previous local filesystem root and username. Machine-specific absolute paths were removed from the public copy.
