# Robustness Evaluation of Anomaly Detection in Non-Stationary Vibration Signals

Public research-code release accompanying the manuscript on bearing anomaly detection under variable noise and operating-condition variations.

## Scope

This repository contains a curated, sanitized subset of the experimental Python code used in the study. The original development archive contained substantially more exploratory, diagnostic, duplicated, and superseded scripts; those are intentionally not part of this public release.

The public pipeline is centered on:

- the Paderborn bearing vibration data;
- the corrected `Y[6]` / `vibration_1` channel;
- nominal 64 kHz sampling-rate standardization;
- strict 10-bearing leave-one-bearing-out (LOBO) evaluation;
- time-domain, absolute-STFT, order-normalized-STFT, and true-RMS-normalized-STFT representations;
- Random Forest and RBF-SVM classical baselines;
- corrected 1D-CNN and CNN-GRU baselines;
- recording-level bootstrap, failure analysis, and selected robustness/ablation analyses.

## Data

Raw Paderborn data are **not redistributed in this repository**. Obtain the dataset from its official source and place it in a local data directory. The scripts are intended to use a configurable data root rather than a machine-specific path.

See [`data/README.md`](data/README.md).

## Repository layout

```text
bearing-anomaly-robustness/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── data/
│   └── README.md
├── scripts/
│   ├── feature-generation scripts
│   ├── primary LOBO analyses
│   ├── robustness/ablation analyses
│   └── corrected deep-learning baselines
├── results/
│   ├── derived/
│   ├── figures/
│   └── tables/
└── docs/
    ├── reproducibility.md
    └── paper_results_map.md
```

## Reproducibility status

This is a **code release candidate**, not a claim that every historical experiment can be regenerated from the Python files alone. Several final analyses consume intermediate CSV/prediction files generated during the study. The public release documents those dependencies rather than silently recreating them with different code.

The original 162-script development archive should be retained separately as provenance and is not part of this public release.

## Important methodological conventions

Do not substitute the superseded channel/time-vector pipeline for the corrected pipeline. The evaluated real-data study uses `Y[6]` / `vibration_1` and nominal 64 kHz standardization. The primary real-data protocol is strict bearing-level LOBO.

The order-normalized representation uses the nominal rotational speed supplied for each recording to map frequency to rotational order; instantaneous tachometer/key-phase tracking and dynamic angle-domain resampling were not used in the evaluated pipeline.

## Running the code

1. Create a Python environment.
2. Install the packages listed in `requirements.txt`.
3. Obtain the Paderborn dataset separately.
4. Set `BEARING_DATA_ROOT` to the local dataset directory when a script requires an explicit data root.
5. Run the scripts in the sequence documented in `docs/reproducibility.md`.

Example on Windows PowerShell:

```powershell
$env:BEARING_DATA_ROOT = "D:\path\to\paderborn_data"
python scripts/corrected_stft_64khz.py
```

Example on Linux/macOS:

```bash
export BEARING_DATA_ROOT=/path/to/paderborn_data
python scripts/corrected_stft_64khz.py
```

No username, university directory, or machine-specific absolute path is required by the public scripts.

## Final reported reference results

The manuscript's primary best classical configuration is the order-normalized STFT with RBF-SVM, with pooled F1 = 0.6159. The corrected 1D-CNN and CNN-GRU baselines report F1 = 0.5714 and 0.6019, respectively. These values are manuscript reference values; the repository does not treat hard-coded summary scripts as substitutes for the underlying experiments.

## License

See `LICENSE`.
