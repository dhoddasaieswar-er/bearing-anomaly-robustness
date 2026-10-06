# Final public release notes

This release is the publication-aligned public code package for the manuscript **Robustness Evaluation of Anomaly Detection in Non-Stationary Vibration Signals Under Variable Noise and Operating Conditions**.

## Included

- 22 curated Python scripts covering corrected preprocessing, classical LOBO evaluation, robustness/ablation analyses, selected synthetic studies, failure analysis, and corrected deep-learning baselines.
- 36 curated derived CSV artifacts supplied with the public release.
- Reproducibility documentation and a paper-to-code map.
- SHA-256 manifests with repository-relative paths.

## Deliberate exclusions

- Raw Paderborn recordings.
- Historical exploratory/duplicate scripts.
- Superseded CNN/CNN-GRU implementations and obsolete comparison values.
- The superseded Wilcoxon envelope-ablation analysis.
- Missing deep-learning intermediate CSV outputs that were not present in the supplied artifact archive.

## Reproduction note

A complete raw-data rerun requires the Paderborn dataset to be obtained separately and configured through `BEARING_DATA_ROOT`. The repository therefore does not claim that the supplied package alone can regenerate every historical intermediate file.

## Version

Public release version: **1.0.0**.

The GitHub URL should be added to `CITATION.cff` once the repository is created.


## Filename cleanup

The public Python scripts use descriptive filenames without the historical `stepNN_` development prefixes. Script contents were not changed by this rename; documentation and SHA-256 script manifests were updated to match the new filenames.
