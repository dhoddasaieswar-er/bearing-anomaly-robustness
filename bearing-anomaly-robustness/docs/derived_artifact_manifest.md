# Derived artifact manifest

The supplied `csvrecords.zip` contained 227 files. The public release includes 36 curated CSV artifacts rather than the complete historical archive.

## Included

- Corrected 64-kHz STFT feature tables and comparison results.
- Time-domain and STFT feature tables used by the primary classical pipeline.
- Step 85 classifier-control outputs.
- Step 125 prediction records used by the recording-level failure/bootstrapping analysis.
- Step 129 paired-validation outputs.
- Step 140 clustered-bootstrap outputs.
- Step 152 envelope-ablation metrics.
- Step 153 OCSVM outputs.
- Step 154 OCSVM sensitivity outputs.
- Step 141 manuscript-level result summaries.
- Bearing-level publication/statistical summary tables.

## Deliberately not included

Historical exploratory CSVs, obsolete pipeline outputs, duplicate diagnostics, and unrelated intermediate results are excluded.

The supplied archive did **not** contain the following downstream outputs referenced by some public scripts:

- Step 92/93/94/95/96 detailed failure-analysis CSVs.
- Step 155 CNN-GRU output CSVs.
- Step 156 CNN output CSVs.
- Step 153 OCSVM files were present and are included.

Those missing files are not recreated or invented by this release.
