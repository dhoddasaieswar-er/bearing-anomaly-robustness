# Public release checklist

- [x] Curated public source set
- [x] Personal absolute paths removed
- [x] Public scripts compile with `py_compile`
- [x] Raw Paderborn data excluded
- [x] Old CNN/CNN-GRU implementations excluded
- [x] Superseded CNN comparison values removed
- [x] Superseded Wilcoxon envelope-ablation analysis removed
- [x] Paper-to-code map added
- [x] Author metadata set in `CITATION.cff`
- [x] Derived-artifact integrity manifest included
- [x] Static privacy/portability audit completed
- [x] Release package excludes Python cache files
- [ ] Add final GitHub URL to `CITATION.cff` after repository creation
- [ ] Run end-to-end reproduction after obtaining the raw Paderborn dataset locally
- [ ] Create GitHub repository
- [ ] Optionally archive the tagged release on Zenodo

## Important

The public release does **not** redistribute the Paderborn raw recordings. Consequently, a full end-to-end rerun cannot be executed from the release package alone. The package is publication-aligned and documents the supplied derived artifacts and code dependencies without fabricating missing intermediate outputs.
