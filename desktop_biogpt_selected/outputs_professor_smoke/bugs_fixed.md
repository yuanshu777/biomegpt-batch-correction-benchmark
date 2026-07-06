# Bugs Fixed and Safeguards Added

- Prevented notebook argparse auto-execution in Colab.
- Fixed notebook checklist cell that contained literal `\n` strings.
- Added stratified training-side calibration split for epoch and threshold selection.
- Applied Diseased-class synthetic augmentation during selection as well as final training.
- Restricted optional L1 feature selection to the selection-training split.
- Added confusion matrix, class-specific accuracy checks, AUROC/probability sanity checks, and probability histogram.
- Added species alignment, taxonomy completeness, and label-balance artifacts.
