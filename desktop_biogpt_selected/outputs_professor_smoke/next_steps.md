# Next Steps

1. Run the notebook with `SMOKE_TEST = False` on Colab.
2. Inspect `exval_hd_metrics.json`, `exval_confusion_matrix.csv`, and `exval_probability_histogram.png`.
3. Compare default run with `USE_L1_FEATURE_SELECTION = True`.
4. Use macro-F1 as the primary model-selection metric.
5. Treat UMAPs as exploratory and cite kNN purity / sample separation summaries as quantitative support.
6. If ExVal predictions collapse to one class in a full run, inspect threshold, probability histogram, calibration split balance, and class-specific accuracies before reporting.
