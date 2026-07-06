options(stringsAsFactors = FALSE)

source(file.path(getwd(), "crc_benchmark_utils.R"))
crc_require_packages()
if (!requireNamespace("MMUPHin", quietly = TRUE)) {
  stop("MMUPHin is required to build the controlled benchmark.")
}

set.seed(20260617)

benchmark_dir <- file.path(getwd(), "crc_controlled_benchmark")
data_dir <- file.path(benchmark_dir, "data")
split_dir <- file.path(benchmark_dir, "splits")
report_dir <- file.path(benchmark_dir, "reports")
plot_dir <- file.path(report_dir, "plots")
manifest_dir <- file.path(benchmark_dir, "manifests")
method_dir <- file.path(benchmark_dir, "methods", "scgpt_biomegpt")
dirs <- c(
  benchmark_dir, data_dir, split_dir, report_dir, plot_dir,
  manifest_dir, method_dir
)
invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

data("CRC_abd", "CRC_meta", package = "MMUPHin")
CRC_meta <- CRC_meta[colnames(CRC_abd), , drop = FALSE]
metadata <- data.frame(
  sample_id = rownames(CRC_meta),
  studyID = as.character(CRC_meta$studyID),
  study_condition = as.character(CRC_meta$study_condition),
  check.names = FALSE
)
metadata$study_condition <- factor(
  metadata$study_condition,
  levels = c("control", "CRC")
)
if (anyNA(metadata$study_condition)) {
  stop("Unexpected CRC disease labels.")
}
metadata$study_condition <- as.character(metadata$study_condition)
crc_validate_inputs(CRC_abd, metadata)

prior_adjusted_path <- file.path(
  getwd(),
  "outputs_mmuphin_dataset_scouting",
  "mmuphin_crc_adjusted_abundance.rds"
)
if (file.exists(prior_adjusted_path)) {
  CRC_abd_mmuphin <- readRDS(prior_adjusted_path)
} else {
  fit <- MMUPHin::adjust_batch(
    feature_abd = CRC_abd,
    batch = "studyID",
    covariates = "study_condition",
    data = CRC_meta,
    control = list(verbose = FALSE)
  )
  CRC_abd_mmuphin <- fit$feature_abd_adj
}
crc_validate_inputs(CRC_abd_mmuphin, metadata, rownames(CRC_abd))

raw_csv <- file.path(data_dir, "crc_raw_abundance.csv")
raw_rds <- file.path(data_dir, "crc_raw_abundance.rds")
adjusted_csv <- file.path(data_dir, "crc_mmuphin_adjusted_abundance.csv")
adjusted_rds <- file.path(data_dir, "crc_mmuphin_adjusted_abundance.rds")
metadata_csv <- file.path(data_dir, "crc_metadata.csv")
crc_write_abundance(CRC_abd, raw_csv, raw_rds)
crc_write_abundance(CRC_abd_mmuphin, adjusted_csv, adjusted_rds)
write.csv(metadata, metadata_csv, row.names = FALSE)

make_stratified_folds <- function(labels, folds, seed) {
  set.seed(seed)
  assignment <- integer(length(labels))
  for (class_name in unique(labels)) {
    idx <- which(labels == class_name)
    assignment[idx] <- sample(rep(seq_len(folds), length.out = length(idx)))
  }
  assignment
}

outer_rows <- list()
for (repeat_id in 1:3) {
  outer_rows[[repeat_id]] <- data.frame(
    sample_id = metadata$sample_id,
    repeat_id = repeat_id,
    outer_fold = make_stratified_folds(
      metadata$studyID,
      folds = 5,
      seed = 20260617 + repeat_id
    )
  )
}
study_outer <- do.call(rbind, outer_rows)
write.csv(
  study_outer,
  file.path(split_dir, "study_prediction_outer_folds.csv"),
  row.names = FALSE
)

study_inner_rows <- list()
row_index <- 1
for (repeat_id in 1:3) {
  repeat_rows <- study_outer[study_outer$repeat_id == repeat_id, ]
  for (outer_fold in 1:5) {
    train_ids <- repeat_rows$sample_id[repeat_rows$outer_fold != outer_fold]
    train_labels <- metadata$studyID[match(train_ids, metadata$sample_id)]
    study_inner_rows[[row_index]] <- data.frame(
      sample_id = train_ids,
      repeat_id = repeat_id,
      outer_fold = outer_fold,
      inner_fold = make_stratified_folds(
        train_labels,
        folds = 5,
        seed = 20260617 + repeat_id * 100 + outer_fold
      )
    )
    row_index <- row_index + 1
  }
}
study_inner <- do.call(rbind, study_inner_rows)
write.csv(
  study_inner,
  gzfile(file.path(split_dir, "study_prediction_inner_folds.csv.gz")),
  row.names = FALSE
)

studies <- unique(metadata$studyID)
loso_rows <- lapply(studies, function(held_out) {
  data.frame(
    sample_id = metadata$sample_id,
    held_out_study = held_out,
    role = ifelse(metadata$studyID == held_out, "test", "train")
  )
})
disease_loso <- do.call(rbind, loso_rows)
write.csv(
  disease_loso,
  file.path(split_dir, "disease_loso_splits.csv"),
  row.names = FALSE
)

disease_inner_rows <- lapply(seq_along(studies), function(study_index) {
  held_out <- studies[study_index]
  train_ids <- metadata$sample_id[metadata$studyID != held_out]
  train_labels <- metadata$study_condition[
    match(train_ids, metadata$sample_id)
  ]
  data.frame(
    sample_id = train_ids,
    held_out_study = held_out,
    inner_fold = make_stratified_folds(
      train_labels,
      folds = 5,
      seed = 20260617 + 1000 + study_index
    )
  )
})
disease_inner <- do.call(rbind, disease_inner_rows)
write.csv(
  disease_inner,
  file.path(split_dir, "disease_loso_inner_folds.csv"),
  row.names = FALSE
)

set.seed(20260617)
permutation_matrix <- permute::shuffleSet(nrow(metadata), nset = 999)
saveRDS(
  permutation_matrix,
  file.path(split_dir, "permanova_permutation_matrix.rds")
)
permutation_export <- data.frame(
  permutation_id = seq_len(nrow(permutation_matrix)),
  permutation_matrix,
  check.names = FALSE
)
names(permutation_export)[-1] <- sprintf(
  "sample_position_%03d",
  seq_len(ncol(permutation_matrix))
)
write.csv(
  permutation_export,
  gzfile(file.path(split_dir, "permanova_permutation_matrix.csv.gz")),
  row.names = FALSE
)
permanova_config <- data.frame(
  setting = c(
    "sample_order_source",
    "distance",
    "permutations",
    "seed",
    "study_unadjusted_formula",
    "condition_unadjusted_formula",
    "joint_formula",
    "joint_test"
  ),
  value = c(
    "../data/crc_metadata.csv",
    "Bray-Curtis",
    "999 fixed rows",
    "20260617",
    "distance ~ studyID",
    "distance ~ study_condition",
    "distance ~ studyID + study_condition",
    "by = margin"
  )
)
write.csv(
  permanova_config,
  file.path(split_dir, "permanova_config.csv"),
  row.names = FALSE
)

raw_metrics <- crc_evaluate_method(
  CRC_abd,
  metadata,
  "raw",
  split_dir,
  plot_dir
)
mmuphin_metrics <- crc_evaluate_method(
  CRC_abd_mmuphin,
  metadata,
  "mmuphin",
  split_dir,
  plot_dir
)
metrics <- rbind(raw_metrics, mmuphin_metrics)
metrics_path <- file.path(report_dir, "crc_raw_vs_mmuphin_metrics.csv")
write.csv(metrics, metrics_path, row.names = FALSE)

metric_value <- function(method, metric) {
  metrics$estimate[metrics$method == method & metrics$metric == metric][1]
}
metric_p <- function(method, metric) {
  metrics$p_value[metrics$method == method & metrics$metric == metric][1]
}
raw_study <- metric_value("raw", "study_R2_condition_controlled")
adj_study <- metric_value("mmuphin", "study_R2_condition_controlled")
raw_condition <- metric_value("raw", "condition_R2_study_controlled")
adj_condition <- metric_value("mmuphin", "condition_R2_study_controlled")
raw_study_ba <- metric_value("raw", "study_prediction_balanced_accuracy")
adj_study_ba <- metric_value("mmuphin", "study_prediction_balanced_accuracy")
raw_auc <- metric_value("raw", "disease_LOSO_mean_within_study_AUC")
adj_auc <- metric_value("mmuphin", "disease_LOSO_mean_within_study_AUC")

comparison <- data.frame(
  metric = c(
    "Study PERMANOVA R2, condition-controlled",
    "Condition PERMANOVA R2, study-controlled",
    "Study prediction balanced accuracy",
    "Disease LOSO mean within-study AUC",
    "Disease LOSO balanced accuracy"
  ),
  raw = c(
    raw_study,
    raw_condition,
    raw_study_ba,
    raw_auc,
    metric_value("raw", "disease_LOSO_balanced_accuracy")
  ),
  mmuphin = c(
    adj_study,
    adj_condition,
    adj_study_ba,
    adj_auc,
    metric_value("mmuphin", "disease_LOSO_balanced_accuracy")
  )
)
comparison$absolute_change <- comparison$mmuphin - comparison$raw
write.csv(
  comparison,
  file.path(report_dir, "crc_raw_vs_mmuphin_comparison.csv"),
  row.names = FALSE
)

plot_comparison <- comparison[1:4, ]
plot_long <- rbind(
  data.frame(metric = plot_comparison$metric, method = "Raw", value = plot_comparison$raw),
  data.frame(metric = plot_comparison$metric, method = "MMUPHin", value = plot_comparison$mmuphin)
)
plot_long$metric <- factor(
  plot_long$metric,
  levels = rev(plot_comparison$metric)
)
metric_plot <- ggplot2::ggplot(
  plot_long,
  ggplot2::aes(metric, value, fill = method)
) +
  ggplot2::geom_col(position = "dodge", width = 0.72) +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_manual(values = c(Raw = "#4C78A8", MMUPHin = "#E45756")) +
  ggplot2::labs(
    title = "CRC controlled benchmark: raw vs MMUPHin",
    x = NULL,
    y = "Metric value",
    fill = NULL
  ) +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(legend.position = "top")
ggplot2::ggsave(
  file.path(report_dir, "crc_raw_vs_mmuphin_metric_comparison.png"),
  metric_plot,
  width = 10,
  height = 5.8,
  dpi = 180
)

counts <- as.data.frame(
  table(metadata$studyID, metadata$study_condition),
  responseName = "n"
)
names(counts)[1:2] <- c("studyID", "study_condition")
write.csv(
  counts,
  file.path(report_dir, "crc_study_condition_counts.csv"),
  row.names = FALSE
)

report_md <- c(
  "# CRC Controlled Benchmark: Raw vs MMUPHin",
  "",
  "## Benchmark contract",
  "",
  sprintf(
    "The benchmark contains %d microbial features, %d samples, %d studies, and two disease labels. All matrices use feature-by-sample orientation and the exact sample order in `data/crc_metadata.csv`.",
    nrow(CRC_abd), ncol(CRC_abd), length(unique(metadata$studyID))
  ),
  "",
  "Evaluation is frozen through explicit outer and inner prediction folds plus a fixed 999-row PERMANOVA permutation matrix. The same evaluator must be used for every correction method.",
  "",
  "**Evaluation scope:** MMUPHin adjustment is fit once on the full dataset with `study_condition` as a preservation covariate. The LOSO classifier therefore measures disease-signal retention in a globally corrected representation; it is not an inductive deployment estimate for a correction model that never saw the held-out study.",
  "",
  "## Before and after",
  "",
  "| Metric | Raw | MMUPHin | Change |",
  "|---|---:|---:|---:|",
  sprintf(
    "| Study PERMANOVA R2, condition-controlled | %.4f | %.4f | %+.4f |",
    raw_study, adj_study, adj_study - raw_study
  ),
  sprintf(
    "| Condition PERMANOVA R2, study-controlled | %.4f | %.4f | %+.4f |",
    raw_condition, adj_condition, adj_condition - raw_condition
  ),
  sprintf(
    "| Study prediction balanced accuracy | %.4f | %.4f | %+.4f |",
    raw_study_ba, adj_study_ba, adj_study_ba - raw_study_ba
  ),
  sprintf(
    "| Disease LOSO mean within-study AUC | %.4f | %.4f | %+.4f |",
    raw_auc, adj_auc, adj_auc - raw_auc
  ),
  "",
  "## Interpretation",
  "",
  sprintf(
    "MMUPHin reduces condition-controlled study R2 by %.1f%% relative and lowers study-prediction balanced accuracy by %.3f. Disease signal is retained: study-controlled condition R2 changes from %.4f to %.4f, while mean within-study LOSO AUC changes from %.3f to %.3f.",
    100 * (raw_study - adj_study) / raw_study,
    raw_study_ba - adj_study_ba,
    raw_condition,
    adj_condition,
    raw_auc,
    adj_auc
  ),
  "",
  sprintf(
    "The study effect remains statistically detectable after correction (fixed-permutation PERMANOVA p = %.3g), so MMUPHin is a meaningful baseline rather than a claim of complete batch removal.",
    metric_p("mmuphin", "study_R2_condition_controlled")
  ),
  "",
  "## Files for the next method",
  "",
  "- Input abundance: `data/crc_raw_abundance.csv`",
  "- Metadata: `data/crc_metadata.csv`",
  "- Required output contract: `methods/scgpt_biomegpt/README.md`",
  "- Common evaluator: `../evaluate_crc_method.R`",
  "- Frozen splits and permutations: `splits/`",
  "",
  "## Visual diagnostics",
  "",
  "![Metric comparison](crc_raw_vs_mmuphin_metric_comparison.png)",
  "",
  "### PCA by study",
  "",
  "| Raw | MMUPHin |",
  "|---|---|",
  "| ![Raw PCA by study](plots/raw_pca_by_study.png) | ![MMUPHin PCA by study](plots/mmuphin_pca_by_study.png) |",
  "",
  "### PCA by disease condition",
  "",
  "| Raw | MMUPHin |",
  "|---|---|",
  "| ![Raw PCA by condition](plots/raw_pca_by_condition.png) | ![MMUPHin PCA by condition](plots/mmuphin_pca_by_condition.png) |"
)
writeLines(report_md, file.path(report_dir, "crc_raw_vs_mmuphin_report.md"))

html_escape <- function(x) {
  x <- gsub("&", "&amp;", x, fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  gsub(">", "&gt;", x, fixed = TRUE)
}
table_rows <- paste(
  vapply(seq_len(nrow(comparison)), function(i) {
    sprintf(
      "<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%+.4f</td></tr>",
      html_escape(comparison$metric[i]),
      comparison$raw[i],
      comparison$mmuphin[i],
      comparison$absolute_change[i]
    )
  }, character(1)),
  collapse = "\n"
)
report_html <- sprintf(
  paste0(
    "<!doctype html><html><head><meta charset='utf-8'>",
    "<title>CRC Raw vs MMUPHin</title><style>",
    "body{font-family:Arial,sans-serif;color:#1f2933;max-width:1120px;margin:36px auto;padding:0 24px;line-height:1.5}",
    "h1{font-size:30px;margin-bottom:6px}h2{margin-top:34px;border-bottom:1px solid #d9e2ec;padding-bottom:7px}",
    ".lede{color:#52606d;font-size:17px}.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}",
    ".kpi{border:1px solid #bcccdc;padding:16px;border-radius:6px;background:#f8fafc}.kpi b{display:block;font-size:24px;color:#0b7285}",
    "table{border-collapse:collapse;width:100%%;margin:16px 0}th,td{border:1px solid #d9e2ec;padding:9px;text-align:right}",
    "th:first-child,td:first-child{text-align:left}th{background:#243b53;color:white}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}",
    "img{max-width:100%%;border:1px solid #d9e2ec}.note{background:#eef6f7;border-left:4px solid #0b7285;padding:14px}",
    "code{background:#f0f4f8;padding:2px 5px}@media(max-width:760px){.kpis,.grid{grid-template-columns:1fr}}",
    "</style></head><body>",
    "<h1>CRC Controlled Benchmark</h1><p class='lede'>Raw abundance versus MMUPHin adjustment under one frozen evaluation protocol.</p>",
    "<div class='kpis'><div class='kpi'><b>%.1f%%</b>relative reduction in study R2</div>",
    "<div class='kpi'><b>%.3f</b>MMUPHin disease LOSO AUC</div>",
    "<div class='kpi'><b>%d</b>samples across %d studies</div></div>",
    "<h2>Before and after</h2><table><thead><tr><th>Metric</th><th>Raw</th><th>MMUPHin</th><th>Change</th></tr></thead><tbody>%s</tbody></table>",
    "<img src='crc_raw_vs_mmuphin_metric_comparison.png' alt='Metric comparison'>",
    "<h2>Interpretation</h2><p class='note'>MMUPHin lowers study signal while retaining cross-study disease discrimination. Residual study structure remains detectable, so this is a strong baseline, not complete batch removal.</p>",
    "<p><strong>Evaluation scope:</strong> MMUPHin is fit globally with disease condition as a preservation covariate. LOSO AUC is therefore a controlled signal-retention probe in the corrected representation, not an inductive estimate for an unseen-study correction pipeline.</p>",
    "<h2>PCA colored by study</h2><div class='grid'><img src='plots/raw_pca_by_study.png'><img src='plots/mmuphin_pca_by_study.png'></div>",
    "<h2>PCA colored by condition</h2><div class='grid'><img src='plots/raw_pca_by_condition.png'><img src='plots/mmuphin_pca_by_condition.png'></div>",
    "<h2>Reproducibility contract</h2><p>Use <code>data/crc_raw_abundance.csv</code>, <code>data/crc_metadata.csv</code>, and every manifest in <code>splits/</code>. Future methods must emit the same feature-by-sample matrix schema and be scored by <code>evaluate_crc_method.R</code>.</p>",
    "</body></html>"
  ),
  100 * (raw_study - adj_study) / raw_study,
  adj_auc,
  nrow(metadata),
  length(unique(metadata$studyID)),
  table_rows
)
writeLines(report_html, file.path(report_dir, "crc_raw_vs_mmuphin_report.html"))

method_readme <- c(
  "# scGPT-style / BiomeGPT Benchmark Handoff",
  "",
  "## Inputs",
  "",
  "- `../../data/crc_raw_abundance.csv`: 484 feature rows by 551 sample columns, with the first column named `feature`.",
  "- `../../data/crc_metadata.csv`: exactly `sample_id`, `studyID`, and `study_condition`.",
  "- `../../splits/`: frozen evaluation assignments and PERMANOVA permutations.",
  "",
  "## Required method output",
  "",
  "Write `scgpt_biomegpt_adjusted_abundance.csv` in this directory.",
  "",
  "The output must:",
  "",
  "1. Have a first column named `feature`.",
  "2. Contain exactly the same 484 feature IDs, in the same order as the raw matrix.",
  "3. Contain exactly the same 551 sample columns, in the same order as the raw matrix.",
  "4. Contain only finite numeric adjusted abundance values.",
  "5. Be generated without using held-out disease labels from the frozen LOSO test study.",
  "",
  "## Evaluation",
  "",
  "From the project root, run:",
  "",
  "```powershell",
  "& 'C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe' .\\evaluate_crc_method.R --method scgpt_biomegpt --matrix .\\crc_controlled_benchmark\\methods\\scgpt_biomegpt\\scgpt_biomegpt_adjusted_abundance.csv",
  "```",
  "",
  "The evaluator uses the identical study-prediction folds, disease LOSO folds, classifier inner folds, PERMANOVA permutations, transforms, and metrics used for raw and MMUPHin.",
  "",
  "## Leakage rule",
  "",
  "If the correction model is trained or tuned separately for disease LOSO evaluation, each held-out study must remain completely unseen during fitting and hyperparameter selection. A global matrix corrected using all samples may be used for unsupervised study/PERMANOVA diagnostics, but it must not be presented as leakage-free LOSO disease performance unless the correction itself is label-free and transductive use is explicitly accepted."
)
writeLines(method_readme, file.path(method_dir, "README.md"))

schema <- data.frame(
  field = c(
    "orientation",
    "feature_id_column",
    "feature_count",
    "sample_count",
    "feature_order_reference",
    "sample_order_reference",
    "required_output_filename"
  ),
  value = c(
    "features_by_samples",
    "feature",
    nrow(CRC_abd),
    ncol(CRC_abd),
    "../../data/crc_raw_abundance.csv",
    "../../data/crc_metadata.csv",
    "scgpt_biomegpt_adjusted_abundance.csv"
  )
)
write.csv(schema, file.path(method_dir, "output_schema.csv"), row.names = FALSE)

readme <- c(
  "# MMUPHin CRC Controlled Benchmark",
  "",
  "This directory is the frozen benchmark package for comparing batch-correction methods on the MMUPHin colorectal cancer dataset.",
  "",
  "## Data",
  "",
  "- `data/crc_raw_abundance.csv`: authentic MMUPHin `CRC_abd` matrix.",
  "- `data/crc_mmuphin_adjusted_abundance.csv`: MMUPHin-adjusted matrix.",
  "- `data/crc_metadata.csv`: sample ID, study, and disease label only.",
  "",
  "All abundance matrices are feature-by-sample with an explicit `feature` first column. RDS copies preserve exact numeric values.",
  "",
  "## Frozen evaluation",
  "",
  "- Study prediction: three repeats of stratified five-fold outer CV with fixed inner folds.",
  "- Disease prediction: leave one complete study out, with fixed inner folds.",
  "- PERMANOVA: Bray-Curtis distance and one fixed set of 999 permutations.",
  "- Primary batch metric: study R2 controlling for condition.",
  "- Primary biological metric: condition R2 controlling for study.",
  "",
  "## Evaluation scope",
  "",
  "The MMUPHin matrix is adjusted globally using `study_condition` as a covariate. Disease LOSO metrics quantify retained cross-study disease separability after correction. They should not be described as leakage-free deployment performance for a correction model trained without the held-out study.",
  "",
  "## Reports",
  "",
  "- `reports/crc_raw_vs_mmuphin_report.html`: clean before/after report.",
  "- `reports/crc_raw_vs_mmuphin_report.md`: portable Markdown version.",
  "- `reports/crc_raw_vs_mmuphin_metrics.csv`: complete metrics.",
  "",
  "## Adding a method",
  "",
  "Follow `methods/scgpt_biomegpt/README.md`, then run the common evaluator from the project root."
)
writeLines(readme, file.path(benchmark_dir, "README.md"))

artifact_paths <- list.files(
  benchmark_dir,
  recursive = TRUE,
  full.names = TRUE,
  all.files = FALSE
)
artifact_paths <- artifact_paths[file.info(artifact_paths)$isdir == FALSE]
manifest <- data.frame(
  path = gsub(
    "\\\\",
    "/",
    substring(artifact_paths, nchar(benchmark_dir) + 2)
  ),
  bytes = file.info(artifact_paths)$size,
  md5 = unname(tools::md5sum(artifact_paths)),
  stringsAsFactors = FALSE
)
write.csv(
  manifest,
  file.path(manifest_dir, "artifact_manifest.csv"),
  row.names = FALSE
)

cat("CRC controlled benchmark prepared at:", benchmark_dir, "\n")
cat("Raw study R2:", raw_study, "\n")
cat("MMUPHin study R2:", adj_study, "\n")
cat("Raw disease LOSO mean AUC:", raw_auc, "\n")
cat("MMUPHin disease LOSO mean AUC:", adj_auc, "\n")
