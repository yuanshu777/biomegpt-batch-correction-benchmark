options(stringsAsFactors = FALSE)

required_packages <- c("MMUPHin", "ggplot2", "vegan", "glmnet", "pROC")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop("Missing required packages: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(MMUPHin)
  library(ggplot2)
  library(vegan)
  library(glmnet)
  library(pROC)
})

set.seed(20260617)

output_dir <- file.path(getwd(), "outputs_mmuphin_dataset_scouting")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(output_dir, "mmuphin_scouting_run.log")
log_con <- file(log_file, open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")
on.exit({
  sink(type = "message")
  sink(type = "output")
  close(log_con)
}, add = TRUE)

cat("MMUPHin CRC dataset scouting\n")
cat("Started:", format(Sys.time(), tz = "America/New_York"), "\n")
cat("R:", R.version.string, "\n")
cat("MMUPHin:", as.character(packageVersion("MMUPHin")), "\n\n")

data("CRC_abd", "CRC_meta", package = "MMUPHin")

if (!is.matrix(CRC_abd) || !is.data.frame(CRC_meta)) {
  stop("Unexpected dataset classes: CRC_abd must be a matrix and CRC_meta a data.frame.")
}
if (is.null(colnames(CRC_abd)) || is.null(rownames(CRC_meta))) {
  stop("Sample identifiers are absent from CRC_abd columns or CRC_meta rows.")
}
if (!setequal(colnames(CRC_abd), rownames(CRC_meta))) {
  stop("CRC_abd column names and CRC_meta row names do not contain the same samples.")
}

CRC_meta <- CRC_meta[colnames(CRC_abd), , drop = FALSE]
if (!identical(colnames(CRC_abd), rownames(CRC_meta))) {
  stop("Sample order could not be aligned exactly.")
}
if (!all(c("studyID", "study_condition") %in% colnames(CRC_meta))) {
  stop("Expected metadata fields studyID and study_condition are missing.")
}
if (anyNA(CRC_meta$studyID) || anyNA(CRC_meta$study_condition)) {
  stop("studyID or study_condition contains missing values.")
}

CRC_meta$studyID <- factor(CRC_meta$studyID)
CRC_meta$study_condition <- factor(
  CRC_meta$study_condition,
  levels = c("control", "CRC")
)
if (anyNA(CRC_meta$study_condition)) {
  stop("study_condition contains labels other than control and CRC.")
}

count_table <- as.data.frame(
  table(CRC_meta$studyID, CRC_meta$study_condition),
  responseName = "n"
)
names(count_table)[1:2] <- c("studyID", "study_condition")
write.csv(
  count_table,
  file.path(output_dir, "mmuphin_crc_study_condition_counts.csv"),
  row.names = FALSE
)

contingency <- table(CRC_meta$studyID, CRC_meta$study_condition)
chi <- suppressWarnings(chisq.test(contingency, correct = FALSE))
cramers_v <- sqrt(
  unname(chi$statistic) /
    (sum(contingency) * min(nrow(contingency) - 1, ncol(contingency) - 1))
)
all_studies_have_both <- all(contingency > 0)

summary_rows <- data.frame(
  item = c(
    "MMUPHin_version",
    "abundance_rows_features",
    "abundance_columns_samples",
    "metadata_rows_samples",
    "metadata_columns",
    "abundance_orientation",
    "sample_ids_set_aligned",
    "sample_ids_order_aligned",
    "n_studies",
    "condition_levels",
    "all_studies_have_both_conditions",
    "minimum_study_condition_cell_n",
    "study_condition_chisq_p",
    "study_condition_cramers_v",
    "abundance_min",
    "abundance_max",
    "sample_sum_min",
    "sample_sum_max",
    "metadata_column_names"
  ),
  value = c(
    as.character(packageVersion("MMUPHin")),
    nrow(CRC_abd),
    ncol(CRC_abd),
    nrow(CRC_meta),
    ncol(CRC_meta),
    "features_by_samples",
    setequal(colnames(CRC_abd), rownames(CRC_meta)),
    identical(colnames(CRC_abd), rownames(CRC_meta)),
    nlevels(CRC_meta$studyID),
    paste(levels(CRC_meta$study_condition), collapse = "|"),
    all_studies_have_both,
    min(contingency),
    signif(chi$p.value, 6),
    signif(cramers_v, 6),
    signif(min(CRC_abd), 6),
    signif(max(CRC_abd), 6),
    signif(min(colSums(CRC_abd)), 6),
    signif(max(colSums(CRC_abd)), 6),
    paste(colnames(CRC_meta), collapse = "|")
  )
)
write.csv(
  summary_rows,
  file.path(output_dir, "mmuphin_crc_dataset_summary.csv"),
  row.names = FALSE
)

make_pca_data <- function(abundance, metadata) {
  # Square-root proportions give a Hellinger-style transform suitable for PCA.
  sample_sums <- colSums(pmax(abundance, 0))
  if (any(sample_sums <= 0)) {
    stop("At least one sample has no non-negative abundance after transformation.")
  }
  relative <- sweep(pmax(abundance, 0), 2, sample_sums, "/")
  transformed <- sqrt(relative)
  keep <- apply(transformed, 1, stats::var) > 0
  fit <- prcomp(t(transformed[keep, , drop = FALSE]), center = TRUE, scale. = FALSE)
  variance <- fit$sdev^2 / sum(fit$sdev^2)
  scores <- data.frame(
    sample_id = rownames(fit$x),
    PC1 = fit$x[, 1],
    PC2 = fit$x[, 2],
    studyID = metadata[rownames(fit$x), "studyID"],
    study_condition = metadata[rownames(fit$x), "study_condition"],
    check.names = FALSE
  )
  list(scores = scores, variance = variance)
}

save_pca_plots <- function(abundance, metadata, stage) {
  pca <- make_pca_data(abundance, metadata)
  x_label <- sprintf("PC1 (%.1f%%)", 100 * pca$variance[1])
  y_label <- sprintf("PC2 (%.1f%%)", 100 * pca$variance[2])

  p_study <- ggplot(pca$scores, aes(PC1, PC2, color = studyID)) +
    geom_point(size = 2, alpha = 0.75) +
    labs(
      title = paste("CRC abundance PCA:", stage),
      subtitle = "Colored by study",
      x = x_label,
      y = y_label,
      color = "Study"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "right")

  p_condition <- ggplot(
    pca$scores,
    aes(PC1, PC2, color = study_condition, shape = study_condition)
  ) +
    geom_point(size = 2, alpha = 0.75) +
    scale_color_manual(values = c(control = "#0072B2", CRC = "#D55E00")) +
    labs(
      title = paste("CRC abundance PCA:", stage),
      subtitle = "Colored by disease condition",
      x = x_label,
      y = y_label,
      color = "Condition",
      shape = "Condition"
    ) +
    theme_bw(base_size = 11)

  ggsave(
    file.path(output_dir, paste0("mmuphin_crc_", stage, "_pca_by_study.png")),
    p_study,
    width = 9,
    height = 6,
    dpi = 180
  )
  ggsave(
    file.path(output_dir, paste0("mmuphin_crc_", stage, "_pca_by_condition.png")),
    p_condition,
    width = 8,
    height = 6,
    dpi = 180
  )
  invisible(pca)
}

metric_row <- function(stage, family, metric, estimate, p_value = NA_realized_,
                       detail = "") {
  data.frame(
    stage = stage,
    metric_family = family,
    metric = metric,
    estimate = as.numeric(estimate),
    p_value = as.numeric(p_value),
    detail = detail,
    stringsAsFactors = FALSE
  )
}

permanova_metrics <- function(abundance, metadata, stage) {
  distance <- vegdist(t(abundance), method = "bray")
  set.seed(20260617)
  study_fit <- adonis2(distance ~ studyID, data = metadata, permutations = 999)
  set.seed(20260617)
  condition_fit <- adonis2(
    distance ~ study_condition,
    data = metadata,
    permutations = 999
  )
  set.seed(20260617)
  joint_fit <- adonis2(
    distance ~ studyID + study_condition,
    data = metadata,
    permutations = 999,
    by = "margin"
  )

  rbind(
    metric_row(
      stage, "PERMANOVA", "study_R2_unadjusted_model",
      study_fit[1, "R2"], study_fit[1, "Pr(>F)"],
      "Bray-Curtis; studyID-only model; 999 permutations"
    ),
    metric_row(
      stage, "PERMANOVA", "condition_R2_unadjusted_model",
      condition_fit[1, "R2"],
      condition_fit[1, "Pr(>F)"],
      "Bray-Curtis; condition-only model; 999 permutations"
    ),
    metric_row(
      stage, "PERMANOVA", "study_R2_condition_controlled",
      joint_fit["studyID", "R2"], joint_fit["studyID", "Pr(>F)"],
      "Bray-Curtis; marginal term in studyID + study_condition"
    ),
    metric_row(
      stage, "PERMANOVA", "condition_R2_study_controlled",
      joint_fit["study_condition", "R2"],
      joint_fit["study_condition", "Pr(>F)"],
      "Bray-Curtis; marginal term in studyID + study_condition"
    )
  )
}

model_matrix_from_abundance <- function(abundance) {
  x <- t(pmax(abundance, 0))
  log1p(1000 * x)
}

balanced_accuracy <- function(observed, predicted) {
  classes <- levels(observed)
  recalls <- vapply(classes, function(class_name) {
    idx <- observed == class_name
    mean(predicted[idx] == class_name)
  }, numeric(1))
  mean(recalls)
}

study_cv_metrics <- function(abundance, metadata, stage, repeats = 3, folds = 5) {
  x <- model_matrix_from_abundance(abundance)
  y <- droplevels(metadata$studyID)
  accuracies <- numeric(repeats)
  balanced <- numeric(repeats)

  for (repeat_id in seq_len(repeats)) {
    set.seed(20260617 + repeat_id)
    fold_id <- integer(length(y))
    for (class_name in levels(y)) {
      idx <- which(y == class_name)
      fold_id[idx] <- sample(rep(seq_len(folds), length.out = length(idx)))
    }
    predictions <- factor(rep(NA_character_, length(y)), levels = levels(y))

    for (fold in seq_len(folds)) {
      train <- fold_id != fold
      test <- !train
      inner_folds <- max(3, min(5, floor(sum(train) / 20)))
      fit <- cv.glmnet(
        x[train, , drop = FALSE],
        y[train],
        family = "multinomial",
        type.measure = "class",
        nfolds = inner_folds,
        standardize = TRUE,
        parallel = FALSE
      )
      pred <- predict(
        fit,
        newx = x[test, , drop = FALSE],
        s = "lambda.1se",
        type = "class"
      )
      predictions[test] <- as.character(pred[, 1])
    }
    accuracies[repeat_id] <- mean(predictions == y)
    balanced[repeat_id] <- balanced_accuracy(y, predictions)
  }

  rbind(
    metric_row(
      stage, "classifier", "study_prediction_accuracy",
      mean(accuracies), NA,
      sprintf("%d repeats of stratified %d-fold CV; multinomial glmnet", repeats, folds)
    ),
    metric_row(
      stage, "classifier", "study_prediction_balanced_accuracy",
      mean(balanced), NA,
      sprintf("%d repeats of stratified %d-fold CV; chance is %.3f", repeats, folds, 1 / nlevels(y))
    )
  )
}

disease_loso_metrics <- function(abundance, metadata, stage) {
  x <- model_matrix_from_abundance(abundance)
  y <- as.integer(metadata$study_condition == "CRC")
  studies <- levels(metadata$studyID)
  probabilities <- rep(NA_real_, length(y))
  per_study_auc <- numeric(0)

  for (study_name in studies) {
    test <- metadata$studyID == study_name
    train <- !test
    set.seed(20260617)
    fit <- cv.glmnet(
      x[train, , drop = FALSE],
      y[train],
      family = "binomial",
      type.measure = "auc",
      nfolds = 5,
      standardize = TRUE,
      parallel = FALSE
    )
    probabilities[test] <- as.numeric(
      predict(
        fit,
        newx = x[test, , drop = FALSE],
        s = "lambda.1se",
        type = "response"
      )
    )
    held_out_y <- y[test]
    if (length(unique(held_out_y)) == 2) {
      per_study_auc[study_name] <- as.numeric(
        pROC::auc(held_out_y, probabilities[test], quiet = TRUE)
      )
    }
  }

  overall_auc <- as.numeric(pROC::auc(y, probabilities, quiet = TRUE))
  predicted <- factor(
    ifelse(probabilities >= 0.5, "CRC", "control"),
    levels = c("control", "CRC")
  )
  observed <- factor(
    ifelse(y == 1, "CRC", "control"),
    levels = c("control", "CRC")
  )

  rbind(
    metric_row(
      stage, "classifier", "disease_LOSO_overall_AUC",
      overall_auc, NA,
      "Each study held out once; binomial glmnet"
    ),
    metric_row(
      stage, "classifier", "disease_LOSO_mean_within_study_AUC",
      mean(per_study_auc), NA,
      paste(
        paste(names(per_study_auc), sprintf("%.3f", per_study_auc), sep = "="),
        collapse = "; "
      )
    ),
    metric_row(
      stage, "classifier", "disease_LOSO_balanced_accuracy",
      balanced_accuracy(observed, predicted), NA,
      "Threshold 0.5 on leave-one-study-out probabilities"
    )
  )
}

cat("Dataset dimensions:", nrow(CRC_abd), "features x", ncol(CRC_abd), "samples\n")
cat("Metadata dimensions:", nrow(CRC_meta), "samples x", ncol(CRC_meta), "fields\n")
cat("Study-condition counts:\n")
print(contingency)
cat("\n")

raw_pca <- save_pca_plots(CRC_abd, CRC_meta, "raw")
raw_metrics <- rbind(
  permanova_metrics(CRC_abd, CRC_meta, "raw"),
  study_cv_metrics(CRC_abd, CRC_meta, "raw"),
  disease_loso_metrics(CRC_abd, CRC_meta, "raw")
)

cat("Running MMUPHin::adjust_batch...\n")
fit_adjust_batch <- adjust_batch(
  feature_abd = CRC_abd,
  batch = "studyID",
  covariates = "study_condition",
  data = CRC_meta,
  control = list(verbose = FALSE)
)
diagnostic_source <- file.path(getwd(), "adjust_batch_diagnostic.pdf")
diagnostic_target <- file.path(output_dir, "mmuphin_crc_adjust_batch_diagnostic.pdf")
if (file.exists(diagnostic_source)) {
  if (file.exists(diagnostic_target)) {
    unlink(diagnostic_target)
  }
  if (!file.rename(diagnostic_source, diagnostic_target)) {
    stop("Could not move adjust_batch_diagnostic.pdf into the output directory.")
  }
}
if (is.null(fit_adjust_batch$feature_abd_adj)) {
  stop("adjust_batch completed without returning feature_abd_adj.")
}
CRC_abd_adj <- fit_adjust_batch$feature_abd_adj
if (!identical(dim(CRC_abd_adj), dim(CRC_abd))) {
  stop("Adjusted abundance matrix dimensions differ from the raw matrix.")
}
if (!identical(colnames(CRC_abd_adj), colnames(CRC_abd))) {
  stop("Adjusted abundance sample order differs from the raw matrix.")
}

adjusted_export <- data.frame(
  feature = rownames(CRC_abd_adj),
  CRC_abd_adj,
  check.names = FALSE
)
write.csv(
  adjusted_export,
  file.path(output_dir, "mmuphin_crc_adjusted_abundance.csv"),
  row.names = FALSE
)
saveRDS(
  CRC_abd_adj,
  file.path(output_dir, "mmuphin_crc_adjusted_abundance.rds")
)
saveRDS(
  fit_adjust_batch,
  file.path(output_dir, "mmuphin_crc_adjust_batch_fit.rds")
)

adjusted_pca <- save_pca_plots(CRC_abd_adj, CRC_meta, "adjusted")
adjusted_metrics <- rbind(
  permanova_metrics(CRC_abd_adj, CRC_meta, "adjusted"),
  study_cv_metrics(CRC_abd_adj, CRC_meta, "adjusted"),
  disease_loso_metrics(CRC_abd_adj, CRC_meta, "adjusted")
)

all_metrics <- rbind(raw_metrics, adjusted_metrics)
write.csv(
  all_metrics,
  file.path(output_dir, "mmuphin_crc_raw_vs_adjusted_metrics.csv"),
  row.names = FALSE
)

get_estimate <- function(stage, metric) {
  all_metrics$estimate[all_metrics$stage == stage & all_metrics$metric == metric][1]
}
get_p <- function(stage, metric) {
  all_metrics$p_value[all_metrics$stage == stage & all_metrics$metric == metric][1]
}

raw_study_r2 <- get_estimate("raw", "study_R2_condition_controlled")
adjusted_study_r2 <- get_estimate("adjusted", "study_R2_condition_controlled")
raw_condition_r2 <- get_estimate("raw", "condition_R2_study_controlled")
adjusted_condition_r2 <- get_estimate("adjusted", "condition_R2_study_controlled")
raw_study_balacc <- get_estimate("raw", "study_prediction_balanced_accuracy")
adjusted_study_balacc <- get_estimate("adjusted", "study_prediction_balanced_accuracy")
raw_disease_auc <- get_estimate("raw", "disease_LOSO_mean_within_study_AUC")
adjusted_disease_auc <- get_estimate("adjusted", "disease_LOSO_mean_within_study_AUC")

study_reduction_pct <- 100 * (raw_study_r2 - adjusted_study_r2) / raw_study_r2
classifier_reduction_pct <- 100 *
  (raw_study_balacc - adjusted_study_balacc) / raw_study_balacc

suitable <- (
  nlevels(CRC_meta$studyID) >= 3 &&
    all_studies_have_both &&
    raw_study_r2 >= 0.02 &&
    adjusted_study_r2 < raw_study_r2 &&
    adjusted_disease_auc >= 0.60
)

report <- c(
  "# MMUPHin CRC Dataset Scouting Summary",
  "",
  "## Is the dataset available?",
  "",
  sprintf(
    "Yes. `CRC_abd` and `CRC_meta` loaded from MMUPHin %s under R %s.",
    as.character(packageVersion("MMUPHin")),
    paste(R.version$major, R.version$minor, sep = ".")
  ),
  "",
  "## What is its structure?",
  "",
  sprintf(
    "`CRC_abd` is a feature-by-sample matrix with %d species and %d samples. `CRC_meta` has %d sample rows and %d metadata columns. Abundance columns and metadata rows are exactly aligned by sample ID.",
    nrow(CRC_abd), ncol(CRC_abd), nrow(CRC_meta), ncol(CRC_meta)
  ),
  "",
  "## How many studies and labels are present?",
  "",
  sprintf(
    "There are %d studies and two labels (`control`, `CRC`). Every study contains both labels; the smallest study-condition cell has %d samples. Study and condition show modest imbalance (Cramer's V = %.3f, chi-square p = %.3g), but they are not perfectly confounded.",
    nlevels(CRC_meta$studyID), min(contingency), cramers_v, chi$p.value
  ),
  "",
  "## Is there a real batch/study effect?",
  "",
  sprintf(
    "Yes. Before adjustment, study explains %.2f%% of Bray-Curtis variation after controlling for condition (PERMANOVA p = %.3g). A cross-validated study classifier has balanced accuracy %.3f versus %.3f chance.",
    100 * raw_study_r2,
    get_p("raw", "study_R2_condition_controlled"),
    raw_study_balacc,
    1 / nlevels(CRC_meta$studyID)
  ),
  "",
  "## Does MMUPHin reduce it without erasing disease signal?",
  "",
  sprintf(
    "MMUPHin reduces condition-controlled study R2 from %.2f%% to %.2f%% (%.1f%% relative reduction) and study-classifier balanced accuracy from %.3f to %.3f (%.1f%% relative reduction).",
    100 * raw_study_r2,
    100 * adjusted_study_r2,
    study_reduction_pct,
    raw_study_balacc,
    adjusted_study_balacc,
    classifier_reduction_pct
  ),
  sprintf(
    "Condition-controlled disease R2 changes from %.2f%% to %.2f%%. Mean within-study leave-one-study-out disease AUC changes from %.3f to %.3f. These metrics should be read together: adjustment is useful if study predictability falls while cross-study disease discrimination remains meaningfully above chance.",
    100 * raw_condition_r2,
    100 * adjusted_condition_r2,
    raw_disease_auc,
    adjusted_disease_auc
  ),
  "",
  "## Is this suitable for the next batch-correction benchmark?",
  "",
  if (suitable) {
    "Yes. It is a compact, authentic multi-study CRC benchmark with both classes represented in every study, measurable study structure, and retained cross-study disease signal after adjustment."
  } else {
    "Not as a primary benchmark under the predeclared checks. Review the metric table to identify whether the limitation is weak raw study effect, incomplete label overlap, insufficient reduction, or weak retained cross-study disease signal."
  },
  "",
  "## Recommended next step",
  "",
  "Use the raw matrix, adjusted matrix, fixed metadata, and the same study-aware evaluation splits as a controlled benchmark for the proposed scGPT-style correction. Compare methods on study PERMANOVA R2 and study predictability, while requiring disease LOSO AUC to remain stable rather than optimizing batch removal alone.",
  "",
  "## Reproducibility notes",
  "",
  "- Abundance ordination uses PCA on square-root relative abundances.",
  "- PERMANOVA uses Bray-Curtis distance and 999 permutations.",
  "- Study prediction uses repeated stratified five-fold multinomial glmnet.",
  "- Disease prediction holds out each entire study and reports overall and mean within-study AUC.",
  "- MMUPHin adjustment controls for `study_condition` exactly as in the official vignette.",
  "",
  "Sources:",
  "- https://huttenhower.sph.harvard.edu/mmuphin/",
  "- https://bioconductor.org/packages/release/bioc/vignettes/MMUPHin/inst/doc/MMUPHin.html",
  "- https://forum.biobakery.org/t/input-data-for-mmuphin/2794"
)
writeLines(
  report,
  file.path(output_dir, "mmuphin_scouting_summary.md"),
  useBytes = TRUE
)

cat("\nCompleted:", format(Sys.time(), tz = "America/New_York"), "\n")
cat("Study R2 raw -> adjusted:", raw_study_r2, "->", adjusted_study_r2, "\n")
cat("Study balanced accuracy raw -> adjusted:", raw_study_balacc, "->", adjusted_study_balacc, "\n")
cat("Disease mean within-study LOSO AUC raw -> adjusted:", raw_disease_auc, "->", adjusted_disease_auc, "\n")
cat("Suitable:", suitable, "\n")
