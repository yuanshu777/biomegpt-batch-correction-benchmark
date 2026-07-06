options(stringsAsFactors = FALSE)

source(file.path(getwd(), "crc_benchmark_utils.R"))
crc_require_packages()

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop("Missing required argument: ", flag)
  }
  args[index + 1]
}

method_name <- get_arg("--method")
matrix_path <- normalizePath(get_arg("--matrix"), mustWork = TRUE)
if (!grepl("^[A-Za-z0-9_.-]+$", method_name)) {
  stop("Method name may contain only letters, numbers, underscore, dot, and dash.")
}

benchmark_dir <- file.path(getwd(), "crc_controlled_benchmark")
metadata <- read.csv(
  file.path(benchmark_dir, "data", "crc_metadata.csv"),
  check.names = FALSE
)
raw_reference <- crc_read_abundance(
  file.path(benchmark_dir, "data", "crc_raw_abundance.rds")
)
method_abundance <- crc_read_abundance(matrix_path)
crc_validate_inputs(
  method_abundance,
  metadata,
  reference_features = rownames(raw_reference)
)

method_report_dir <- file.path(benchmark_dir, "reports", "methods", method_name)
dir.create(method_report_dir, recursive = TRUE, showWarnings = FALSE)
metrics <- crc_evaluate_method(
  method_abundance,
  metadata,
  method_name,
  file.path(benchmark_dir, "splits"),
  file.path(method_report_dir, "plots")
)
write.csv(
  metrics,
  file.path(method_report_dir, paste0(method_name, "_metrics.csv")),
  row.names = FALSE
)

baseline_metrics <- read.csv(
  file.path(benchmark_dir, "reports", "crc_raw_vs_mmuphin_metrics.csv")
)
combined <- rbind(baseline_metrics, metrics)
write.csv(
  combined,
  file.path(method_report_dir, paste0("raw_mmuphin_", method_name, "_metrics.csv")),
  row.names = FALSE
)

primary_metrics <- c(
  "study_R2_condition_controlled",
  "condition_R2_study_controlled",
  "study_prediction_balanced_accuracy",
  "disease_LOSO_mean_within_study_AUC",
  "disease_LOSO_balanced_accuracy"
)
comparison <- combined[combined$metric %in% primary_metrics, ]
comparison <- comparison[
  match(
    as.vector(outer(c("raw", "mmuphin", method_name), primary_metrics, paste)),
    paste(comparison$method, comparison$metric)
  ),
]
comparison <- comparison[!is.na(comparison$method), ]
write.csv(
  comparison,
  file.path(method_report_dir, paste0(method_name, "_primary_comparison.csv")),
  row.names = FALSE
)

report <- c(
  paste("# CRC Benchmark Method Report:", method_name),
  "",
  "This method was evaluated with the frozen CRC benchmark splits, classifier inner folds, and PERMANOVA permutations.",
  "",
  "| Method | Metric | Estimate | P value |",
  "|---|---|---:|---:|",
  vapply(seq_len(nrow(comparison)), function(i) {
    p_text <- ifelse(
      is.na(comparison$p_value[i]),
      "",
      sprintf("%.3g", comparison$p_value[i])
    )
    sprintf(
      "| %s | %s | %.4f | %s |",
      comparison$method[i],
      comparison$metric[i],
      comparison$estimate[i],
      p_text
    )
  }, character(1)),
  "",
  "PCA plots are stored in the adjacent `plots/` directory."
)
writeLines(
  report,
  file.path(method_report_dir, paste0(method_name, "_report.md"))
)

cat("Method evaluated:", method_name, "\n")
cat("Metrics:", file.path(method_report_dir, paste0(method_name, "_metrics.csv")), "\n")
