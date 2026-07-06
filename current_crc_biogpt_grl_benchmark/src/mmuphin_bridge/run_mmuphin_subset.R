args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript run_mmuphin_subset.R <raw_abundance_389.csv> <metadata_389.csv> <output_csv>")
}

raw_path <- args[[1]]
metadata_path <- args[[2]]
output_path <- args[[3]]

suppressPackageStartupMessages({
  library(MMUPHin)
})

raw <- utils::read.csv(raw_path, check.names = FALSE)
metadata <- utils::read.csv(metadata_path, check.names = FALSE)

if (!"sample_id" %in% names(raw)) {
  stop("raw abundance must be sample x feature with a sample_id column")
}
rownames(raw) <- raw$sample_id
raw$sample_id <- NULL
feature_abd <- t(as.matrix(raw))

required <- c("sample_id", "studyID", "study_condition")
missing <- setdiff(required, names(metadata))
if (length(missing) > 0) {
  stop(paste("metadata missing columns:", paste(missing, collapse = ", ")))
}
metadata <- metadata[match(colnames(feature_abd), metadata$sample_id), , drop = FALSE]
rownames(metadata) <- metadata$sample_id

fit <- adjust_batch(
  feature_abd = feature_abd,
  batch = "studyID",
  covariates = "study_condition",
  data = metadata,
  control = list(verbose = FALSE)
)

adjusted <- as.data.frame(t(fit$feature_abd_adj), check.names = FALSE)
adjusted <- cbind(sample_id = rownames(adjusted), adjusted)
utils::write.csv(adjusted, output_path, row.names = FALSE)

