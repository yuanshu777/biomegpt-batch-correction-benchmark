options(stringsAsFactors = FALSE)

source(file.path(getwd(), "crc_benchmark_utils.R"))
crc_require_packages()

benchmark_dir <- file.path(getwd(), "crc_controlled_benchmark")
data_dir <- file.path(benchmark_dir, "data")
split_dir <- file.path(benchmark_dir, "splits")
report_dir <- file.path(benchmark_dir, "reports")

metadata <- read.csv(
  file.path(data_dir, "crc_metadata.csv"),
  check.names = FALSE
)
raw_rds <- readRDS(file.path(data_dir, "crc_raw_abundance.rds"))
raw_csv <- crc_read_abundance(file.path(data_dir, "crc_raw_abundance.csv"))
mmuphin_rds <- readRDS(
  file.path(data_dir, "crc_mmuphin_adjusted_abundance.rds")
)
mmuphin_csv <- crc_read_abundance(
  file.path(data_dir, "crc_mmuphin_adjusted_abundance.csv")
)

stopifnot(
  identical(dim(raw_rds), c(484L, 551L)),
  identical(dim(mmuphin_rds), dim(raw_rds)),
  identical(colnames(raw_rds), metadata$sample_id),
  identical(rownames(mmuphin_rds), rownames(raw_rds)),
  identical(colnames(mmuphin_rds), colnames(raw_rds)),
  isTRUE(all.equal(raw_csv, raw_rds, check.attributes = TRUE)),
  isTRUE(all.equal(mmuphin_csv, mmuphin_rds, check.attributes = TRUE))
)

outer <- read.csv(file.path(split_dir, "study_prediction_outer_folds.csv"))
stopifnot(
  nrow(outer) == 3 * nrow(metadata),
  all(table(outer$repeat_id, outer$sample_id) == 1),
  identical(sort(unique(outer$repeat_id)), 1:3),
  identical(sort(unique(outer$outer_fold)), 1:5)
)
outer_with_labels <- merge(outer, metadata, by = "sample_id", sort = FALSE)
outer_balance <- aggregate(
  sample_id ~ repeat_id + outer_fold + studyID,
  data = outer_with_labels,
  FUN = length
)
for (repeat_id in 1:3) {
  repeat_counts <- outer_balance[outer_balance$repeat_id == repeat_id, ]
  spread <- aggregate(
    sample_id ~ studyID,
    data = repeat_counts,
    FUN = function(x) max(x) - min(x)
  )
  stopifnot(all(spread$sample_id <= 1))
}

study_inner <- read.csv(
  gzfile(file.path(split_dir, "study_prediction_inner_folds.csv.gz"))
)
for (repeat_id in 1:3) {
  for (outer_fold in 1:5) {
    train_ids <- outer$sample_id[
      outer$repeat_id == repeat_id & outer$outer_fold != outer_fold
    ]
    rows <- study_inner[
      study_inner$repeat_id == repeat_id &
        study_inner$outer_fold == outer_fold,
    ]
    stopifnot(
      setequal(rows$sample_id, train_ids),
      all(rows$inner_fold %in% 1:5)
    )
  }
}

loso <- read.csv(file.path(split_dir, "disease_loso_splits.csv"))
stopifnot(
  nrow(loso) == length(unique(metadata$studyID)) * nrow(metadata),
  all(table(loso$held_out_study, loso$sample_id) == 1)
)
for (study_name in unique(metadata$studyID)) {
  rows <- loso[loso$held_out_study == study_name, ]
  expected_test <- metadata$sample_id[metadata$studyID == study_name]
  stopifnot(
    setequal(rows$sample_id[rows$role == "test"], expected_test),
    setequal(rows$sample_id[rows$role == "train"],
             setdiff(metadata$sample_id, expected_test))
  )
}

disease_inner <- read.csv(
  file.path(split_dir, "disease_loso_inner_folds.csv")
)
for (study_name in unique(metadata$studyID)) {
  train_ids <- metadata$sample_id[metadata$studyID != study_name]
  rows <- disease_inner[disease_inner$held_out_study == study_name, ]
  stopifnot(
    setequal(rows$sample_id, train_ids),
    all(rows$inner_fold %in% 1:5)
  )
}

permutations <- readRDS(
  file.path(split_dir, "permanova_permutation_matrix.rds")
)
stopifnot(
  identical(dim(permutations), c(999L, 551L)),
  all(apply(
    permutations,
    1,
    function(permutation) identical(sort(permutation), seq_len(551))
  ))
)

expected <- read.csv(file.path(report_dir, "crc_raw_vs_mmuphin_metrics.csv"))
raw_observed <- crc_evaluate_method(raw_rds, metadata, "raw", split_dir)
mmuphin_observed <- crc_evaluate_method(
  mmuphin_rds,
  metadata,
  "mmuphin",
  split_dir
)
observed <- rbind(raw_observed, mmuphin_observed)
stopifnot(
  identical(observed$method, expected$method),
  identical(observed$metric, expected$metric),
  max(abs(observed$estimate - expected$estimate), na.rm = TRUE) < 1e-12,
  all(
    (is.na(observed$p_value) & is.na(expected$p_value)) |
      observed$p_value == expected$p_value
  )
)

required_files <- c(
  file.path(report_dir, "crc_raw_vs_mmuphin_report.html"),
  file.path(report_dir, "crc_raw_vs_mmuphin_report.md"),
  file.path(benchmark_dir, "methods", "scgpt_biomegpt", "README.md"),
  file.path(benchmark_dir, "methods", "scgpt_biomegpt", "output_schema.csv"),
  file.path(benchmark_dir, "manifests", "artifact_manifest.csv")
)
stopifnot(all(file.exists(required_files)), all(file.info(required_files)$size > 0))

cat("CRC_BENCHMARK_VALIDATION_OK\n")
cat("samples=551 features=484 studies=5\n")
cat("study_outer_assignments=", nrow(outer), "\n", sep = "")
cat("study_inner_assignments=", nrow(study_inner), "\n", sep = "")
cat("disease_loso_assignments=", nrow(loso), "\n", sep = "")
cat("permutations=", nrow(permutations), "\n", sep = "")
