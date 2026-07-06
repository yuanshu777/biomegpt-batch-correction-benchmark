source("crc_benchmark_utils.R")

method_id <- "grl_mech_context_only_l8_lam10_rw5_rel1_var1"
benchmark_dir <- "crc_controlled_benchmark"
data_dir <- file.path(benchmark_dir, "data")
method_dir <- file.path(benchmark_dir, "methods", "scgpt_biomegpt")
output_dir <- file.path(
  benchmark_dir,
  "reports",
  "methods",
  method_id,
  "plots"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_abundance_csv <- function(path) {
  as.matrix(utils::read.csv(path, row.names = 1, check.names = FALSE))
}

metadata <- utils::read.csv(
  file.path(data_dir, "crc_metadata.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

matrices <- list(
  Raw = read_abundance_csv(file.path(data_dir, "crc_raw_abundance.csv")),
  MMUPHin = read_abundance_csv(
    file.path(data_dir, "crc_mmuphin_adjusted_abundance.csv")
  ),
  `Mechanism-only best` = read_abundance_csv(
    file.path(method_dir, paste0(method_id, ".csv"))
  )
)

score_list <- lapply(names(matrices), function(name) {
  abundance <- matrices[[name]]
  crc_validate_inputs(
    abundance,
    metadata,
    reference_features = rownames(matrices$Raw)
  )
  pca <- crc_make_pca(abundance, metadata)
  pca$scores$method <- sprintf(
    "%s\nPC1 %.1f%%, PC2 %.1f%%",
    name,
    100 * pca$variance[1],
    100 * pca$variance[2]
  )
  pca$scores
})
scores <- do.call(rbind, score_list)
scores$method <- factor(scores$method, levels = unique(scores$method))

study_plot <- ggplot2::ggplot(
  scores,
  ggplot2::aes(PC1, PC2, color = studyID)
) +
  ggplot2::geom_point(size = 1.55, alpha = 0.75) +
  ggplot2::facet_wrap(~ method, scales = "free", nrow = 1) +
  ggplot2::labs(
    title = "CRC abundance PCA using MMUPHin benchmark transform",
    subtitle = "Colored by study; each panel is PCA on sqrt relative abundance for that method",
    x = "PC1",
    y = "PC2",
    color = "Study"
  ) +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(
    legend.position = "bottom",
    strip.text = ggplot2::element_text(size = 9)
  )

condition_plot <- ggplot2::ggplot(
  scores,
  ggplot2::aes(PC1, PC2, color = study_condition, shape = study_condition)
) +
  ggplot2::geom_point(size = 1.55, alpha = 0.75) +
  ggplot2::facet_wrap(~ method, scales = "free", nrow = 1) +
  ggplot2::scale_color_manual(values = c(control = "#0072B2", CRC = "#D55E00")) +
  ggplot2::labs(
    title = "CRC abundance PCA using MMUPHin benchmark transform",
    subtitle = "Colored by CRC/control condition; each panel is PCA on sqrt relative abundance for that method",
    x = "PC1",
    y = "PC2",
    color = "Condition",
    shape = "Condition"
  ) +
  ggplot2::theme_bw(base_size = 11) +
  ggplot2::theme(
    legend.position = "bottom",
    strip.text = ggplot2::element_text(size = 9)
  )

study_path <- file.path(output_dir, "raw_mmuphin_mechanism_best_pca_by_study_panel.png")
condition_path <- file.path(
  output_dir,
  "raw_mmuphin_mechanism_best_pca_by_condition_panel.png"
)
score_path <- file.path(output_dir, "raw_mmuphin_mechanism_best_pca_scores.csv")

ggplot2::ggsave(study_path, study_plot, width = 12, height = 5.2, dpi = 180)
ggplot2::ggsave(condition_path, condition_plot, width = 12, height = 5.2, dpi = 180)
utils::write.csv(scores, score_path, row.names = FALSE)

message("Wrote study PCA panel: ", study_path)
message("Wrote condition PCA panel: ", condition_path)
message("Wrote PCA scores: ", score_path)
