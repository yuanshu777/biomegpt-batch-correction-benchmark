source(file.path("..", "crc_benchmark_utils.R"))

root <- getwd()
full_root <- file.path("..", "crc_controlled_benchmark")
data_dir <- file.path(full_root, "data")
output_dir <- file.path(root, "outputs", "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_abundance <- function(path) {
  as.matrix(utils::read.csv(path, row.names = 1, check.names = FALSE))
}

metadata <- utils::read.csv(
  file.path(data_dir, "crc_metadata.csv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

matrices <- list(
  Raw = read_abundance(file.path(data_dir, "crc_raw_abundance.csv")),
  MMUPHin = read_abundance(file.path(data_dir, "crc_mmuphin_adjusted_abundance.csv"))
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

save_panel <- function(color_column, output_name) {
  if (color_column == "study_condition") {
    mapping <- ggplot2::aes(PC1, PC2, color = study_condition, shape = study_condition)
  } else {
    mapping <- ggplot2::aes(PC1, PC2, color = studyID)
  }
  plot <- ggplot2::ggplot(scores, mapping) +
    ggplot2::geom_point(size = 1.7, alpha = 0.75) +
    ggplot2::facet_wrap(~ method, scales = "free", nrow = 1) +
    ggplot2::labs(
      title = paste("Full551 MMUPHin-style PCA colored by", color_column),
      subtitle = "Transform: sqrt(relative abundance) + PCA",
      x = "PC1",
      y = "PC2"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text = ggplot2::element_text(size = 9)
    )
  if (color_column == "study_condition") {
    plot <- plot + ggplot2::scale_color_manual(
      values = c(control = "#0072B2", CRC = "#D55E00")
    )
  }
  out <- file.path(output_dir, output_name)
  ggplot2::ggsave(out, plot, width = 10, height = 5.2, dpi = 180)
  out
}

study_path <- save_panel("studyID", "full551_raw_mmuphin_pca_by_study.png")
condition_path <- save_panel(
  "study_condition",
  "full551_raw_mmuphin_pca_by_condition.png"
)
score_path <- file.path(output_dir, "full551_raw_mmuphin_pca_scores.csv")
utils::write.csv(scores, score_path, row.names = FALSE)

message("Wrote study PCA panel: ", study_path)
message("Wrote condition PCA panel: ", condition_path)
message("Wrote PCA scores: ", score_path)
