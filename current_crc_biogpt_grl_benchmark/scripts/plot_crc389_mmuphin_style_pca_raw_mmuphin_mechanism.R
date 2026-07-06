source(file.path("..", "crc_benchmark_utils.R"))

root <- getwd()
data_dir <- file.path(root, "outputs", "crc_overlap_benchmark")
output_dir <- file.path(root, "outputs", "figures", "mechanism_grl_crc389")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_matrix <- function(path) {
  x <- utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  first <- names(x)[1]
  if (first == "sample_id") {
    rownames(x) <- x$sample_id
    x$sample_id <- NULL
    return(t(as.matrix(x)))
  }
  if (tolower(first) %in% c("feature", "taxon", "species")) {
    rownames(x) <- x[[first]]
    x[[first]] <- NULL
    return(as.matrix(x))
  }
  stop("Unrecognized matrix format: ", path)
}

metadata <- utils::read.csv(
  file.path(data_dir, "metadata_389.csv"),
  check.names = FALSE,
  stringsAsFactors = FALSE
)

matrices <- list(
  Raw = read_matrix(file.path(data_dir, "raw_abundance_389.csv")),
  MMUPHin = read_matrix(file.path(data_dir, "mmuphin_adjusted_abundance_389.csv")),
  `Mechanism-only` = read_matrix(file.path(data_dir, "mechanism_grl_abundance_389.csv"))
)

score_list <- lapply(names(matrices), function(name) {
  abundance <- matrices[[name]]
  abundance <- abundance[, metadata$sample_id, drop = FALSE]
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
  aes_args <- ggplot2::aes_string("PC1", "PC2", color = color_column)
  if (color_column == "study_condition") {
    aes_args <- ggplot2::aes_string(
      "PC1",
      "PC2",
      color = color_column,
      shape = color_column
    )
  }
  plot <- ggplot2::ggplot(scores, aes_args) +
    ggplot2::geom_point(size = 1.6, alpha = 0.75) +
    ggplot2::facet_wrap(~ method, scales = "free", nrow = 1) +
    ggplot2::labs(
      title = paste("CRC389 MMUPHin-style PCA colored by", color_column),
      subtitle = "Transform: sqrt(relative abundance) + PCA, matching controlled benchmark ordination",
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
  ggplot2::ggsave(out, plot, width = 12, height = 5.2, dpi = 180)
  out
}

study_path <- save_panel(
  "studyID",
  "crc389_mmuphin_style_raw_mmuphin_mechanism_pca_by_study_panel.png"
)
condition_path <- save_panel(
  "study_condition",
  "crc389_mmuphin_style_raw_mmuphin_mechanism_pca_by_condition_panel.png"
)
score_path <- file.path(
  output_dir,
  "crc389_mmuphin_style_raw_mmuphin_mechanism_pca_scores.csv"
)
utils::write.csv(scores, score_path, row.names = FALSE)

message("Wrote study PCA panel: ", study_path)
message("Wrote condition PCA panel: ", condition_path)
message("Wrote PCA scores: ", score_path)
