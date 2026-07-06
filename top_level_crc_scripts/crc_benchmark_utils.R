options(stringsAsFactors = FALSE)

crc_require_packages <- function() {
  packages <- c("ggplot2", "glmnet", "pROC", "vegan", "permute")
  missing <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0) {
    stop("Missing required packages: ", paste(missing, collapse = ", "))
  }
}

crc_read_abundance <- function(path) {
  if (!file.exists(path)) {
    stop("Abundance file does not exist: ", path)
  }
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    abundance <- readRDS(path)
  } else {
    exported <- read.csv(path, check.names = FALSE)
    if (!"feature" %in% names(exported)) {
      stop("CSV abundance matrix must have a first column named 'feature': ", path)
    }
    features <- exported$feature
    abundance <- as.matrix(exported[, setdiff(names(exported), "feature"), drop = FALSE])
    storage.mode(abundance) <- "double"
    rownames(abundance) <- features
  }
  if (!is.matrix(abundance) || is.null(rownames(abundance)) ||
      is.null(colnames(abundance))) {
    stop("Abundance data must be a named feature-by-sample matrix: ", path)
  }
  if (anyNA(abundance) || any(!is.finite(abundance))) {
    stop("Abundance matrix contains missing or non-finite values: ", path)
  }
  abundance
}

crc_write_abundance <- function(abundance, csv_path, rds_path = NULL) {
  exported <- data.frame(
    feature = rownames(abundance),
    abundance,
    check.names = FALSE
  )
  write.csv(exported, csv_path, row.names = FALSE)
  if (!is.null(rds_path)) {
    saveRDS(abundance, rds_path)
  }
}

crc_validate_inputs <- function(abundance, metadata, reference_features = NULL) {
  required_metadata <- c("sample_id", "studyID", "study_condition")
  if (!all(required_metadata %in% names(metadata))) {
    stop(
      "Metadata is missing required columns: ",
      paste(setdiff(required_metadata, names(metadata)), collapse = ", ")
    )
  }
  if (anyDuplicated(metadata$sample_id)) {
    stop("Metadata sample_id values are not unique.")
  }
  if (anyDuplicated(rownames(abundance))) {
    stop("Abundance feature identifiers are not unique.")
  }
  if (anyDuplicated(colnames(abundance))) {
    stop("Abundance sample identifiers are not unique.")
  }
  if (!identical(colnames(abundance), metadata$sample_id)) {
    stop("Abundance columns must exactly match metadata sample_id order.")
  }
  if (!is.null(reference_features) &&
      !identical(rownames(abundance), reference_features)) {
    stop("Abundance features must exactly match the benchmark feature order.")
  }
  invisible(TRUE)
}

crc_make_pca <- function(abundance, metadata) {
  sample_sums <- colSums(pmax(abundance, 0))
  if (any(sample_sums <= 0)) {
    stop("At least one sample has a non-positive abundance sum.")
  }
  relative <- sweep(pmax(abundance, 0), 2, sample_sums, "/")
  transformed <- sqrt(relative)
  keep <- apply(transformed, 1, stats::var) > 0
  fit <- prcomp(t(transformed[keep, , drop = FALSE]), center = TRUE)
  variance <- fit$sdev^2 / sum(fit$sdev^2)
  scores <- data.frame(
    sample_id = rownames(fit$x),
    PC1 = fit$x[, 1],
    PC2 = fit$x[, 2],
    studyID = metadata$studyID[match(rownames(fit$x), metadata$sample_id)],
    study_condition = metadata$study_condition[
      match(rownames(fit$x), metadata$sample_id)
    ],
    check.names = FALSE
  )
  list(scores = scores, variance = variance)
}

crc_save_pca_plots <- function(abundance, metadata, method_name, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  pca <- crc_make_pca(abundance, metadata)
  x_label <- sprintf("PC1 (%.1f%%)", 100 * pca$variance[1])
  y_label <- sprintf("PC2 (%.1f%%)", 100 * pca$variance[2])

  study_plot <- ggplot2::ggplot(
    pca$scores,
    ggplot2::aes(PC1, PC2, color = studyID)
  ) +
    ggplot2::geom_point(size = 1.8, alpha = 0.75) +
    ggplot2::labs(
      title = paste("CRC abundance PCA:", method_name),
      subtitle = "Colored by study",
      x = x_label,
      y = y_label,
      color = "Study"
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(legend.position = "right")

  condition_plot <- ggplot2::ggplot(
    pca$scores,
    ggplot2::aes(
      PC1,
      PC2,
      color = study_condition,
      shape = study_condition
    )
  ) +
    ggplot2::geom_point(size = 1.8, alpha = 0.75) +
    ggplot2::scale_color_manual(
      values = c(control = "#0072B2", CRC = "#D55E00")
    ) +
    ggplot2::labs(
      title = paste("CRC abundance PCA:", method_name),
      subtitle = "Colored by disease condition",
      x = x_label,
      y = y_label,
      color = "Condition",
      shape = "Condition"
    ) +
    ggplot2::theme_bw(base_size = 11)

  study_path <- file.path(output_dir, paste0(method_name, "_pca_by_study.png"))
  condition_path <- file.path(
    output_dir,
    paste0(method_name, "_pca_by_condition.png")
  )
  ggplot2::ggsave(study_path, study_plot, width = 9, height = 6, dpi = 180)
  ggplot2::ggsave(
    condition_path,
    condition_plot,
    width = 8,
    height = 6,
    dpi = 180
  )
  c(study = study_path, condition = condition_path)
}

crc_metric_row <- function(method, family, metric, estimate,
                           p_value = NA_real_, detail = "") {
  data.frame(
    method = method,
    metric_family = family,
    metric = metric,
    estimate = as.numeric(estimate),
    p_value = as.numeric(p_value),
    detail = detail,
    stringsAsFactors = FALSE
  )
}

crc_balanced_accuracy <- function(observed, predicted) {
  classes <- levels(observed)
  recalls <- vapply(classes, function(class_name) {
    idx <- observed == class_name
    mean(predicted[idx] == class_name)
  }, numeric(1))
  mean(recalls)
}

crc_model_matrix <- function(abundance) {
  log1p(1000 * t(pmax(abundance, 0)))
}

crc_evaluate_permanova <- function(abundance, metadata, method_name,
                                   permutation_matrix) {
  distance <- vegan::vegdist(t(abundance), method = "bray")
  model_data <- data.frame(
    studyID = factor(metadata$studyID),
    study_condition = factor(
      metadata$study_condition,
      levels = c("control", "CRC")
    )
  )
  study_fit <- vegan::adonis2(
    distance ~ studyID,
    data = model_data,
    permutations = permutation_matrix
  )
  condition_fit <- vegan::adonis2(
    distance ~ study_condition,
    data = model_data,
    permutations = permutation_matrix
  )
  joint_fit <- vegan::adonis2(
    distance ~ studyID + study_condition,
    data = model_data,
    permutations = permutation_matrix,
    by = "margin"
  )

  rbind(
    crc_metric_row(
      method_name, "PERMANOVA", "study_R2_unadjusted_model",
      study_fit[1, "R2"], study_fit[1, "Pr(>F)"],
      "Bray-Curtis; studyID-only model; fixed 999 permutations"
    ),
    crc_metric_row(
      method_name, "PERMANOVA", "condition_R2_unadjusted_model",
      condition_fit[1, "R2"], condition_fit[1, "Pr(>F)"],
      "Bray-Curtis; condition-only model; fixed 999 permutations"
    ),
    crc_metric_row(
      method_name, "PERMANOVA", "study_R2_condition_controlled",
      joint_fit["studyID", "R2"], joint_fit["studyID", "Pr(>F)"],
      "Bray-Curtis; marginal term in studyID + study_condition"
    ),
    crc_metric_row(
      method_name, "PERMANOVA", "condition_R2_study_controlled",
      joint_fit["study_condition", "R2"],
      joint_fit["study_condition", "Pr(>F)"],
      "Bray-Curtis; marginal term in studyID + study_condition"
    )
  )
}

crc_evaluate_study_prediction <- function(
    abundance,
    metadata,
    method_name,
    outer_folds,
    inner_folds) {
  x <- crc_model_matrix(abundance)
  rownames(x) <- metadata$sample_id
  y <- factor(metadata$studyID)
  names(y) <- metadata$sample_id
  repeats <- sort(unique(outer_folds$repeat_id))
  accuracies <- numeric(length(repeats))
  balanced <- numeric(length(repeats))

  for (repeat_id in repeats) {
    repeat_outer <- outer_folds[outer_folds$repeat_id == repeat_id, ]
    predictions <- factor(rep(NA_character_, length(y)), levels = levels(y))
    names(predictions) <- names(y)

    for (outer_fold in sort(unique(repeat_outer$outer_fold))) {
      test_ids <- repeat_outer$sample_id[
        repeat_outer$outer_fold == outer_fold
      ]
      train_ids <- setdiff(metadata$sample_id, test_ids)
      fold_rows <- inner_folds[
        inner_folds$repeat_id == repeat_id &
          inner_folds$outer_fold == outer_fold,
      ]
      fold_id <- fold_rows$inner_fold[
        match(train_ids, fold_rows$sample_id)
      ]
      if (anyNA(fold_id)) {
        stop("Missing study-prediction inner folds.")
      }
      fit <- glmnet::cv.glmnet(
        x[train_ids, , drop = FALSE],
        y[train_ids],
        family = "multinomial",
        type.measure = "class",
        foldid = fold_id,
        standardize = TRUE,
        parallel = FALSE
      )
      pred <- predict(
        fit,
        newx = x[test_ids, , drop = FALSE],
        s = "lambda.1se",
        type = "class"
      )
      predictions[test_ids] <- as.character(pred[, 1])
    }
    accuracies[repeat_id] <- mean(predictions == y)
    balanced[repeat_id] <- crc_balanced_accuracy(y, predictions)
  }

  rbind(
    crc_metric_row(
      method_name, "classifier", "study_prediction_accuracy",
      mean(accuracies), NA,
      sprintf("%d fixed repeats of stratified 5-fold CV", length(repeats))
    ),
    crc_metric_row(
      method_name, "classifier", "study_prediction_balanced_accuracy",
      mean(balanced), NA,
      sprintf(
        "%d fixed repeats; multinomial glmnet; chance %.3f",
        length(repeats),
        1 / nlevels(y)
      )
    )
  )
}

crc_evaluate_disease_loso <- function(
    abundance,
    metadata,
    method_name,
    loso_splits,
    inner_folds) {
  x <- crc_model_matrix(abundance)
  rownames(x) <- metadata$sample_id
  y <- as.integer(metadata$study_condition == "CRC")
  names(y) <- metadata$sample_id
  studies <- unique(loso_splits$held_out_study)
  probabilities <- rep(NA_real_, length(y))
  names(probabilities) <- names(y)
  per_study_auc <- numeric(0)

  for (study_name in studies) {
    split_rows <- loso_splits[loso_splits$held_out_study == study_name, ]
    train_ids <- split_rows$sample_id[split_rows$role == "train"]
    test_ids <- split_rows$sample_id[split_rows$role == "test"]
    fold_rows <- inner_folds[inner_folds$held_out_study == study_name, ]
    fold_id <- fold_rows$inner_fold[match(train_ids, fold_rows$sample_id)]
    if (anyNA(fold_id)) {
      stop("Missing disease-LOSO inner folds.")
    }
    fit <- glmnet::cv.glmnet(
      x[train_ids, , drop = FALSE],
      y[train_ids],
      family = "binomial",
      type.measure = "auc",
      foldid = fold_id,
      standardize = TRUE,
      parallel = FALSE
    )
    probabilities[test_ids] <- as.numeric(
      predict(
        fit,
        newx = x[test_ids, , drop = FALSE],
        s = "lambda.1se",
        type = "response"
      )
    )
    per_study_auc[study_name] <- as.numeric(
      pROC::auc(y[test_ids], probabilities[test_ids], quiet = TRUE)
    )
  }

  observed <- factor(
    ifelse(y == 1, "CRC", "control"),
    levels = c("control", "CRC")
  )
  predicted <- factor(
    ifelse(probabilities >= 0.5, "CRC", "control"),
    levels = c("control", "CRC")
  )

  rbind(
    crc_metric_row(
      method_name, "classifier", "disease_LOSO_overall_AUC",
      as.numeric(pROC::auc(y, probabilities, quiet = TRUE)), NA,
      "Each complete study held out once; fixed inner folds"
    ),
    crc_metric_row(
      method_name, "classifier", "disease_LOSO_mean_within_study_AUC",
      mean(per_study_auc), NA,
      paste(
        paste(names(per_study_auc), sprintf("%.3f", per_study_auc), sep = "="),
        collapse = "; "
      )
    ),
    crc_metric_row(
      method_name, "classifier", "disease_LOSO_balanced_accuracy",
      crc_balanced_accuracy(observed, predicted), NA,
      "Threshold 0.5 on fixed leave-one-study-out probabilities"
    )
  )
}

crc_evaluate_method <- function(abundance, metadata, method_name,
                                split_dir, report_plot_dir = NULL) {
  outer_folds <- read.csv(
    file.path(split_dir, "study_prediction_outer_folds.csv")
  )
  inner_study <- read.csv(
    gzfile(file.path(split_dir, "study_prediction_inner_folds.csv.gz"))
  )
  loso_splits <- read.csv(
    file.path(split_dir, "disease_loso_splits.csv")
  )
  inner_disease <- read.csv(
    file.path(split_dir, "disease_loso_inner_folds.csv")
  )
  permutation_matrix <- readRDS(
    file.path(split_dir, "permanova_permutation_matrix.rds")
  )

  metrics <- rbind(
    crc_evaluate_permanova(
      abundance,
      metadata,
      method_name,
      permutation_matrix
    ),
    crc_evaluate_study_prediction(
      abundance,
      metadata,
      method_name,
      outer_folds,
      inner_study
    ),
    crc_evaluate_disease_loso(
      abundance,
      metadata,
      method_name,
      loso_splits,
      inner_disease
    )
  )

  if (!is.null(report_plot_dir)) {
    crc_save_pca_plots(abundance, metadata, method_name, report_plot_dir)
  }
  metrics
}
