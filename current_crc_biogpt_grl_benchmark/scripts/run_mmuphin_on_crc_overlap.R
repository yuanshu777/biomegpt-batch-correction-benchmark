cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)]
if (length(file_arg) == 0) {
  root <- normalizePath(getwd())
} else {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[[1]])))
  root <- normalizePath(file.path(script_dir, ".."))
}

source_script <- file.path(root, "src", "mmuphin_bridge", "run_mmuphin_subset.R")
raw_path <- file.path(root, "outputs", "crc_overlap_benchmark", "raw_abundance_389.csv")
metadata_path <- file.path(root, "outputs", "crc_overlap_benchmark", "metadata_389.csv")
output_path <- file.path(root, "outputs", "crc_overlap_benchmark", "mmuphin_adjusted_abundance_389_rerun.csv")

rscript <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript)) {
  rscript <- file.path(R.home("bin"), "Rscript")
}
status <- system2(rscript, c(source_script, raw_path, metadata_path, output_path))
if (!identical(status, 0L)) {
  stop("MMUPHin overlap rerun failed with status: ", status)
}
cat("MMUPHin overlap adjusted matrix written to:", output_path, "\n")
