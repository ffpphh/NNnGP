#!/usr/bin/env Rscript
# ============================================================
# Gaussian NNMP baseline for a multi-order CSV and neighbor grid.
#
# The CSV may contain multiple reference orders distinguished by order_id.
# This script runs every selected order crossed with m = 4,6,...,20 by default.
# Each (order_id, m) combo is fitted independently, with up to N_WORKERS combos
# running in parallel. It saves only:
#   1) posterior sample CSVs for w_S, w_U, and y_U;
#   2) one combined summary.csv with y_U metrics.
#
# Expected CSV columns:
# order_id, order_type, row_order, point_index, split, reference_order,
# x, y, w, y_obs, x0, x1, x2
#
# Usage:
#   Rscript run_gnnmp_from_csv_by_order_m_grid_parallel.R \
#     sine_systematic_data.csv nnmp_sine_m_grid_results \
#     "4,6,8,10,12,14,16,18,20" 10000 5000 5 1000 9 ALL
#
# Arguments:
#   1 csv_file
#   2 out_dir
#   3 m_values, e.g. "4,6,8,10,12,14,16,18,20" or "4:20:2"
#   4 n_iter
#   5 n_burn
#   6 n_thin
#   7 n_report
#   8 n_workers, number of parallel (order_id, m) combos; default 9
#   9 order_ids, e.g. ALL or 1,2,3
# ============================================================

suppressPackageStartupMessages(library(parallel))

if (!requireNamespace("nnmp", quietly = TRUE)) {
  stop(
    paste0(
      "Package 'nnmp' is not installed.\n",
      "Install the locally patched C++14 version first, restart RStudio, then rerun this script."
    )
  )
}
suppressPackageStartupMessages(library(nnmp))

set.seed(20260610)
METHOD_NAME <- "NNMP"

args <- commandArgs(trailingOnly = TRUE)
script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
SCRIPT_DIR <- if (length(script_file_arg) > 0L) dirname(normalizePath(sub("^--file=", "", script_file_arg[1L]))) else getwd()
EXPERIMENT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
CSV_FILE <- if (length(args) >= 1L) args[[1]] else file.path(EXPERIMENT_DIR, "outputs", "no_split", "nnngp", "ordered_csv", "sine_systematic_data.csv")
OUT_DIR <- if (length(args) >= 2L) args[[2]] else file.path(EXPERIMENT_DIR, "outputs", "no_split", "nnmp")
M_VALUES_TEXT <- if (length(args) >= 3L) args[[3]] else "4,6,8,10,12,14,16,18,20"
N_ITER <- if (length(args) >= 4L) as.integer(args[[4]]) else 10000L
N_BURN <- if (length(args) >= 5L) as.integer(args[[5]]) else 5000L
N_THIN <- if (length(args) >= 6L) as.integer(args[[6]]) else 5L
N_REPORT <- if (length(args) >= 7L) as.integer(args[[7]]) else 1000L
N_WORKERS <- if (length(args) >= 8L) as.integer(args[[8]]) else 9L
ORDER_ID_TEXT <- if (length(args) >= 9L) args[[9]] else "ALL"

if (!file.exists(CSV_FILE)) stop("CSV not found: ", CSV_FILE, ". Run the Python systematic VI script first so it generates ordered_csv/sine_systematic_data.csv.")
if (N_ITER <= N_BURN) stop("N_ITER must be greater than N_BURN.")
if (N_THIN < 1L) stop("N_THIN must be >= 1.")
if (N_WORKERS < 1L) stop("N_WORKERS must be >= 1.")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "posterior_samples"), recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Helper functions
# ----------------------------
parse_m_values_arg <- function(text) {
  text <- gsub("\\s+", "", as.character(text))
  if (grepl(":", text, fixed = TRUE)) {
    parts <- as.integer(strsplit(text, ":", fixed = TRUE)[[1]])
    if (length(parts) == 2L) return(seq.int(parts[1], parts[2], by = 2L))
    if (length(parts) == 3L) return(seq.int(parts[1], parts[2], by = parts[3]))
    stop("m_values with ':' must look like '4:20' or '4:20:2'.")
  }
  vals <- as.integer(strsplit(text, ",", fixed = TRUE)[[1]])
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0L) stop("No valid m values were supplied.")
  sort(unique(vals))
}

parse_order_id_arg <- function(text, available_order_ids) {
  if (is.null(text) || length(text) == 0L || is.na(text) || !nzchar(text) || toupper(text) == "ALL") {
    return(available_order_ids)
  }
  ids <- as.integer(strsplit(text, ",", fixed = TRUE)[[1]])
  ids <- ids[!is.na(ids)]
  missing_ids <- setdiff(ids, available_order_ids)
  if (length(missing_ids) > 0L) {
    stop("Requested order_id not found in CSV: ", paste(missing_ids, collapse = ", "))
  }
  ids
}

rmse <- function(truth, estimate) {
  sqrt(mean((truth - estimate)^2, na.rm = TRUE))
}

rsr <- function(truth, estimate) {
  ok <- is.finite(truth) & is.finite(estimate)
  truth <- truth[ok]
  estimate <- estimate[ok]
  if (length(truth) < 2L || sd(truth) == 0) return(NA_real_)
  rmse(truth, estimate) / sd(truth)
}

row_sd <- function(sample_matrix) {
  apply(sample_matrix, 1, sd)
}

row_quantiles <- function(sample_matrix) {
  t(apply(sample_matrix, 1, quantile, probs = c(0.025, 0.5, 0.975), names = FALSE))
}

# Empirical ensemble CRPS using the sorted-sample identity.
crps_ensemble_rows <- function(truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  if (nrow(sample_matrix) != length(truth)) {
    stop("CRPS: truth length must equal the number of rows in sample_matrix.")
  }
  n_draws <- ncol(sample_matrix)
  if (n_draws < 2L) stop("CRPS requires at least two ensemble draws.")
  weights <- 2 * seq_len(n_draws) - n_draws - 1
  vapply(
    seq_len(nrow(sample_matrix)),
    function(i) {
      draws <- as.numeric(sample_matrix[i, ])
      first_term <- mean(abs(draws - truth[i]))
      sorted_draws <- sort(draws)
      second_term <- sum(weights * sorted_draws) / (n_draws^2)
      first_term - second_term
    },
    numeric(1)
  )
}

make_prediction_table <- function(method, variable, data_frame, truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  qq <- row_quantiles(sample_matrix)
  pred_mean <- rowMeans(sample_matrix)
  pred_sd <- row_sd(sample_matrix)
  crps <- crps_ensemble_rows(truth, sample_matrix)
  
  data.frame(
    method = method,
    variable = variable,
    split = as.character(data_frame$split),
    point_index = data_frame$point_index,
    true_value = as.numeric(truth),
    pred_mean = pred_mean,
    pred_sd = pred_sd,
    pred_q025 = qq[, 1],
    pred_median = qq[, 2],
    pred_q975 = qq[, 3],
    ci_width_95 = qq[, 3] - qq[, 1],
    crps = crps,
    stringsAsFactors = FALSE
  )
}

make_y_u_summary_row <- function(order_id, order_name, m, result_table) {
  data.frame(
    order_id = as.integer(order_id),
    order_name = as.character(order_name),
    m = as.integer(m),
    RMSPE = rmse(result_table$true_value, result_table$pred_mean),
    RSR = rsr(result_table$true_value, result_table$pred_mean),
    CRPS = mean(result_table$crps),
    CI_coverage_percent = 100 * mean(
      result_table$true_value >= result_table$pred_q025 &
        result_table$true_value <= result_table$pred_q975
    ),
    CI_width = mean(result_table$ci_width_95),
    stringsAsFactors = FALSE
  )
}

safe_write_csv <- function(x, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  write.csv(x, file, row.names = FALSE)
}

sanitize_filename <- function(x) {
  x <- as.character(x)
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  if (!nzchar(x)) x <- "order"
  x
}

write_posterior_samples <- function(sample_matrix, meta_df, file, order_id, order_name, m, variable) {
  sample_matrix <- as.matrix(sample_matrix)
  colnames(sample_matrix) <- sprintf("sample_%04d", seq_len(ncol(sample_matrix)))
  meta <- data.frame(
    order_id = as.integer(order_id),
    order_name = as.character(order_name),
    m = as.integer(m),
    variable = as.character(variable),
    split = as.character(meta_df$split),
    point_index = meta_df$point_index,
    row_order = meta_df$row_order,
    reference_order = if ("reference_order" %in% names(meta_df)) meta_df$reference_order else NA,
    x = meta_df$x,
    y = meta_df$y,
    stringsAsFactors = FALSE
  )
  safe_write_csv(cbind(meta, as.data.frame(sample_matrix, check.names = FALSE)), file)
}

read_csv_for_order <- function(csv_file, order_id) {
  dat <- read.csv(csv_file, stringsAsFactors = FALSE, check.names = FALSE)
  
  if (!("order_id" %in% names(dat))) {
    dat$order_id <- 1L
    dat$order_type <- "single_order"
  }
  if (!("order_type" %in% names(dat))) {
    dat$order_type <- paste0("order", dat$order_id)
  }
  if (!("reference_order" %in% names(dat))) {
    dat$reference_order <- NA
  }
  
  dat <- dat[as.integer(dat$order_id) == as.integer(order_id), , drop = FALSE]
  if (nrow(dat) == 0L) stop("No rows found for order_id=", order_id)
  
  required_cols <- c("row_order", "point_index", "split", "x", "y", "y_obs", "x0", "x1", "x2")
  missing_cols <- setdiff(required_cols, names(dat))
  if (length(missing_cols) > 0L) stop("Missing required CSV columns: ", paste(missing_cols, collapse = ", "))
  
  if (!("w" %in% names(dat))) dat$w <- NA_real_
  dat$w <- suppressWarnings(as.numeric(dat$w))
  dat$row_order <- as.integer(dat$row_order)
  dat$point_index <- as.integer(dat$point_index)
  dat$order_id <- as.integer(dat$order_id)
  
  if (!all(dat$split %in% c("S", "U"))) stop("split must contain only 'S' and 'U'.")
  if (anyDuplicated(dat$point_index) > 0L) stop("point_index must be unique within order_id=", order_id)
  dat
}

discover_order_ids <- function(csv_file) {
  dat <- read.csv(csv_file, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("order_id" %in% names(dat))) return(1L)
  sort(unique(as.integer(dat$order_id)))
}

run_single_combo <- function(job) {
  order_id <- as.integer(job$order_id)
  m <- as.integer(job$m)
  set.seed(20260610 + 1000L * order_id + m)
  
  dat <- read_csv_for_order(CSV_FILE, order_id)
  S <- dat[dat$split == "S", , drop = FALSE]
  U <- dat[dat$split == "U", , drop = FALSE]
  if (nrow(S) <= m) stop("The reference set must contain more rows than m for order_id=", order_id, ", m=", m)
  
  S <- S[order(S$row_order), , drop = FALSE]
  U <- U[order(U$point_index), , drop = FALSE]
  rownames(S) <- NULL
  rownames(U) <- NULL
  
  expected_S_order <- seq.int(0L, nrow(S) - 1L)
  if (!identical(as.integer(S$row_order), expected_S_order)) {
    stop("After sorting, S$row_order must equal 0, 1, ..., nrow(S)-1 for order_id=", order_id)
  }
  
  order_name <- unique(S$order_type)[1]
  safe_order <- sanitize_filename(order_name)
  prefix <- file.path(
    OUT_DIR,
    "posterior_samples",
    sprintf("order%02d_%s_m%02d", order_id, safe_order, m)
  )
  
  coords_S <- as.matrix(S[, c("x", "y")])
  coords_U <- as.matrix(U[, c("x", "y")])
  X_S <- as.matrix(S[, c("x0", "x1", "x2")])
  X_U <- as.matrix(U[, c("x0", "x1", "x2")])
  y_S <- as.numeric(S$y_obs)
  
  lm_start <- lm(y_obs ~ x1 + x2, data = S)
  beta_start <- as.numeric(coef(lm_start))
  
  priors <- list(
    sigmasq_invgamma = c(2, 1),
    tausq_invgamma = c(2, 0.1),
    phi_invgamma = c(3, 1 / 3),
    zeta_invgamma = c(3, 0.2),
    ga_gaus = list(mean_vec = matrix(c(-1.5, 0, 0), ncol = 1), var_mat = 2 * diag(3)),
    kasq_invgamma = c(3, 1)
  )
  starting <- list(
    bb = beta_start,
    ga = matrix(c(-1.5, 0, 0), ncol = 1),
    kasq = 1,
    phi = 0.20,
    zeta = 0.10
  )
  tuning <- list(phi = 0.08, zeta = 0.08)
  mcmc_settings <- list(n_iter = N_ITER, n_burn = N_BURN, n_thin = N_THIN, n_report = N_REPORT)
  
  cat(sprintf("\n[NNMP order %02d/%s, m=%02d] S=%d, U=%d\n", order_id, order_name, m, nrow(S), nrow(U)))
  
  fit_gnnmp <- nnmp(
    response = y_S,
    covars = X_S,
    coords = coords_S,
    neighbor_size = m,
    marg_family = "gaussian",
    priors = priors,
    starting = starting,
    tuning = tuning,
    ord = seq_len(nrow(S)),
    mcmc_settings = mcmc_settings,
    verbose = FALSE,
    neighbor_info = TRUE,
    model_diag = list("dic", "pplc")
  )
  
  pred_U <- predict(
    fit_gnnmp,
    nonref_covars = X_U,
    nonref_coords = coords_U,
    probs = c(0.025, 0.5, 0.975),
    predict_sam = TRUE,
    verbose = FALSE,
    nreport = 500
  )
  
  w_samples_S <- as.matrix(fit_gnnmp$post_samples$zz)
  w_samples_U <- as.matrix(pred_U$zz_sam)
  y_samples_U <- as.matrix(pred_U$obs_sam)
  if (nrow(w_samples_S) != nrow(S)) stop("Unexpected dimension for w_S samples for order_id=", order_id, ", m=", m)
  if (nrow(w_samples_U) != nrow(U) || nrow(y_samples_U) != nrow(U)) {
    stop("Unexpected dimensions in U prediction samples for order_id=", order_id, ", m=", m)
  }
  
  write_posterior_samples(w_samples_S, S, paste0(prefix, "_w_S_samples.csv"), order_id, order_name, m, "w_S")
  write_posterior_samples(w_samples_U, U, paste0(prefix, "_w_U_samples.csv"), order_id, order_name, m, "w_U")
  write_posterior_samples(y_samples_U, U, paste0(prefix, "_y_U_samples.csv"), order_id, order_name, m, "y_U")
  
  pred_y_U <- make_prediction_table(METHOD_NAME, "y", U, U$y_obs, y_samples_U)
  summary_row <- make_y_u_summary_row(order_id, order_name, m, pred_y_U)
  cat(sprintf("[NNMP order %02d, m=%02d] done. RMSPE=%.4f, RSR=%.4f, CRPS=%.4f\n", order_id, m, summary_row$RMSPE, summary_row$RSR, summary_row$CRPS))
  summary_row
}

run_jobs <- function(jobs) {
  job_list <- lapply(seq_len(nrow(jobs)), function(i) list(order_id = jobs$order_id[i], m = jobs$m[i]))
  n_workers <- min(as.integer(N_WORKERS), length(job_list))
  cat("Total combos:", length(job_list), "\n")
  cat("Parallel combo workers:", n_workers, "\n")
  
  if (n_workers <= 1L) return(lapply(job_list, run_single_combo))
  
  cl <- parallel::makeCluster(n_workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterSetRNGStream(cl, 20260610)
  parallel::clusterEvalQ(cl, { suppressPackageStartupMessages(library(nnmp)); NULL })
  parallel::clusterExport(cl, varlist = setdiff(ls(envir = .GlobalEnv), c("cl")), envir = .GlobalEnv)
  parallel::parLapply(cl, job_list, run_single_combo)
}

available_order_ids <- discover_order_ids(CSV_FILE)
#order_ids <- parse_order_id_arg(ORDER_ID_TEXT, available_order_ids)
order_ids <- c(1,2,3,4,5)
m_values <- parse_m_values_arg(M_VALUES_TEXT)

jobs <- expand.grid(order_id = order_ids, m = m_values, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
jobs <- jobs[order(jobs$order_id, jobs$m), , drop = FALSE]

cat("Orders to run:", paste(order_ids, collapse = ", "), "\n")
cat("m values:", paste(m_values, collapse = ", "), "\n")

summary_list <- run_jobs(jobs)
summary_all <- do.call(rbind, summary_list)
summary_all <- summary_all[order(summary_all$order_id, summary_all$m), , drop = FALSE]
rownames(summary_all) <- NULL
safe_write_csv(summary_all, file.path(OUT_DIR, "summary.csv"))

cat("\nDone. Combined summary written to:\n")
cat(normalizePath(file.path(OUT_DIR, "summary.csv")), "\n")
cat("Posterior sample CSVs written under:\n")
cat(normalizePath(file.path(OUT_DIR, "posterior_samples")), "\n")
