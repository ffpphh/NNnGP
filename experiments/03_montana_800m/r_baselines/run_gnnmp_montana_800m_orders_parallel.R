#!/usr/bin/env Rscript
# ============================================================
# Montana 800m split-seed Gaussian NNMP baseline.
# Reads original Montana data plus a 10-row point-order table.
# For each seed/order row, first TRAIN_SIZE rows are S and the rest are U.
# ============================================================

rm(list = ls())
set.seed(20260610)
if (!requireNamespace("nnmp", quietly = TRUE)) {
  stop("Package 'nnmp' is not installed. Install the developer/local nnmp package first.")
}
suppressPackageStartupMessages(library(nnmp))

args <- commandArgs(trailingOnly = TRUE)
ORIGINAL_CSV <- if (length(args) >= 1L) args[[1]] else NA_character_
ORDER_CSV <- if (length(args) >= 2L) args[[2]] else NA_character_
OUT_DIR <- if (length(args) >= 3L) args[[3]] else NA_character_
TRAIN_SIZE <- if (length(args) >= 4L) as.integer(args[[4]]) else 800L
NEIGHBOR_SIZE <- if (length(args) >= 5L) as.integer(args[[5]]) else 10L
N_ITER <- if (length(args) >= 6L) as.integer(args[[6]]) else 10000L
N_BURN <- if (length(args) >= 7L) as.integer(args[[7]]) else 5000L
N_THIN <- if (length(args) >= 8L) as.integer(args[[8]]) else 5L
N_REPORT <- if (length(args) >= 9L) as.integer(args[[9]]) else 1000L
SEEDS_ARG <- if (length(args) >= 10L) args[[10]] else "all"
SAVE_SAMPLE_CSV <- if (length(args) >= 11L) as.logical(as.integer(args[[11]])) else FALSE
N_WORKERS <- if (length(args) >= 12L) as.integer(args[[12]]) else 10L
if (!is.finite(N_WORKERS) || N_WORKERS < 1L) N_WORKERS <- 1L

script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
SCRIPT_DIR <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[1L])))
} else {
  getwd()
}
EXPERIMENT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
if (length(args) < 1L) ORIGINAL_CSV <- file.path(EXPERIMENT_DIR, "data", "split_seeds", "montana_800m_original_data.csv")
if (length(args) < 2L) ORDER_CSV <- file.path(EXPERIMENT_DIR, "data", "split_seeds", "montana_800m_split_point_orders.csv")
if (length(args) < 3L) OUT_DIR <- file.path(EXPERIMENT_DIR, "outputs", "r_nnmp")
HELPER_FILE <- file.path(SCRIPT_DIR, "montana_helpers_parallel.R")
if (!file.exists(HELPER_FILE)) {
  HELPER_FILE <- "montana_helpers_parallel.R"
}
source(HELPER_FILE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
inputs <- read_montana_inputs(ORIGINAL_CSV, ORDER_CSV)
original <- inputs$original
orders <- inputs$orders
available_seeds <- as.integer(orders$random_seed)
if (identical(SEEDS_ARG, "all")) {
  seeds <- available_seeds
} else {
  seeds <- as.integer(strsplit(SEEDS_ARG, ",")[[1]])
  missing <- setdiff(seeds, available_seeds)
  if (length(missing) > 0L) stop("Requested seeds absent from order table: ", paste(missing, collapse = ", "))
}

run_one_seed <- function(seed) {
  cat("\n============================================================\n")
  cat("NNMP Montana 800m seed ", seed, "\n", sep = "")
  cat("============================================================\n")
  seed_dir <- file.path(OUT_DIR, paste0("seed_", seed))
  dir.create(seed_dir, recursive = TRUE, showWarnings = FALSE)
  dat <- make_seed_split(original, orders, seed, TRAIN_SIZE)
  write.csv(dat, file.path(seed_dir, "split_data.csv"), row.names = FALSE)
  S <- dat[dat$split == "S", , drop = FALSE]
  U <- dat[dat$split == "U", , drop = FALSE]
  S <- S[order(S$row_order), , drop = FALSE]
  U <- U[order(U$point_index), , drop = FALSE]
  rownames(S) <- NULL; rownames(U) <- NULL
  if (nrow(S) <= NEIGHBOR_SIZE) stop("Training size must exceed NEIGHBOR_SIZE.")

  coords_S <- as.matrix(S[, c("lon", "lat")])
  coords_U <- as.matrix(U[, c("lon", "lat")])
  X_S <- as.matrix(S[, c("x0", "x1", "x2")])
  X_U <- as.matrix(U[, c("x0", "x1", "x2")])
  y_S <- as.numeric(S$y_obs)
  lm_start <- lm(y_obs ~ x1 + x2, data = S)
  beta_start <- as.numeric(coef(lm_start))
  n_ga <- ncol(coords_S) + 1L
  ga_start <- matrix(c(-1.5, rep(0, n_ga - 1L)), ncol = 1)
  priors <- list(
    sigmasq_invgamma = c(2, 1),
    tausq_invgamma = c(2, 0.1),
    phi_invgamma = c(3, 1 / 3),
    zeta_invgamma = c(3, 0.2),
    ga_gaus = list(mean_vec = ga_start, var_mat = 2 * diag(n_ga)),
    kasq_invgamma = c(3, 1)
  )
  starting <- list(bb = beta_start, ga = ga_start, kasq = 1, phi = 0.20, zeta = 0.10)
  tuning <- list(phi = 0.08, zeta = 0.08)
  mcmc_settings <- list(n_iter = N_ITER, n_burn = N_BURN, n_thin = N_THIN, n_report = N_REPORT)

  fit_start <- proc.time()
  fit <- nnmp(
    response = y_S,
    covars = X_S,
    coords = coords_S,
    neighbor_size = NEIGHBOR_SIZE,
    marg_family = "gaussian",
    priors = priors,
    starting = starting,
    tuning = tuning,
    ord = seq_len(nrow(S)),
    mcmc_settings = mcmc_settings,
    verbose = TRUE,
    neighbor_info = TRUE,
    model_diag = list("dic", "pplc")
  )
  fit_seconds <- unname((proc.time() - fit_start)[["elapsed"]])
  saveRDS(fit, file.path(seed_dir, "gnnmp_fit.rds"), compress = "xz")

  pred_start <- proc.time()
  pred_U <- predict(
    fit,
    nonref_covars = X_U,
    nonref_coords = coords_U,
    probs = c(0.025, 0.5, 0.975),
    predict_sam = TRUE,
    verbose = TRUE,
    nreport = 500
  )
  prediction_seconds <- unname((proc.time() - pred_start)[["elapsed"]])
  saveRDS(pred_U, file.path(seed_dir, "gnnmp_pred_U.rds"), compress = "xz")

  w_samples_S <- as.matrix(fit$post_samples$zz)
  beta_samples <- as.matrix(fit$post_samples$bb)
  tausq_samples <- as.numeric(fit$post_samples$tausq)
  mu_samples_S <- X_S %*% beta_samples + w_samples_S
  noise_samples_S <- matrix(rnorm(nrow(S) * ncol(mu_samples_S)), nrow = nrow(S), ncol = ncol(mu_samples_S))
  noise_samples_S <- sweep(noise_samples_S, 2, sqrt(tausq_samples), FUN = "*")
  y_samples_S <- mu_samples_S + noise_samples_S
  y_samples_U <- as.matrix(pred_U$obs_sam)

  save_samples(y_samples_S, file.path(seed_dir, "posterior_samples_y_S"), SAVE_SAMPLE_CSV)
  save_samples(y_samples_U, file.path(seed_dir, "posterior_samples_y_U"), SAVE_SAMPLE_CSV)

  pred_y_S <- make_prediction_table("NNMP", seed, S, S$y_obs, y_samples_S)
  pred_y_U <- make_prediction_table("NNMP", seed, U, U$y_obs, y_samples_U)
  write.csv(pred_y_S, file.path(seed_dir, "predictions_y_S.csv"), row.names = FALSE)
  write.csv(pred_y_U, file.path(seed_dir, "predictions_y_U.csv"), row.names = FALSE)

  metric_rows <- rbind(
    make_compact_metric_row("NNMP", seed, "S", pred_y_S),
    make_compact_metric_row("NNMP", seed, "U", pred_y_U)
  )
  write.csv(metric_rows, file.path(seed_dir, "summary.csv"), row.names = FALSE)
  tail_rows <- summarize_tail_events("NNMP", seed, "U", U$y_obs, y_samples_U)
  write.csv(tail_rows, file.path(seed_dir, "tail_event_metrics_U.csv"), row.names = FALSE)

  gamma_summary <- do.call(rbind, lapply(seq_len(nrow(fit$post_samples$ga)), function(j) {
    summarize_vector(fit$post_samples$ga[j, ], paste0("gamma_", j - 1L))
  }))
  parameter_summary <- rbind(
    summarize_vector(fit$post_samples$bb[1, ], "beta_0"),
    summarize_vector(fit$post_samples$bb[2, ], "beta_1"),
    summarize_vector(fit$post_samples$bb[3, ], "beta_2"),
    summarize_vector(fit$post_samples$sigmasq, "sigma_sq"),
    summarize_vector(fit$post_samples$tausq, "tau_sq"),
    summarize_vector(fit$post_samples$phi, "phi"),
    summarize_vector(fit$post_samples$zeta, "zeta"),
    gamma_summary,
    summarize_vector(fit$post_samples$kasq, "kappa_sq")
  )
  write.csv(parameter_summary, file.path(seed_dir, "parameter_summary.csv"), row.names = FALSE)
  runtime <- data.frame(method = "NNMP", random_seed = seed, stage = c("fit_seconds", "prediction_seconds"), value = c(fit_seconds, prediction_seconds))
  write.csv(runtime, file.path(seed_dir, "runtime.csv"), row.names = FALSE)
  diagnostics <- flatten_numeric_object(fit$mod_diag)
  write.csv(diagnostics, file.path(seed_dir, "diagnostics.csv"), row.names = FALSE)
  config <- data.frame(key = c("original_csv", "order_csv", "train_size", "n_total", "n_train", "n_test", "neighbor_size", "n_iter", "n_burn", "n_thin", "n_retained_samples", "coord_method"), value = c(ORIGINAL_CSV, ORDER_CSV, TRAIN_SIZE, nrow(dat), nrow(S), nrow(U), NEIGHBOR_SIZE, N_ITER, N_BURN, N_THIN, (N_ITER - N_BURN) / N_THIN, "raw_lon_lat_euclidean"))
  write.csv(config, file.path(seed_dir, "configuration.csv"), row.names = FALSE)
  list(summary = metric_rows, tail = tail_rows, runtime = runtime)
}

run_seed_set <- function(seeds) {
  if (N_WORKERS <= 1L || length(seeds) <= 1L) {
    return(lapply(seeds, run_one_seed))
  }

  workers <- min(N_WORKERS, length(seeds))
  cat("
Running ", length(seeds), " seed(s) with ", workers, " parallel worker(s).
", sep = "")
  cat("Each seed writes to its own seed_<seed>/ directory.
")

  cl <- parallel::makeCluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  parallel::clusterExport(cl, varlist = "HELPER_FILE", envir = environment())
  parallel::clusterEvalQ(cl, {
    source(HELPER_FILE)
    suppressPackageStartupMessages(library(nnmp))
    NULL
  })
  parallel::clusterSetRNGStream(cl, 20260712)

  export_names <- setdiff(ls(envir = .GlobalEnv), c("cl"))
  parallel::clusterExport(cl, varlist = export_names, envir = .GlobalEnv)

  results <- parallel::parLapply(cl, seeds, function(seed_value) {
    tryCatch(
      run_one_seed(seed_value),
      error = function(e) {
        list(error = conditionMessage(e), random_seed = seed_value)
      }
    )
  })

  errors <- vapply(results, function(x) !is.null(x$error), logical(1))
  if (any(errors)) {
    msg <- paste(
      vapply(results[errors], function(x) paste0("seed ", x$random_seed, ": ", x$error), character(1)),
      collapse = "
"
    )
    stop("One or more parallel seed jobs failed:
", msg)
  }

  results
}

seed_results <- run_seed_set(seeds)
all_summary <- lapply(seed_results, `[[`, "summary")
all_tail <- lapply(seed_results, `[[`, "tail")
all_runtime <- lapply(seed_results, `[[`, "runtime")
summary_all <- do.call(rbind, all_summary)
tail_all <- do.call(rbind, all_tail)
runtime_all <- do.call(rbind, all_runtime)
write.csv(summary_all, file.path(OUT_DIR, "summary_by_seed.csv"), row.names = FALSE)
write.csv(tail_all, file.path(OUT_DIR, "tail_event_metrics_by_seed.csv"), row.names = FALSE)
write.csv(runtime_all, file.path(OUT_DIR, "runtime_by_seed.csv"), row.names = FALSE)
summary_overall <- summarise_across_seeds(summary_all, id_cols = c("method", "split"))
tail_overall <- summarise_across_seeds(tail_all, id_cols = c("method", "split", "quantile_probability", "tail"))
write.csv(summary_overall, file.path(OUT_DIR, "summary_across_seeds.csv"), row.names = FALSE)
write.csv(tail_overall, file.path(OUT_DIR, "tail_event_metrics_across_seeds.csv"), row.names = FALSE)
cat("\nDone. NNMP results written to:\n", normalizePath(OUT_DIR), "\n", sep = "")
