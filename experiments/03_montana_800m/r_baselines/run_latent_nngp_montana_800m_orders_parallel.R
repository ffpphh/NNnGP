#!/usr/bin/env Rscript
# ============================================================
# Montana 800m split-seed latent NNGP baseline.
# Reads:
#   1) montana_800m_original_data.csv
#   2) montana_800m_split_point_orders.csv
# For each order-table row, the first TRAIN_SIZE rows are S and the rest are U.
# The order entries are zero-based row indices into the original data table.
# ============================================================

suppressPackageStartupMessages(library(spNNGP))
set.seed(2026)

args <- commandArgs(trailingOnly = TRUE)
ORIGINAL_CSV <- if (length(args) >= 1L) args[[1]] else NA_character_
ORDER_CSV <- if (length(args) >= 2L) args[[2]] else NA_character_
OUT_DIR <- if (length(args) >= 3L) args[[3]] else NA_character_
TRAIN_SIZE <- if (length(args) >= 4L) as.integer(args[[4]]) else 800L
NEIGHBOR_SIZE <- if (length(args) >= 5L) as.integer(args[[5]]) else 10L
N_SAMPLES <- if (length(args) >= 6L) as.integer(args[[6]]) else 10000L
BURN_START <- if (length(args) >= 7L) as.integer(args[[7]]) else 5001L
THIN <- if (length(args) >= 8L) as.integer(args[[8]]) else 5L
N_THREADS <- if (length(args) >= 9L) as.integer(args[[9]]) else 1L
SEEDS_ARG <- if (length(args) >= 10L) args[[10]] else "all"
SAVE_SAMPLE_CSV <- if (length(args) >= 11L) as.logical(as.integer(args[[11]])) else FALSE
COV_MODEL <- if (length(args) >= 12L) args[[12]] else "exponential"
N_WORKERS <- if (length(args) >= 13L) as.integer(args[[13]]) else 10L
if (!is.finite(N_WORKERS) || N_WORKERS < 1L) N_WORKERS <- 1L
if (N_WORKERS > 1L && N_THREADS > 1L) {
  warning("N_WORKERS > 1 and N_THREADS > 1 may oversubscribe CPU cores. Consider setting N_THREADS = 1.")
}

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
if (length(args) < 3L) OUT_DIR <- file.path(EXPERIMENT_DIR, "outputs", "r_nngp")
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
  cat("NNGP Montana 800m seed ", seed, "\n", sep = "")
  cat("============================================================\n")
  seed_dir <- file.path(OUT_DIR, paste0("seed_", seed))
  dir.create(seed_dir, recursive = TRUE, showWarnings = FALSE)
  dat <- make_seed_split(original, orders, seed, TRAIN_SIZE)
  write.csv(dat, file.path(seed_dir, "split_data.csv"), row.names = FALSE)
  train <- dat[dat$split == "S", , drop = FALSE]
  pred <- dat[dat$split == "U", , drop = FALSE]
  if (nrow(train) <= NEIGHBOR_SIZE) stop("Training size must exceed NEIGHBOR_SIZE.")
  train <- train[order(train$row_order), , drop = FALSE]
  pred <- pred[order(pred$point_index), , drop = FALSE]
  rownames(train) <- NULL; rownames(pred) <- NULL

  coords_S <- as.matrix(train[, c("lon", "lat")])
  coords_U <- as.matrix(pred[, c("lon", "lat")])
  X_S <- model.matrix(y_obs ~ x1 + x2, data = train)
  X_U <- model.matrix(~ x1 + x2, data = pred)
  formula_nngp <- y_obs ~ x1 + x2

  y_var <- var(train$y_obs)
  if (!is.finite(y_var) || y_var <= 0) y_var <- 1
  coord_d <- as.matrix(dist(coords_S)); diag(coord_d) <- Inf
  nn_dist <- apply(coord_d, 1, min)
  med_nn <- median(nn_dist[is.finite(nn_dist) & nn_dist > 0])
  if (!is.finite(med_nn) || med_nn <= 0) med_nn <- 1
  sigma_sq_start <- max(0.80 * y_var, 0.10)
  tau_sq_start <- max(0.20 * y_var, 0.05)
  phi_start <- min(max(3 / med_nn, 0.01), 5.0)
  phi_lower <- 0.001
  phi_upper <- max(2.0, min(10.0, 10 / med_nn))
  starting <- list("sigma.sq" = sigma_sq_start, "tau.sq" = tau_sq_start, "phi" = phi_start)
  tuning <- list("sigma.sq" = 0.05 * sigma_sq_start, "tau.sq" = 0.05 * tau_sq_start, "phi" = max(0.01, 0.02 * phi_start))
  priors <- list("sigma.sq.IG" = c(3.0, 2.0 * sigma_sq_start), "tau.sq.IG" = c(3.0, 2.0 * tau_sq_start), "phi.Unif" = c(phi_lower, phi_upper))
  if (tolower(COV_MODEL) == "matern") {
    starting[["nu"]] <- 1.5
    tuning[["nu"]] <- 0.0
    priors[["nu.Unif"]] <- c(0.25, 2.5)
  }

  fit_start <- proc.time()
  fit <- spNNGP(
    formula = formula_nngp,
    data = train[, c("y_obs", "x1", "x2"), drop = FALSE],
    coords = coords_S,
    method = "latent",
    family = "gaussian",
    n.neighbors = NEIGHBOR_SIZE,
    starting = starting,
    tuning = tuning,
    priors = priors,
    cov.model = COV_MODEL,
    n.samples = N_SAMPLES,
    n.omp.threads = N_THREADS,
    search.type = "brute",
    ord = seq_len(nrow(train)),
    return.neighbor.info = TRUE,
    verbose = TRUE,
    n.report = max(100L, floor(N_SAMPLES / 20L))
  )
  fit_seconds <- unname((proc.time() - fit_start)[["elapsed"]])
  saveRDS(fit, file.path(seed_dir, "latent_nngp_fit.rds"), compress = "xz")

  keep <- seq.int(BURN_START, N_SAMPLES, by = THIN)
  sub_sample <- list(start = BURN_START, end = N_SAMPLES, thin = THIN)
  pred_start <- proc.time()
  pred_U <- predict(
    fit,
    X.0 = X_U,
    coords.0 = coords_U,
    sub.sample = sub_sample,
    n.omp.threads = N_THREADS,
    verbose = TRUE,
    n.report = 100L
  )
  prediction_seconds <- unname((proc.time() - pred_start)[["elapsed"]])
  saveRDS(pred_U, file.path(seed_dir, "latent_nngp_prediction.rds"), compress = "xz")

  beta_samples <- as.matrix(fit$p.beta.samples)[keep, , drop = FALSE]
  theta_samples <- as.matrix(fit$p.theta.samples)[keep, , drop = FALSE]
  w_samples_S <- as.matrix(fit$p.w.samples)[, keep, drop = FALSE]
  y_samples_U <- as.matrix(pred_U$p.y.0)
  tau_col <- grep("^tau(\\.sq)?$|tau\\.sq|tausq", colnames(theta_samples), value = TRUE, ignore.case = TRUE)
  if (length(tau_col) != 1L) stop("Could not identify tau.sq column in p.theta.samples.")
  tau_sq_samples <- as.numeric(theta_samples[, tau_col])
  mu_samples_S <- X_S %*% t(beta_samples) + w_samples_S
  noise_samples_S <- matrix(rnorm(nrow(train) * ncol(mu_samples_S)), nrow = nrow(train), ncol = ncol(mu_samples_S))
  noise_samples_S <- sweep(noise_samples_S, 2, sqrt(tau_sq_samples), FUN = "*")
  y_samples_S <- mu_samples_S + noise_samples_S

  save_samples(y_samples_S, file.path(seed_dir, "posterior_samples_y_S"), SAVE_SAMPLE_CSV)
  save_samples(y_samples_U, file.path(seed_dir, "posterior_samples_y_U"), SAVE_SAMPLE_CSV)

  pred_y_S <- make_prediction_table("NNGP", seed, train, train$y_obs, y_samples_S)
  pred_y_U <- make_prediction_table("NNGP", seed, pred, pred$y_obs, y_samples_U)
  write.csv(pred_y_S, file.path(seed_dir, "predictions_y_S.csv"), row.names = FALSE)
  write.csv(pred_y_U, file.path(seed_dir, "predictions_y_U.csv"), row.names = FALSE)

  metric_rows <- rbind(
    make_compact_metric_row("NNGP", seed, "S", pred_y_S),
    make_compact_metric_row("NNGP", seed, "U", pred_y_U)
  )
  write.csv(metric_rows, file.path(seed_dir, "summary.csv"), row.names = FALSE)

  tail_rows <- summarize_tail_events("NNGP", seed, "U", pred$y_obs, y_samples_U)
  write.csv(tail_rows, file.path(seed_dir, "tail_event_metrics_U.csv"), row.names = FALSE)

  parameter_summary <- rbind(
    data.frame(parameter_group = "beta", parameter = colnames(beta_samples), mean = colMeans(beta_samples), sd = apply(beta_samples, 2, sd), q025 = apply(beta_samples, 2, quantile, 0.025), median = apply(beta_samples, 2, median), q975 = apply(beta_samples, 2, quantile, 0.975), row.names = NULL),
    data.frame(parameter_group = "theta", parameter = colnames(theta_samples), mean = colMeans(theta_samples), sd = apply(theta_samples, 2, sd), q025 = apply(theta_samples, 2, quantile, 0.025), median = apply(theta_samples, 2, median), q975 = apply(theta_samples, 2, quantile, 0.975), row.names = NULL)
  )
  write.csv(parameter_summary, file.path(seed_dir, "parameter_summary.csv"), row.names = FALSE)
  runtime <- data.frame(method = "NNGP", random_seed = seed, stage = c("fit_seconds", "prediction_seconds"), value = c(fit_seconds, prediction_seconds))
  write.csv(runtime, file.path(seed_dir, "runtime.csv"), row.names = FALSE)
  config <- data.frame(key = c("original_csv", "order_csv", "train_size", "n_total", "n_train", "n_test", "neighbor_size", "n_samples", "burn_start", "thin", "n_retained_samples", "cov_model", "coord_method"), value = c(ORIGINAL_CSV, ORDER_CSV, TRAIN_SIZE, nrow(dat), nrow(train), nrow(pred), NEIGHBOR_SIZE, N_SAMPLES, BURN_START, THIN, length(keep), COV_MODEL, "raw_lon_lat_euclidean"))
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
    suppressPackageStartupMessages(library(spNNGP))
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
cat("\nDone. NNGP results written to:\n", normalizePath(OUT_DIR), "\n", sep = "")
