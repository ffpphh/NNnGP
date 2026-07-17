#!/usr/bin/env Rscript
# ============================================================
# Standalone batch script: latent NNGP for weak / median / strong
#
# Expected directory structure:
# simulation_data/
# ├── weak/matern_gp_nnngp_data.csv
# ├── median/matern_gp_nnngp_data.csv
# └── strong/matern_gp_nnngp_data.csv
#
# Outputs:
# results/nngp/
# ├── summary_by_scenario.csv
# ├── summary_by_scenario_long.csv
# ├── weak/
# ├── median/
# └── strong/
#
# No other R scripts are required.
# ============================================================

rm(list = ls())
suppressPackageStartupMessages(library(spNNGP))
set.seed(2026)

# ----------------------------
# 0. User settings
# ----------------------------
script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
SCRIPT_DIR <- if (length(script_file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", script_file_arg[1L])))
} else {
  getwd()
}
EXPERIMENT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
INPUT_ROOT <- file.path(EXPERIMENT_DIR, "data")
RESULTS_ROOT <- file.path(EXPERIMENT_DIR, "outputs", "r_baselines")
METHOD_DIR <- file.path(RESULTS_ROOT, "nngp")
DATA_FILE_NAME <- "matern_gp_nnngp_data.csv"

SCENARIOS <- c("weak", "median", "strong")
SCENARIO_LABELS <- c(
  weak = "Weak non-linearity",
  median = "Medium non-linearity",
  strong = "Strong non-linearity"
)

NEIGHBOR_SIZE <- 10L
N_SAMPLES <- 10000L
BURN_START <- 5001L
THIN <- 5L
N_THREADS <- 1L

dir.create(METHOD_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# 1. Settings matching the Python simulation defaults
# ----------------------------
python_sigma_f <- 1.0
python_length_scale <- 0.2
python_sigma_epsilon <- 0.1

sigma_sq_start <- python_sigma_f^2
tau_sq_start <- python_sigma_epsilon^2
phi_start <- sqrt(3) / python_length_scale
nu_fixed <- 1.5

starting <- list(
  "sigma.sq" = sigma_sq_start,
  "tau.sq" = tau_sq_start,
  "phi" = phi_start,
  "nu" = nu_fixed
)

tuning <- list(
  "phi" = max(0.05, 0.02 * phi_start),
  "nu" = 0.0
)

priors <- list(
  "sigma.sq.IG" = c(2.0, sigma_sq_start),
  "tau.sq.IG" = c(2.0, tau_sq_start),
  "phi.Unif" = c(max(phi_start / 10, 1e-4), phi_start * 10),
  "nu.Unif" = c(0.25, 2.5)
)

# ----------------------------
# 2. Helper functions
# ----------------------------
rmse <- function(truth, estimate) {
  sqrt(mean((truth - estimate)^2, na.rm = TRUE))
}

mae <- function(truth, estimate) {
  mean(abs(truth - estimate), na.rm = TRUE)
}

safe_cor <- function(truth, estimate) {
  ok <- is.finite(truth) & is.finite(estimate)
  truth <- truth[ok]
  estimate <- estimate[ok]
  
  if (length(truth) < 3L || sd(truth) == 0 || sd(estimate) == 0) {
    return(NA_real_)
  }
  
  cor(truth, estimate)
}

rsr <- function(truth, estimate) {
  ok <- is.finite(truth) & is.finite(estimate)
  truth <- truth[ok]
  estimate <- estimate[ok]
  
  if (length(truth) < 2L || sd(truth) == 0) {
    return(NA_real_)
  }
  
  rmse(truth, estimate) / sd(truth)
}

row_sd <- function(sample_matrix) {
  apply(sample_matrix, 1, sd)
}

row_quantiles <- function(sample_matrix) {
  t(
    apply(
      sample_matrix,
      1,
      quantile,
      probs = c(0.025, 0.5, 0.975),
      names = FALSE
    )
  )
}

crps_ensemble_rows <- function(truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  
  if (nrow(sample_matrix) != length(truth)) {
    stop("CRPS: truth length must equal the number of sample-matrix rows.")
  }
  
  n_draws <- ncol(sample_matrix)
  
  if (n_draws < 2L) {
    stop("CRPS requires at least two posterior draws.")
  }
  
  weights <- 2 * seq_len(n_draws) - n_draws - 1
  
  vapply(
    seq_len(nrow(sample_matrix)),
    function(i) {
      draws <- as.numeric(sample_matrix[i, ])
      first_term <- mean(abs(draws - truth[i]))
      second_term <- sum(weights * sort(draws)) / (n_draws^2)
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
  
  out <- data.frame(
    method = method,
    variable = variable,
    split = as.character(data_frame$split),
    point_index = data_frame$point_index,
    row_order = data_frame$row_order,
    x = data_frame$x,
    y = data_frame$y,
    true_value = as.numeric(truth),
    pred_mean = pred_mean,
    pred_sd = pred_sd,
    pred_q025 = qq[, 1],
    pred_median = qq[, 2],
    pred_q975 = qq[, 3],
    ci_width_95 = qq[, 3] - qq[, 1],
    crps = crps_ensemble_rows(truth, sample_matrix),
    stringsAsFactors = FALSE
  )
  
  out[order(out$point_index), , drop = FALSE]
}

make_metric_row <- function(method, variable, split, result_table) {
  data.frame(
    method = method,
    variable = variable,
    split = split,
    n_locations = nrow(result_table),
    rmse = rmse(result_table$true_value, result_table$pred_mean),
    rsr = rsr(result_table$true_value, result_table$pred_mean),
    mae = mae(result_table$true_value, result_table$pred_mean),
    corr = safe_cor(result_table$true_value, result_table$pred_mean),
    mean_pred_sd = mean(result_table$pred_sd),
    coverage_mean_plus_minus_1.96sd = mean(
      result_table$true_value >= result_table$pred_mean - 1.96 * result_table$pred_sd &
        result_table$true_value <= result_table$pred_mean + 1.96 * result_table$pred_sd
    ),
    coverage_quantile_95 = mean(
      result_table$true_value >= result_table$pred_q025 &
        result_table$true_value <= result_table$pred_q975
    ),
    ci_width_mean_plus_minus_1.96sd = mean(2 * 1.96 * result_table$pred_sd),
    ci_width_quantile_95 = mean(result_table$ci_width_95),
    crps = mean(result_table$crps),
    stringsAsFactors = FALSE
  )
}

posterior_summary <- function(samples, parameter_group) {
  samples <- as.matrix(samples)
  
  if (is.null(colnames(samples))) {
    colnames(samples) <- paste0("V", seq_len(ncol(samples)))
  }
  
  data.frame(
    parameter_group = parameter_group,
    parameter = colnames(samples),
    mean = colMeans(samples),
    sd = apply(samples, 2, sd),
    q025 = apply(samples, 2, quantile, probs = 0.025),
    median = apply(samples, 2, median),
    q975 = apply(samples, 2, quantile, probs = 0.975),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

summarize_derived_vector <- function(samples, parameter_group, parameter_name) {
  data.frame(
    parameter_group = parameter_group,
    parameter = parameter_name,
    mean = mean(samples),
    sd = sd(samples),
    q025 = unname(quantile(samples, 0.025)),
    median = unname(quantile(samples, 0.5)),
    q975 = unname(quantile(samples, 0.975)),
    stringsAsFactors = FALSE
  )
}

# ----------------------------
# 3. Run one scenario
# ----------------------------
run_one_scenario <- function(scenario_name) {
  scenario_label <- unname(SCENARIO_LABELS[[scenario_name]])
  csv_file <- file.path(INPUT_ROOT, scenario_name, DATA_FILE_NAME)
  out_dir <- file.path(METHOD_DIR, scenario_name)
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (!file.exists(csv_file)) {
    stop("CSV not found: ", csv_file)
  }
  
  cat("\n============================================================\n")
  cat("Running NNGP scenario:", scenario_name, "\n")
  cat("Display label:        ", scenario_label, "\n")
  cat("CSV file:             ", csv_file, "\n")
  cat("Output directory:     ", out_dir, "\n")
  cat("============================================================\n\n")
  
  dat <- read.csv(csv_file, stringsAsFactors = FALSE, check.names = FALSE)
  
  required_cols <- c(
    "row_order", "point_index", "split",
    "x", "y", "w", "y_obs",
    "x0", "x1", "x2"
  )
  
  missing_cols <- setdiff(required_cols, names(dat))
  
  if (length(missing_cols) > 0L) {
    stop("Missing required CSV columns: ", paste(missing_cols, collapse = ", "))
  }
  
  if (!all(dat$split %in% c("S", "U"))) {
    stop("split must contain only 'S' and 'U'.")
  }
  
  train <- dat[dat$split == "S", , drop = FALSE]
  pred <- dat[dat$split == "U", , drop = FALSE]
  
  if (nrow(train) <= NEIGHBOR_SIZE) {
    stop("Training size must exceed NEIGHBOR_SIZE.")
  }
  
  if (anyDuplicated(train[, c("x", "y")])) {
    stop("Duplicate training coordinates found.")
  }
  
  if (anyNA(train$row_order) || anyDuplicated(train$row_order)) {
    stop("Training row_order must be finite and unique.")
  }
  
  if (anyNA(pred$row_order) || anyDuplicated(pred$row_order)) {
    stop("Prediction row_order must be finite and unique.")
  }
  
  train <- train[order(train$row_order), , drop = FALSE]
  pred <- pred[order(pred$point_index), , drop = FALSE]
  
  rownames(train) <- NULL
  rownames(pred) <- NULL
  
  if (max(train$row_order) >= min(pred$row_order)) {
    stop("Expected all S rows to precede all U rows according to row_order.")
  }
  
  ordering_used <- "python_maximin_order_from_row_order"
  
  coords_S <- as.matrix(train[, c("x", "y")])
  coords_U <- as.matrix(pred[, c("x", "y")])
  
  formula_nngp <- y_obs ~ x1 + x2
  X_S <- model.matrix(delete.response(terms(formula_nngp)), data = train)
  X_U <- model.matrix(delete.response(terms(formula_nngp)), data = pred)
  
  cat("CSV summary:\n")
  cat("  total grid points:", nrow(dat), "\n")
  cat("  reference set S:  ", nrow(train), "\n")
  cat("  prediction set U: ", nrow(pred), "\n")
  cat("  DAG order:        Python maximin order from row_order\n\n")
  
  cat("Fitting latent NNGP...\n")
  fit_start <- proc.time()
  
  fit <- spNNGP(
    formula = formula_nngp,
    data = train,
    coords = coords_S,
    method = "latent",
    family = "gaussian",
    n.neighbors = NEIGHBOR_SIZE,
    starting = starting,
    tuning = tuning,
    priors = priors,
    cov.model = "matern",
    n.samples = N_SAMPLES,
    n.omp.threads = N_THREADS,
    search.type = "brute",
    ord = seq_len(nrow(train)),
    return.neighbor.info = TRUE,
    verbose = TRUE,
    n.report = max(100L, floor(N_SAMPLES / 20L))
  )
  
  fit_seconds <- unname((proc.time() - fit_start)[["elapsed"]])
  
  saveRDS(fit, file.path(out_dir, "latent_nngp_fit.rds"))
  
  sub_sample <- list(
    start = BURN_START,
    end = N_SAMPLES,
    thin = THIN
  )
  
  cat("\nPredicting at U locations...\n")
  prediction_start <- proc.time()
  
  pred_U <- predict(
    fit,
    X.0 = X_U,
    coords.0 = coords_U,
    sub.sample = sub_sample,
    n.omp.threads = N_THREADS,
    verbose = TRUE,
    n.report = 100L
  )
  
  prediction_seconds <- unname((proc.time() - prediction_start)[["elapsed"]])
  
  saveRDS(pred_U, file.path(out_dir, "latent_nngp_prediction.rds"))
  
  if (is.null(pred_U$p.y.0) || is.null(pred_U$p.w.0)) {
    stop("Prediction did not return p.y.0 and p.w.0.")
  }
  
  keep <- seq.int(BURN_START, N_SAMPLES, by = THIN)
  
  beta_samples_all <- as.matrix(fit$p.beta.samples)
  theta_samples_all <- as.matrix(fit$p.theta.samples)
  
  beta_samples <- beta_samples_all[keep, , drop = FALSE]
  theta_samples <- theta_samples_all[keep, , drop = FALSE]
  
  w_samples_S <- as.matrix(fit$p.w.samples)[, keep, drop = FALSE]
  w_samples_U <- as.matrix(pred_U$p.w.0)
  y_samples_U <- as.matrix(pred_U$p.y.0)
  
  if (nrow(w_samples_S) != nrow(train)) {
    stop("Unexpected dimensions in fit$p.w.samples.")
  }
  
  if (nrow(w_samples_U) != nrow(pred) || nrow(y_samples_U) != nrow(pred)) {
    stop("Unexpected dimensions in the NNGP U prediction object.")
  }
  
  tau_candidates <- grep(
    "^tau(\\.sq)?$|tau\\.sq|tausq",
    colnames(theta_samples),
    value = TRUE,
    ignore.case = TRUE
  )
  
  if (length(tau_candidates) != 1L) {
    stop(
      "Could not identify a unique tau-squared column in fit$p.theta.samples. ",
      "Available columns: ",
      paste(colnames(theta_samples), collapse = ", ")
    )
  }
  
  tau_sq_samples <- as.numeric(theta_samples[, tau_candidates])
  
  mu_samples_S <- X_S %*% t(beta_samples) + w_samples_S
  
  noise_samples_S <- matrix(
    rnorm(nrow(train) * ncol(mu_samples_S)),
    nrow = nrow(train),
    ncol = ncol(mu_samples_S)
  )
  
  noise_samples_S <- sweep(
    noise_samples_S,
    2,
    sqrt(tau_sq_samples),
    FUN = "*"
  )
  
  y_samples_S <- mu_samples_S + noise_samples_S
  
  pred_y_S <- make_prediction_table("NNGP", "y", train, train$y_obs, y_samples_S)
  pred_y_U <- make_prediction_table("NNGP", "y", pred, pred$y_obs, y_samples_U)
  pred_w_S <- make_prediction_table("NNGP", "w", train, train$w, w_samples_S)
  pred_w_U <- make_prediction_table("NNGP", "w", pred, pred$w, w_samples_U)
  
  predictions_y <- rbind(pred_y_S, pred_y_U)
  predictions_y <- predictions_y[order(predictions_y$point_index), , drop = FALSE]
  
  predictions_w <- rbind(pred_w_S, pred_w_U)
  predictions_w <- predictions_w[order(predictions_w$point_index), , drop = FALSE]
  
  write.csv(
    predictions_y,
    file.path(out_dir, "predictions_y.csv"),
    row.names = FALSE
  )
  
  write.csv(
    predictions_w,
    file.path(out_dir, "predictions_w.csv"),
    row.names = FALSE
  )
  
  # Save the retained posterior samples used for CRPS, intervals, and SD.
  # Rows are spatial locations and columns are posterior draws.
  write.csv(
    y_samples_S,
    file.path(out_dir, "posterior_samples_y_S.csv"),
    row.names = FALSE
  )
  
  write.csv(
    y_samples_U,
    file.path(out_dir, "posterior_samples_y_U.csv"),
    row.names = FALSE
  )
  
  write.csv(
    w_samples_S,
    file.path(out_dir, "posterior_samples_w_S.csv"),
    row.names = FALSE
  )
  
  write.csv(
    w_samples_U,
    file.path(out_dir, "posterior_samples_w_U.csv"),
    row.names = FALSE
  )
  
  summary_metrics <- rbind(
    make_metric_row("NNGP", "y", "S", pred_y_S),
    make_metric_row("NNGP", "y", "U", pred_y_U),
    make_metric_row("NNGP", "w", "S", pred_w_S),
    make_metric_row("NNGP", "w", "U", pred_w_U)
  )
  
  write.csv(
    summary_metrics,
    file.path(out_dir, "summary.csv"),
    row.names = FALSE
  )
  
  metrics_long <- reshape(
    summary_metrics,
    varying = c(
      "rmse",
      "rsr",
      "mae",
      "corr",
      "mean_pred_sd",
      "coverage_mean_plus_minus_1.96sd",
      "coverage_quantile_95",
      "ci_width_mean_plus_minus_1.96sd",
      "ci_width_quantile_95",
      "crps"
    ),
    v.names = "value",
    timevar = "metric",
    times = c(
      "rmse",
      "rsr",
      "mae",
      "corr",
      "mean_pred_sd",
      "coverage_mean_plus_minus_1.96sd",
      "coverage_quantile_95",
      "ci_width_mean_plus_minus_1.96sd",
      "ci_width_quantile_95",
      "crps"
    ),
    direction = "long"
  )
  
  metrics_long <- metrics_long[
    order(metrics_long$variable, metrics_long$split, metrics_long$metric),
    c("method", "variable", "split", "n_locations", "metric", "value"),
    drop = FALSE
  ]
  
  rownames(metrics_long) <- NULL
  
  write.csv(
    metrics_long,
    file.path(out_dir, "metrics_long.csv"),
    row.names = FALSE
  )
  
  beta_parameter_summary <- posterior_summary(beta_samples, "beta")
  theta_parameter_summary <- posterior_summary(theta_samples, "theta")
  tau_parameter_summary <- summarize_derived_vector(
    sqrt(tau_sq_samples),
    "derived_theta",
    "tau"
  )
  
  parameter_summary <- rbind(
    beta_parameter_summary,
    theta_parameter_summary,
    tau_parameter_summary
  )
  
  write.csv(
    parameter_summary,
    file.path(out_dir, "parameter_summary.csv"),
    row.names = FALSE
  )
  
  runtime <- data.frame(
    stage = c(
      "fit_seconds",
      "fit_minutes",
      "prediction_seconds",
      "prediction_minutes"
    ),
    value = c(
      fit_seconds,
      fit_seconds / 60,
      prediction_seconds,
      prediction_seconds / 60
    ),
    stringsAsFactors = FALSE
  )
  
  write.csv(
    runtime,
    file.path(out_dir, "runtime.csv"),
    row.names = FALSE
  )
  
  configuration <- data.frame(
    key = c(
      "scenario_folder",
      "scenario_label",
      "csv_file",
      "n_total",
      "n_reference_S",
      "n_prediction_U",
      "neighbor_size",
      "n_samples",
      "burn_start",
      "thin",
      "n_retained_samples",
      "n_threads",
      "ordering_used",
      "sigma_sq_start",
      "tau_sq_start",
      "phi_start",
      "nu_fixed"
    ),
    value = c(
      scenario_name,
      scenario_label,
      csv_file,
      nrow(dat),
      nrow(train),
      nrow(pred),
      NEIGHBOR_SIZE,
      N_SAMPLES,
      BURN_START,
      THIN,
      length(keep),
      N_THREADS,
      ordering_used,
      sigma_sq_start,
      tau_sq_start,
      phi_start,
      nu_fixed
    ),
    stringsAsFactors = FALSE
  )
  
  write.csv(
    configuration,
    file.path(out_dir, "configuration.csv"),
    row.names = FALSE
  )
  
  y_U_metrics <- summary_metrics[
    summary_metrics$variable == "y" & summary_metrics$split == "U",
    ,
    drop = FALSE
  ]
  
  beta_names <- beta_parameter_summary$parameter
  
  find_beta_mean <- function(index) {
    if (length(beta_names) < index) {
      return(NA_real_)
    }
    
    beta_parameter_summary$mean[index]
  }
  
  publication_row <- data.frame(
    scenario_folder = scenario_name,
    method = "NNGP",
    scenario = scenario_label,
    beta_0 = find_beta_mean(1),
    beta_1 = find_beta_mean(2),
    beta_2 = find_beta_mean(3),
    sigma_e_sq = mean(tau_sq_samples),
    rmspe = y_U_metrics$rmse,
    rsr = y_U_metrics$rsr,
    crps = y_U_metrics$crps,
    ci_cover_95_percent = 100 * y_U_metrics$coverage_quantile_95,
    ci_width_95 = y_U_metrics$ci_width_quantile_95,
    stringsAsFactors = FALSE
  )
  
  cat("\nScenario publication row:\n")
  print(publication_row, row.names = FALSE)
  
  rm(
    fit,
    pred_U,
    w_samples_S,
    w_samples_U,
    y_samples_S,
    y_samples_U,
    beta_samples,
    theta_samples,
    noise_samples_S,
    mu_samples_S
  )
  
  invisible(gc())
  
  publication_row
}

# ----------------------------
# 4. Run weak / median / strong and save method-level summary
# ----------------------------
publication_rows <- lapply(SCENARIOS, run_one_scenario)
summary_by_scenario <- do.call(rbind, publication_rows)

write.csv(
  summary_by_scenario,
  file.path(METHOD_DIR, "summary_by_scenario.csv"),
  row.names = FALSE
)

summary_by_scenario_long <- reshape(
  summary_by_scenario,
  varying = c(
    "beta_0",
    "beta_1",
    "beta_2",
    "sigma_e_sq",
    "rmspe",
    "rsr",
    "crps",
    "ci_cover_95_percent",
    "ci_width_95"
  ),
  v.names = "value",
  timevar = "metric",
  times = c(
    "beta_0",
    "beta_1",
    "beta_2",
    "sigma_e_sq",
    "rmspe",
    "rsr",
    "crps",
    "ci_cover_95_percent",
    "ci_width_95"
  ),
  direction = "long"
)

summary_by_scenario_long <- summary_by_scenario_long[
  order(
    match(summary_by_scenario_long$scenario_folder, SCENARIOS),
    summary_by_scenario_long$metric
  ),
  c("method", "scenario_folder", "scenario", "metric", "value"),
  drop = FALSE
]

rownames(summary_by_scenario_long) <- NULL

write.csv(
  summary_by_scenario_long,
  file.path(METHOD_DIR, "summary_by_scenario_long.csv"),
  row.names = FALSE
)

cat("\n============================================================\n")
cat("All NNGP scenarios completed.\n")
cat("Method-level summary saved to:\n")
cat(file.path(normalizePath(METHOD_DIR), "summary_by_scenario.csv"), "\n")
cat("============================================================\n\n")

print(summary_by_scenario, row.names = FALSE)
