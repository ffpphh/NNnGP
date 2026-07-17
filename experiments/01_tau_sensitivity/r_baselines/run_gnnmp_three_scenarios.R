# ============================================================
# Standalone batch script: Gaussian NNMP for weak / median / strong
#
# Expected directory structure:
# simulation_data/
# ├── weak/matern_gp_nnngp_data.csv
# ├── median/matern_gp_nnngp_data.csv
# └── strong/matern_gp_nnngp_data.csv
#
# Outputs:
# results/nnmp/
# ├── summary_by_scenario.csv
# ├── summary_by_scenario_long.csv
# ├── weak/
# ├── median/
# └── strong/
#
# No other R scripts are required.
# ============================================================

rm(list = ls())
set.seed(20260610)

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
METHOD_DIR <- file.path(RESULTS_ROOT, "nnmp")
DATA_FILE_NAME <- "matern_gp_nnngp_data.csv"

SCENARIOS <- c("weak", "median", "strong")
SCENARIO_LABELS <- c(
  weak = "Weak non-linearity",
  median = "Medium non-linearity",
  strong = "Strong non-linearity"
)

NEIGHBOR_SIZE <- 10L

N_ITER <- 10000L
N_BURN <- 5000L
N_THIN <- 5L
N_REPORT <- 1000L

dir.create(METHOD_DIR, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("nnmp", quietly = TRUE)) {
  stop(
    paste0(
      "Package 'nnmp' is not installed.\n",
      "Install the locally patched C++14 version first, restart RStudio, ",
      "then rerun this script."
    )
  )
}

suppressPackageStartupMessages(library(nnmp))

# ----------------------------
# 1. Helper functions
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

# Ensemble CRPS:
# mean_j |x_j - y| - (1 / (2 m^2)) sum_j sum_k |x_j - x_k|
# Uses a sorted-sample identity to avoid an m x m matrix.
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

summarize_vector <- function(samples, parameter_name) {
  data.frame(
    parameter = parameter_name,
    mean = mean(samples),
    sd = sd(samples),
    q025 = unname(quantile(samples, 0.025)),
    median = unname(quantile(samples, 0.5)),
    q975 = unname(quantile(samples, 0.975)),
    stringsAsFactors = FALSE
  )
}

flatten_numeric_object <- function(x, prefix = "") {
  output <- list()
  
  recurse <- function(value, current_name) {
    if (is.null(value)) {
      return(invisible(NULL))
    }
    
    if (is.list(value)) {
      child_names <- names(value)
      
      if (is.null(child_names)) {
        child_names <- paste0("item_", seq_along(value))
      }
      
      for (i in seq_along(value)) {
        child_name <- if (nzchar(current_name)) {
          paste0(current_name, ".", child_names[i])
        } else {
          child_names[i]
        }
        
        recurse(value[[i]], child_name)
      }
      
      return(invisible(NULL))
    }
    
    value_vector <- as.vector(value)
    
    if (length(value_vector) == 0L) {
      return(invisible(NULL))
    }
    
    item_names <- names(value_vector)
    
    if (is.null(item_names) || any(!nzchar(item_names))) {
      item_names <- if (length(value_vector) == 1L) {
        current_name
      } else {
        paste0(current_name, "[", seq_along(value_vector), "]")
      }
    } else if (nzchar(current_name)) {
      item_names <- paste0(current_name, ".", item_names)
    }
    
    for (i in seq_along(value_vector)) {
      numeric_value <- suppressWarnings(as.numeric(value_vector[i]))
      
      output[[length(output) + 1L]] <<- data.frame(
        item = item_names[i],
        value = if (is.finite(numeric_value)) numeric_value else NA_real_,
        note = if (is.finite(numeric_value)) "" else as.character(value_vector[i]),
        stringsAsFactors = FALSE
      )
    }
    
    invisible(NULL)
  }
  
  recurse(x, prefix)
  
  if (length(output) == 0L) {
    return(
      data.frame(
        item = character(),
        value = numeric(),
        note = character(),
        stringsAsFactors = FALSE
      )
    )
  }
  
  do.call(rbind, output)
}

# ----------------------------
# 2. Run one scenario
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
  cat("Running NNMP scenario:", scenario_name, "\n")
  cat("Display label:        ", scenario_label, "\n")
  cat("CSV file:             ", csv_file, "\n")
  cat("Output directory:     ", out_dir, "\n")
  cat("============================================================\n\n")
  
  dat <- read.csv(csv_file, stringsAsFactors = FALSE)
  
  required_cols <- c(
    "row_order", "point_index", "split",
    "x", "y", "w", "y_obs",
    "x0", "x1", "x2"
  )
  
  missing_cols <- setdiff(required_cols, names(dat))
  
  if (length(missing_cols) > 0L) {
    stop("Missing required CSV columns: ", paste(missing_cols, collapse = ", "))
  }
  
  if (anyDuplicated(dat$point_index) > 0L) {
    stop("point_index must be unique.")
  }
  
  if (!all(dat$split %in% c("S", "U"))) {
    stop("split must contain only 'S' and 'U'.")
  }
  
  S <- dat[dat$split == "S", , drop = FALSE]
  U <- dat[dat$split == "U", , drop = FALSE]
  
  if (nrow(S) <= NEIGHBOR_SIZE) {
    stop("The reference set must contain more rows than NEIGHBOR_SIZE.")
  }
  
  S <- S[order(S$row_order), , drop = FALSE]
  U <- U[order(U$point_index), , drop = FALSE]
  
  rownames(S) <- NULL
  rownames(U) <- NULL
  
  expected_S_order <- seq.int(0L, nrow(S) - 1L)
  
  if (!identical(as.integer(S$row_order), expected_S_order)) {
    stop(
      "After sorting, S$row_order must equal 0, 1, ..., nrow(S)-1. ",
      "Please inspect the CSV export."
    )
  }
  
  ord <- seq_len(nrow(S))
  
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
    ga_gaus = list(
      mean_vec = matrix(c(-1.5, 0, 0), ncol = 1),
      var_mat = 2 * diag(3)
    ),
    kasq_invgamma = c(3, 1)
  )
  
  starting <- list(
    bb = beta_start,
    ga = matrix(c(-1.5, 0, 0), ncol = 1),
    kasq = 1,
    phi = 0.20,
    zeta = 0.10
  )
  
  tuning <- list(
    phi = 0.08,
    zeta = 0.08
  )
  
  mcmc_settings <- list(
    n_iter = N_ITER,
    n_burn = N_BURN,
    n_thin = N_THIN,
    n_report = N_REPORT
  )
  
  cat("CSV summary:\n")
  cat("  total grid points:", nrow(dat), "\n")
  cat("  reference set S:  ", nrow(S), "\n")
  cat("  prediction set U: ", nrow(U), "\n")
  cat("  DAG order:        Python maximin order from row_order\n\n")
  
  cat("Fitting Gaussian NNMP...\n")
  fit_start <- proc.time()
  
  fit_gnnmp <- nnmp(
    response = y_S,
    covars = X_S,
    coords = coords_S,
    neighbor_size = NEIGHBOR_SIZE,
    marg_family = "gaussian",
    priors = priors,
    starting = starting,
    tuning = tuning,
    ord = ord,
    mcmc_settings = mcmc_settings,
    verbose = TRUE,
    neighbor_info = TRUE,
    model_diag = list("dic", "pplc")
  )
  
  fit_seconds <- unname((proc.time() - fit_start)[["elapsed"]])
  
  saveRDS(fit_gnnmp, file.path(out_dir, "gnnmp_fit.rds"))
  
  cat("\nPredicting at U locations...\n")
  prediction_start <- proc.time()
  
  pred_U <- predict(
    fit_gnnmp,
    nonref_covars = X_U,
    nonref_coords = coords_U,
    probs = c(0.025, 0.5, 0.975),
    predict_sam = TRUE,
    verbose = TRUE,
    nreport = 500
  )
  
  prediction_seconds <- unname((proc.time() - prediction_start)[["elapsed"]])
  
  saveRDS(pred_U, file.path(out_dir, "gnnmp_pred_U.rds"))
  
  w_samples_S <- as.matrix(fit_gnnmp$post_samples$zz)
  beta_samples <- as.matrix(fit_gnnmp$post_samples$bb)
  tausq_samples <- as.numeric(fit_gnnmp$post_samples$tausq)
  
  if (nrow(w_samples_S) != nrow(S)) {
    stop("Unexpected dimension for fit_gnnmp$post_samples$zz.")
  }
  
  if (ncol(beta_samples) != ncol(w_samples_S)) {
    stop("Unexpected dimension for fit_gnnmp$post_samples$bb.")
  }
  
  if (length(tausq_samples) != ncol(w_samples_S)) {
    stop("Unexpected dimension for fit_gnnmp$post_samples$tausq.")
  }
  
  mu_samples_S <- X_S %*% beta_samples + w_samples_S
  
  noise_samples_S <- matrix(
    rnorm(nrow(S) * ncol(mu_samples_S)),
    nrow = nrow(S),
    ncol = ncol(mu_samples_S)
  )
  
  noise_samples_S <- sweep(
    noise_samples_S,
    2,
    sqrt(tausq_samples),
    FUN = "*"
  )
  
  y_samples_S <- mu_samples_S + noise_samples_S
  
  w_samples_U <- as.matrix(pred_U$zz_sam)
  y_samples_U <- as.matrix(pred_U$obs_sam)
  
  if (nrow(w_samples_U) != nrow(U) || nrow(y_samples_U) != nrow(U)) {
    stop("Unexpected dimensions in the NNMP U prediction object.")
  }
  
  pred_y_S <- make_prediction_table("NNMP", "y", S, S$y_obs, y_samples_S)
  pred_y_U <- make_prediction_table("NNMP", "y", U, U$y_obs, y_samples_U)
  pred_w_S <- make_prediction_table("NNMP", "w", S, S$w, w_samples_S)
  pred_w_U <- make_prediction_table("NNMP", "w", U, U$w, w_samples_U)
  
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
    make_metric_row("NNMP", "y", "S", pred_y_S),
    make_metric_row("NNMP", "y", "U", pred_y_U),
    make_metric_row("NNMP", "w", "S", pred_w_S),
    make_metric_row("NNMP", "w", "U", pred_w_U)
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
  
  parameter_summary <- rbind(
    summarize_vector(fit_gnnmp$post_samples$bb[1, ], "beta_0"),
    summarize_vector(fit_gnnmp$post_samples$bb[2, ], "beta_1"),
    summarize_vector(fit_gnnmp$post_samples$bb[3, ], "beta_2"),
    summarize_vector(fit_gnnmp$post_samples$sigmasq, "sigma_sq"),
    summarize_vector(sqrt(fit_gnnmp$post_samples$sigmasq), "sigma"),
    summarize_vector(fit_gnnmp$post_samples$tausq, "tau_sq"),
    summarize_vector(sqrt(fit_gnnmp$post_samples$tausq), "tau"),
    summarize_vector(fit_gnnmp$post_samples$phi, "phi"),
    summarize_vector(fit_gnnmp$post_samples$zeta, "zeta"),
    summarize_vector(fit_gnnmp$post_samples$ga[1, ], "gamma_0"),
    summarize_vector(fit_gnnmp$post_samples$ga[2, ], "gamma_1"),
    summarize_vector(fit_gnnmp$post_samples$ga[3, ], "gamma_2"),
    summarize_vector(fit_gnnmp$post_samples$kasq, "kappa_sq")
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
      "n_iter",
      "n_burn",
      "n_thin",
      "n_retained_samples",
      "ordering_used"
    ),
    value = c(
      scenario_name,
      scenario_label,
      csv_file,
      nrow(dat),
      nrow(S),
      nrow(U),
      NEIGHBOR_SIZE,
      N_ITER,
      N_BURN,
      N_THIN,
      (N_ITER - N_BURN) / N_THIN,
      "python_maximin_order_from_row_order"
    ),
    stringsAsFactors = FALSE
  )
  
  write.csv(
    configuration,
    file.path(out_dir, "configuration.csv"),
    row.names = FALSE
  )
  
  diagnostics <- flatten_numeric_object(fit_gnnmp$mod_diag)
  
  write.csv(
    diagnostics,
    file.path(out_dir, "diagnostics.csv"),
    row.names = FALSE
  )
  
  y_U_metrics <- summary_metrics[
    summary_metrics$variable == "y" & summary_metrics$split == "U",
    ,
    drop = FALSE
  ]
  
  get_parameter_mean <- function(parameter_name) {
    parameter_summary$mean[
      parameter_summary$parameter == parameter_name
    ][1]
  }
  
  publication_row <- data.frame(
    scenario_folder = scenario_name,
    method = "NNMP",
    scenario = scenario_label,
    beta_0 = get_parameter_mean("beta_0"),
    beta_1 = get_parameter_mean("beta_1"),
    beta_2 = get_parameter_mean("beta_2"),
    sigma_e_sq = get_parameter_mean("tau_sq"),
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
    fit_gnnmp,
    pred_U,
    w_samples_S,
    w_samples_U,
    y_samples_S,
    y_samples_U,
    beta_samples,
    tausq_samples,
    noise_samples_S,
    mu_samples_S
  )
  
  invisible(gc())
  
  publication_row
}

# ----------------------------
# 3. Run weak / median / strong and save method-level summary
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
cat("All NNMP scenarios completed.\n")
cat("Method-level summary saved to:\n")
cat(file.path(normalizePath(METHOD_DIR), "summary_by_scenario.csv"), "\n")
cat("============================================================\n\n")

print(summary_by_scenario, row.names = FALSE)
