#!/usr/bin/env Rscript
# ============================================================
# Compute Montana 800m tail-event metrics from saved posterior samples
#
# This script DOES NOT refit NNGP/NNMP. It only reads saved posterior
# predictive samples, usually files like:
#   seed_43/posterior_samples_y_U.rds
#   seed_43/split_data.csv
# and computes the same style of tail-event summary table as
# tail_event_metric_summary.csv.
#
# Main output:
#   tail_event_metric_summary.csv
#
# This is ONE complete long table. It does NOT average over seeds.
# A column random_seed identifies which split/order each row belongs to.
# Columns:
#   random_seed, event, method, evaluation_set, tail, quantile_probability,
#   threshold, threshold_abs, n_locations, n_S, n_U, event_count,
#   event_rate, mean_predicted_event_probability, brier_score,
#   brier_reference_climatology, brier_skill_score, crps,
#   coverage_95_all_locations, ci_width_95_all_locations,
#   twcrps_tail_weighted, extreme_value_coverage_95,
#   extreme_value_ci_width_95
#
# Usage examples:
#   Rscript compute_tail_event_metrics_from_saved_samples.R \
#     "NNGP=nngp_montana_800m_parallel_results,NNMP=nnmp_montana_800m_parallel_results" \
#     tail_metric_outputs \
#     montana_800m_original_data.csv \
#     montana_800m_split_point_orders.csv \
#     800 all
#
# For one method only:
#   Rscript compute_tail_event_metrics_from_saved_samples.R \
#     "NNGP=nngp_montana_800m_parallel_results" tail_metric_outputs
# ============================================================

options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
script_args <- commandArgs(trailingOnly = FALSE)
script_file_arg <- grep("^--file=", script_args, value = TRUE)
SCRIPT_DIR <- if (length(script_file_arg) > 0L) dirname(normalizePath(sub("^--file=", "", script_file_arg[1L]))) else getwd()
EXPERIMENT_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)

RESULT_SPECS <- if (length(args) >= 1L) {
  args[[1]]
} else {
  paste0(
    "NNGP=", file.path(EXPERIMENT_DIR, "outputs", "r_nngp"),
    ",NNMP=", file.path(EXPERIMENT_DIR, "outputs", "r_nnmp")
  )
}

OUT_DIR <- if (length(args) >= 2L) args[[2]] else file.path(EXPERIMENT_DIR, "outputs", "nnngp", "tail_event_metric_summaries")
ORIGINAL_DATA_CSV <- if (length(args) >= 3L) args[[3]] else file.path(EXPERIMENT_DIR, "data", "split_seeds", "montana_800m_original_data.csv")
ORDER_TABLE_CSV <- if (length(args) >= 4L) args[[4]] else file.path(EXPERIMENT_DIR, "data", "split_seeds", "montana_800m_split_point_orders.csv")
TRAIN_SIZE <- if (length(args) >= 5L) as.integer(args[[5]]) else 800L
SEEDS_ARG <- if (length(args) >= 6L) args[[6]] else "all"
PROBS_ARG <- if (length(args) >= 7L) args[[7]] else "0.80:0.98:0.01"
VALUE_COL <- if (length(args) >= 8L) args[[8]] else "log_ppt_2025_standardized"

# If TRUE, save per-location CRPS and coverage diagnostics for each seed.
SAVE_LOCATION_DIAGNOSTICS <- if (length(args) >= 9L) as.logical(as.integer(args[[9]])) else FALSE

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------
# Parsing helpers
# ----------------------------
parse_result_specs <- function(text) {
  parts <- unlist(strsplit(text, ",", fixed = TRUE))
  parts <- trimws(parts[nzchar(trimws(parts))])
  if (length(parts) == 0L) stop("No result directories were provided.")
  
  out <- list()
  for (part in parts) {
    if (grepl("=", part, fixed = TRUE)) {
      kv <- unlist(strsplit(part, "=", fixed = TRUE))
      method <- trimws(kv[[1]])
      path <- trimws(paste(kv[-1], collapse = "="))
    } else {
      path <- part
      b <- basename(normalizePath(path, mustWork = FALSE))
      method <- if (grepl("nnmp", b, ignore.case = TRUE)) {
        "NNMP"
      } else if (grepl("nngp", b, ignore.case = TRUE)) {
        "NNGP"
      } else {
        b
      }
    }
    out[[length(out) + 1L]] <- list(method = method, path = path)
  }
  out
}

parse_probs <- function(text) {
  text <- trimws(text)
  if (grepl(":", text, fixed = TRUE)) {
    z <- as.numeric(unlist(strsplit(text, ":", fixed = TRUE)))
    if (length(z) != 3L || any(!is.finite(z))) {
      stop("PROBS_ARG using ':' must have form start:end:step, e.g. 0.80:0.98:0.01")
    }
    return(seq(z[1], z[2], by = z[3]))
  }
  z <- as.numeric(unlist(strsplit(text, ",", fixed = TRUE)))
  if (any(!is.finite(z))) stop("Could not parse PROBS_ARG.")
  z
}

parse_seeds <- function(text, available = NULL) {
  text <- trimws(text)
  if (tolower(text) == "all") return(available)
  z <- as.integer(unlist(strsplit(text, ",", fixed = TRUE)))
  if (any(is.na(z))) stop("Could not parse SEEDS_ARG. Use 'all' or comma-separated integers.")
  z
}

find_seed_dirs <- function(result_dir) {
  if (!dir.exists(result_dir)) stop("Result directory not found: ", result_dir)
  dirs <- list.dirs(result_dir, full.names = TRUE, recursive = FALSE)
  seed_dirs <- dirs[grepl("seed_[0-9]+$", basename(dirs))]
  if (length(seed_dirs) == 0L) {
    # Allow a result directory that itself contains posterior_samples_y_U.*
    has_samples <- any(file.exists(file.path(result_dir, c(
      "posterior_samples_y_U.rds",
      "posterior_samples_y_U.csv"
    ))))
    if (has_samples) seed_dirs <- result_dir
  }
  seed_dirs
}

seed_from_dir <- function(seed_dir) {
  b <- basename(seed_dir)
  z <- sub("^seed_", "", b)
  if (grepl("^[0-9]+$", z)) as.integer(z) else NA_integer_
}

# ----------------------------
# Numeric helpers
# ----------------------------
safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  mean(x)
}

safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) return(NA_real_)
  sd(x)
}

crps_ensemble_rows <- function(truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  truth <- as.numeric(truth)
  if (nrow(sample_matrix) != length(truth)) {
    stop("CRPS: truth length does not match number of sample rows.")
  }
  n_draws <- ncol(sample_matrix)
  if (n_draws < 2L) stop("CRPS requires at least two posterior draws.")
  
  weights <- 2 * seq_len(n_draws) - n_draws - 1
  out <- numeric(nrow(sample_matrix))
  
  for (i in seq_len(nrow(sample_matrix))) {
    draws <- as.numeric(sample_matrix[i, ])
    y <- truth[i]
    if (!is.finite(y) || any(!is.finite(draws))) {
      out[i] <- NA_real_
    } else {
      sorted_draws <- sort(draws)
      first_term <- mean(abs(draws - y))
      second_term <- sum(weights * sorted_draws) / (n_draws^2)
      out[i] <- first_term - second_term
    }
  }
  out
}

# Threshold-weighted CRPS used here:
#   upper tail: CRPS(max(Y-c, 0), max(y-c, 0))
#   lower tail: CRPS(max(c-Y, 0), max(c-y, 0))
# This produces a proper tail-focused score and matches the output column
# twcrps_tail_weighted in the uploaded summary style.
twcrps_ensemble_rows <- function(truth, sample_matrix, threshold, tail) {
  sample_matrix <- as.matrix(sample_matrix)
  truth <- as.numeric(truth)
  if (nrow(sample_matrix) != length(truth)) {
    stop("twCRPS: truth length does not match number of sample rows.")
  }
  n_draws <- ncol(sample_matrix)
  weights <- 2 * seq_len(n_draws) - n_draws - 1
  out <- numeric(nrow(sample_matrix))
  
  for (i in seq_len(nrow(sample_matrix))) {
    draws <- as.numeric(sample_matrix[i, ])
    y <- truth[i]
    if (!is.finite(y) || any(!is.finite(draws))) {
      out[i] <- NA_real_
    } else {
      if (tail == "upper") {
        draws_tw <- pmax(draws - threshold, 0)
        y_tw <- max(y - threshold, 0)
      } else {
        draws_tw <- pmax(threshold - draws, 0)
        y_tw <- max(threshold - y, 0)
      }
      sorted_draws <- sort(draws_tw)
      first_term <- mean(abs(draws_tw - y_tw))
      second_term <- sum(weights * sorted_draws) / (n_draws^2)
      out[i] <- first_term - second_term
    }
  }
  out
}

format_event <- function(tail, threshold_abs) {
  if (tail == "lower") {
    sprintf("Y < -%.2f", threshold_abs)
  } else {
    sprintf("Y > %.2f", threshold_abs)
  }
}

summarize_one_seed <- function(method, seed, truth, sample_matrix, probabilities, out_seed_dir = NULL) {
  sample_matrix <- as.matrix(sample_matrix)
  truth <- as.numeric(truth)
  n <- length(truth)
  if (nrow(sample_matrix) != n) {
    stop("Seed ", seed, ": sample rows = ", nrow(sample_matrix), " but truth length = ", n)
  }
  
  cat("  computing ordinary CRPS and 95% coverage for seed ", seed, "\n", sep = "")
  crps_values <- crps_ensemble_rows(truth, sample_matrix)
  pred_q025 <- apply(sample_matrix, 1, quantile, probs = 0.025, names = FALSE, na.rm = TRUE)
  pred_q975 <- apply(sample_matrix, 1, quantile, probs = 0.975, names = FALSE, na.rm = TRUE)
  ci_width_95 <- pred_q975 - pred_q025
  covered <- truth >= pred_q025 & truth <= pred_q975
  
  crps_mean <- safe_mean(crps_values)
  coverage_all <- safe_mean(as.numeric(covered))
  ci_width_all <- safe_mean(ci_width_95)
  
  if (SAVE_LOCATION_DIAGNOSTICS && !is.null(out_seed_dir)) {
    loc_diag <- data.frame(
      row_id_U = seq_len(n),
      truth = truth,
      crps = crps_values,
      pred_q025 = pred_q025,
      pred_q975 = pred_q975,
      ci_width_95 = ci_width_95,
      covered_95 = covered,
      stringsAsFactors = FALSE
    )
    write.csv(loc_diag, file.path(out_seed_dir, "location_diagnostics_y_U.csv"), row.names = FALSE)
  }
  
  rows <- list()
  rr <- 0L
  eps <- 1e-12
  
  for (probability in probabilities) {
    magnitude <- qnorm(probability)
    for (tail in c("lower", "upper")) {
      threshold <- if (tail == "lower") -magnitude else magnitude
      threshold_abs <- abs(threshold)
      
      if (tail == "lower") {
        truth_event <- truth < threshold
        pred_prob <- rowMeans(sample_matrix < threshold, na.rm = TRUE)
      } else {
        truth_event <- truth > threshold
        pred_prob <- rowMeans(sample_matrix > threshold, na.rm = TRUE)
      }
      
      event_rate <- mean(truth_event, na.rm = TRUE)
      event_count <- sum(truth_event, na.rm = TRUE)
      brier <- mean((pred_prob - as.numeric(truth_event))^2, na.rm = TRUE)
      brier_ref <- mean((event_rate - as.numeric(truth_event))^2, na.rm = TRUE)
      bss <- if (is.finite(brier_ref) && brier_ref > 0) 1 - brier / brier_ref else NA_real_
      
      cat("    ", format_event(tail, threshold_abs), ": twCRPS\n", sep = "")
      tw_values <- twcrps_ensemble_rows(truth, sample_matrix, threshold, tail)
      tw_mean <- safe_mean(tw_values)
      
      extreme_coverage <- if (event_count > 0L) {
        safe_mean(as.numeric(covered[truth_event]))
      } else {
        NA_real_
      }
      
      extreme_ci_width <- if (event_count > 0L) {
        safe_mean(ci_width_95[truth_event])
      } else {
        NA_real_
      }
      
      rr <- rr + 1L
      rows[[rr]] <- data.frame(
        random_seed = seed,
        event = format_event(tail, threshold_abs),
        method = method,
        evaluation_set = "y_U",
        tail = tail,
        quantile_probability = probability,
        threshold = threshold,
        threshold_abs = threshold_abs,
        n_locations = n,
        n_S = 0L,
        n_U = n,
        event_count = event_count,
        event_rate = event_rate,
        mean_predicted_event_probability = mean(pred_prob, na.rm = TRUE),
        brier_score = brier,
        brier_reference_climatology = brier_ref,
        brier_skill_score = bss,
        crps = crps_mean,
        coverage_95_all_locations = coverage_all,
        ci_width_95_all_locations = ci_width_all,
        twcrps_tail_weighted = tw_mean,
        extreme_value_coverage_95 = extreme_coverage,
        extreme_value_ci_width_95 = extreme_ci_width,
        stringsAsFactors = FALSE
      )
    }
  }
  
  do.call(rbind, rows)
}

# ----------------------------
# Loading posterior samples and truth
# ----------------------------
read_samples_y_U <- function(seed_dir) {
  candidates <- c(
    file.path(seed_dir, "posterior_samples_y_U.rds"),
    file.path(seed_dir, "posterior_samples_y_U.csv"),
    file.path(seed_dir, "posterior_samples_y_U.RDS")
  )
  hit <- candidates[file.exists(candidates)][1]
  if (is.na(hit)) {
    stop("Could not find posterior_samples_y_U.rds or posterior_samples_y_U.csv in ", seed_dir)
  }
  
  if (grepl("\\.rds$|\\.RDS$", hit)) {
    obj <- readRDS(hit)
    if (is.data.frame(obj)) obj <- as.matrix(obj)
    if (is.list(obj) && !is.null(obj$samples)) obj <- obj$samples
    samples <- as.matrix(obj)
  } else {
    samples <- as.matrix(read.csv(hit, check.names = FALSE))
  }
  
  storage.mode(samples) <- "double"
  if (any(!is.finite(samples))) stop("Non-finite values found in posterior samples: ", hit)
  attr(samples, "source_file") <- hit
  samples
}

read_truth_from_split_data <- function(seed_dir, expected_n, value_col = VALUE_COL) {
  split_path <- file.path(seed_dir, "split_data.csv")
  if (!file.exists(split_path)) return(NULL)
  dat <- read.csv(split_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("split" %in% names(dat))) return(NULL)
  u <- dat[dat$split == "U", , drop = FALSE]
  candidates <- c(value_col, "y_obs", "true_value", "log_ppt_2025_standardized")
  value_hit <- candidates[candidates %in% names(u)][1]
  if (is.na(value_hit)) return(NULL)
  truth <- suppressWarnings(as.numeric(u[[value_hit]]))
  if (length(truth) != expected_n || any(!is.finite(truth))) return(NULL)
  truth
}

read_truth_from_original_order <- function(seed, expected_n, original, orders, train_size, value_col = VALUE_COL) {
  if (is.null(original) || is.null(orders) || !is.finite(seed)) return(NULL)
  if (!("random_seed" %in% names(orders))) return(NULL)
  row_id <- which(as.integer(orders$random_seed) == as.integer(seed))
  if (length(row_id) != 1L) return(NULL)
  
  order_cols <- setdiff(names(orders), "random_seed")
  point_order <- as.integer(orders[row_id, order_cols])
  if (length(point_order) != nrow(original)) {
    stop("Seed ", seed, ": point order length does not match original data rows.")
  }
  if (!setequal(point_order, seq.int(0L, nrow(original) - 1L))) {
    stop("Seed ", seed, ": point order is not a 0-based permutation of original row indices.")
  }
  
  if (!(value_col %in% names(original))) {
    stop("Value column '", value_col, "' was not found in original data.")
  }
  u_index0 <- point_order[(train_size + 1L):length(point_order)]
  truth <- suppressWarnings(as.numeric(original[[value_col]][u_index0 + 1L]))
  if (length(truth) != expected_n || any(!is.finite(truth))) return(NULL)
  truth
}

read_truth_from_predictions <- function(seed_dir, expected_n) {
  candidates <- c(
    file.path(seed_dir, "predictions_y_U.csv"),
    file.path(seed_dir, "predictions_y.csv")
  )
  hit <- candidates[file.exists(candidates)][1]
  if (is.na(hit)) return(NULL)
  dat <- read.csv(hit, stringsAsFactors = FALSE, check.names = FALSE)
  if ("split" %in% names(dat)) dat <- dat[dat$split == "U", , drop = FALSE]
  if (!("true_value" %in% names(dat))) return(NULL)
  truth <- suppressWarnings(as.numeric(dat$true_value))
  if (length(truth) != expected_n || any(!is.finite(truth))) return(NULL)
  truth
}

load_truth_y_U <- function(seed_dir, expected_n, seed, original, orders, train_size) {
  truth <- read_truth_from_split_data(seed_dir, expected_n)
  if (!is.null(truth)) return(truth)
  
  truth <- read_truth_from_original_order(seed, expected_n, original, orders, train_size)
  if (!is.null(truth)) return(truth)
  
  truth <- read_truth_from_predictions(seed_dir, expected_n)
  if (!is.null(truth)) return(truth)
  
  stop(
    "Could not load y_U truth for ", seed_dir, ". Need one of: split_data.csv, ",
    "original data + order table, or predictions_y_U.csv/predictions_y.csv."
  )
}

# ----------------------------
# Across-seed summaries
# ----------------------------
finite_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else mean(x)
}

finite_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2L) NA_real_ else sd(x)
}

make_across_seed_summary <- function(by_seed) {
  group_cols <- c(
    "event", "method", "evaluation_set", "tail", "quantile_probability",
    "threshold", "threshold_abs"
  )
  metric_cols <- c(
    "n_locations", "n_S", "n_U", "event_count", "event_rate",
    "mean_predicted_event_probability", "brier_score",
    "brier_reference_climatology", "brier_skill_score", "crps",
    "coverage_95_all_locations", "ci_width_95_all_locations",
    "twcrps_tail_weighted", "extreme_value_coverage_95",
    "extreme_value_ci_width_95"
  )
  
  keys <- unique(by_seed[group_cols])
  rows <- vector("list", nrow(keys))
  for (i in seq_len(nrow(keys))) {
    mask <- rep(TRUE, nrow(by_seed))
    for (cc in group_cols) mask <- mask & by_seed[[cc]] == keys[[cc]][i]
    sub <- by_seed[mask, , drop = FALSE]
    vals <- lapply(metric_cols, function(cc) finite_mean(sub[[cc]]))
    names(vals) <- metric_cols
    rows[[i]] <- cbind(keys[i, , drop = FALSE], as.data.frame(vals), stringsAsFactors = FALSE)
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

make_across_seed_mean_sd_long <- function(by_seed) {
  group_cols <- c(
    "event", "method", "evaluation_set", "tail", "quantile_probability",
    "threshold", "threshold_abs"
  )
  metric_cols <- setdiff(names(by_seed), c("random_seed", group_cols))
  keys <- unique(by_seed[group_cols])
  rows <- list()
  rr <- 0L
  for (i in seq_len(nrow(keys))) {
    mask <- rep(TRUE, nrow(by_seed))
    for (cc in group_cols) mask <- mask & by_seed[[cc]] == keys[[cc]][i]
    sub <- by_seed[mask, , drop = FALSE]
    for (metric in metric_cols) {
      vals <- sub[[metric]]
      vals_ok <- vals[is.finite(vals)]
      rr <- rr + 1L
      rows[[rr]] <- cbind(
        keys[i, , drop = FALSE],
        metric = metric,
        mean = if (length(vals_ok) == 0L) NA_real_ else mean(vals_ok),
        sd = if (length(vals_ok) < 2L) NA_real_ else sd(vals_ok),
        min = if (length(vals_ok) == 0L) NA_real_ else min(vals_ok),
        max = if (length(vals_ok) == 0L) NA_real_ else max(vals_ok),
        n_available = length(vals_ok),
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

# ----------------------------
# Main
# ----------------------------
probabilities <- parse_probs(PROBS_ARG)
if (any(probabilities <= 0 | probabilities >= 1)) {
  stop("All quantile probabilities must be inside (0, 1).")
}

original <- NULL
orders <- NULL
if (file.exists(ORIGINAL_DATA_CSV)) {
  original <- read.csv(ORIGINAL_DATA_CSV, stringsAsFactors = FALSE, check.names = FALSE)
}
if (file.exists(ORDER_TABLE_CSV)) {
  orders <- read.csv(ORDER_TABLE_CSV, stringsAsFactors = FALSE, check.names = FALSE)
}

specs <- parse_result_specs(RESULT_SPECS)
all_rows <- list()
row_id <- 0L

for (spec in specs) {
  method <- spec$method
  result_dir <- spec$path
  cat("\nMethod: ", method, "\n", sep = "")
  cat("Result dir: ", result_dir, "\n", sep = "")
  
  seed_dirs <- find_seed_dirs(result_dir)
  if (length(seed_dirs) == 0L) stop("No seed directories or sample files found in ", result_dir)
  
  seeds_found <- vapply(seed_dirs, seed_from_dir, integer(1))
  selected_seeds <- parse_seeds(SEEDS_ARG, available = seeds_found[is.finite(seeds_found)])
  if (!is.null(selected_seeds)) {
    keep <- seeds_found %in% selected_seeds
    seed_dirs <- seed_dirs[keep]
    seeds_found <- seeds_found[keep]
  }
  
  if (length(seed_dirs) == 0L) stop("No selected seeds found for ", method)
  
  for (j in seq_along(seed_dirs)) {
    seed_dir <- seed_dirs[[j]]
    seed <- seeds_found[[j]]
    cat("Seed dir: ", seed_dir, "\n", sep = "")
    
    samples <- read_samples_y_U(seed_dir)
    cat("  samples: ", nrow(samples), " locations x ", ncol(samples), " draws\n", sep = "")
    truth <- load_truth_y_U(seed_dir, nrow(samples), seed, original, orders, TRAIN_SIZE)
    
    out_seed_dir <- file.path(OUT_DIR, method, if (is.finite(seed)) paste0("seed_", seed) else basename(seed_dir))
    dir.create(out_seed_dir, recursive = TRUE, showWarnings = FALSE)
    
    seed_metrics <- summarize_one_seed(
      method = method,
      seed = seed,
      truth = truth,
      sample_matrix = samples,
      probabilities = probabilities,
      out_seed_dir = out_seed_dir
    )
    
    # Do not write separate seed-level metric tables by default.
    # The final output is one complete long table with random_seed as a column.
    if (SAVE_LOCATION_DIAGNOSTICS) {
      write.csv(seed_metrics, file.path(out_seed_dir, "tail_event_metrics_y_U.csv"), row.names = FALSE)
    }
    
    row_id <- row_id + 1L
    all_rows[[row_id]] <- seed_metrics
  }
}

summary_table <- do.call(rbind, all_rows)
rownames(summary_table) <- NULL

# One complete long table: random_seed is kept as an identifying column.
summary_cols <- c(
  "random_seed", "event", "method", "evaluation_set", "tail",
  "quantile_probability", "threshold", "threshold_abs",
  "n_locations", "n_S", "n_U", "event_count", "event_rate",
  "mean_predicted_event_probability", "brier_score",
  "brier_reference_climatology", "brier_skill_score", "crps",
  "coverage_95_all_locations", "ci_width_95_all_locations",
  "twcrps_tail_weighted", "extreme_value_coverage_95",
  "extreme_value_ci_width_95"
)
summary_table <- summary_table[, summary_cols, drop = FALSE]
write.csv(summary_table, file.path(OUT_DIR, "tail_event_metric_summary_nngp_nnmp.csv"), row.names = FALSE)

# Optional auxiliary across-seed summaries. These are not the main output.
summary_across_seed_mean <- make_across_seed_summary(summary_table)
write.csv(summary_across_seed_mean, file.path(OUT_DIR, "tail_event_metric_summary_across_seed_mean.csv"), row.names = FALSE)

summary_long <- make_across_seed_mean_sd_long(summary_table)
write.csv(summary_long, file.path(OUT_DIR, "tail_event_metric_summary_across_seed_mean_sd_long.csv"), row.names = FALSE)

cat("\nDone. Main output is one complete table with random_seed column:\n")
cat("  ", normalizePath(file.path(OUT_DIR, "tail_event_metric_summary.csv")), "\n", sep = "")
cat("Auxiliary across-seed summaries were also written:\n")
cat("  ", normalizePath(file.path(OUT_DIR, "tail_event_metric_summary_across_seed_mean.csv")), "\n", sep = "")
cat("  ", normalizePath(file.path(OUT_DIR, "tail_event_metric_summary_across_seed_mean_sd_long.csv")), "\n", sep = "")
