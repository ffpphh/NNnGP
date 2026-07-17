# Shared helper functions for Montana 800m split-seed R baselines.

standardize_safe <- function(x) {
  x <- as.numeric(x)
  sx <- sd(x, na.rm = TRUE)
  mx <- mean(x, na.rm = TRUE)
  if (!is.finite(sx) || sx == 0) return(rep(0, length(x)))
  (x - mx) / sx
}

rmse <- function(truth, estimate) sqrt(mean((truth - estimate)^2, na.rm = TRUE))
rsr <- function(truth, estimate) {
  ok <- is.finite(truth) & is.finite(estimate)
  truth <- truth[ok]; estimate <- estimate[ok]
  if (length(truth) < 2L || !is.finite(sd(truth)) || sd(truth) == 0) return(NA_real_)
  rmse(truth, estimate) / sd(truth)
}
row_sd <- function(sample_matrix) apply(as.matrix(sample_matrix), 1, sd)
row_quantiles <- function(sample_matrix) {
  t(apply(as.matrix(sample_matrix), 1, quantile, probs = c(0.025, 0.5, 0.975), names = FALSE))
}

crps_ensemble_rows <- function(truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  if (nrow(sample_matrix) != length(truth)) stop("CRPS: truth length mismatch.")
  n_draws <- ncol(sample_matrix)
  if (n_draws < 2L) stop("CRPS requires at least two posterior predictive draws.")
  weights <- 2 * seq_len(n_draws) - n_draws - 1
  vapply(seq_len(nrow(sample_matrix)), function(i) {
    draws <- as.numeric(sample_matrix[i, ])
    first_term <- mean(abs(draws - truth[i]))
    sorted_draws <- sort(draws)
    second_term <- sum(weights * sorted_draws) / (n_draws^2)
    first_term - second_term
  }, numeric(1))
}

make_prediction_table <- function(method, seed, data_frame, truth, sample_matrix) {
  sample_matrix <- as.matrix(sample_matrix)
  qq <- row_quantiles(sample_matrix)
  pred_mean <- rowMeans(sample_matrix)
  pred_sd <- row_sd(sample_matrix)
  crps <- crps_ensemble_rows(truth, sample_matrix)
  out <- data.frame(
    method = method,
    random_seed = seed,
    split = as.character(data_frame$split),
    point_index = data_frame$point_index,
    row_order = data_frame$row_order,
    lon = data_frame$lon,
    lat = data_frame$lat,
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
  out[order(out$point_index), , drop = FALSE]
}

make_compact_metric_row <- function(method, seed, split, result_table) {
  data.frame(
    method = method,
    random_seed = seed,
    split = split,
    n_locations = nrow(result_table),
    RMSPE = rmse(result_table$true_value, result_table$pred_mean),
    RSR = rsr(result_table$true_value, result_table$pred_mean),
    CRPS = mean(result_table$crps, na.rm = TRUE),
    CI_coverage_percent = 100 * mean(
      result_table$true_value >= result_table$pred_q025 &
        result_table$true_value <= result_table$pred_q975,
      na.rm = TRUE
    ),
    CI_width = mean(result_table$ci_width_95, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

safe_auc <- function(labels, scores) {
  ok <- is.finite(scores) & !is.na(labels)
  labels <- as.integer(labels[ok]); scores <- scores[ok]
  n_pos <- sum(labels == 1L); n_neg <- sum(labels == 0L)
  if (n_pos == 0L || n_neg == 0L) return(NA_real_)
  ranks <- rank(scores, ties.method = "average")
  (sum(ranks[labels == 1L]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

summarize_tail_events <- function(method, seed, split_name, truth, sample_matrix,
                                  quantile_probs = c(0.90, 0.95, 0.975, 0.99),
                                  probability_cutoff = 0.5) {
  sample_matrix <- as.matrix(sample_matrix)
  out <- list()
  eps <- 1e-12
  row_id <- 1L
  for (prob in quantile_probs) {
    magnitude <- qnorm(prob)
    for (tail in c("lower", "upper")) {
      threshold <- if (tail == "lower") -magnitude else magnitude
      if (tail == "lower") {
        truth_event <- truth <= threshold
        pred_prob <- rowMeans(sample_matrix <= threshold)
      } else {
        truth_event <- truth >= threshold
        pred_prob <- rowMeans(sample_matrix >= threshold)
      }
      pred_event <- pred_prob >= probability_cutoff
      tp <- sum(pred_event & truth_event, na.rm = TRUE)
      fp <- sum(pred_event & !truth_event, na.rm = TRUE)
      fn <- sum(!pred_event & truth_event, na.rm = TRUE)
      tn <- sum(!pred_event & !truth_event, na.rm = TRUE)
      precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
      recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
      f1 <- if (is.finite(precision) && is.finite(recall) && (precision + recall) > 0) {
        2 * precision * recall / (precision + recall)
      } else NA_real_
      pred_prob_clip <- pmin(pmax(pred_prob, eps), 1 - eps)
      brier <- mean((pred_prob - as.numeric(truth_event))^2, na.rm = TRUE)
      log_score <- -mean(
        as.numeric(truth_event) * log(pred_prob_clip) +
          (1 - as.numeric(truth_event)) * log(1 - pred_prob_clip),
        na.rm = TRUE
      )
      out[[row_id]] <- data.frame(
        method = method,
        random_seed = seed,
        split = split_name,
        quantile_probability = prob,
        tail = tail,
        threshold = threshold,
        n_locations = length(truth),
        event_count = sum(truth_event, na.rm = TRUE),
        event_rate_percent = 100 * mean(truth_event, na.rm = TRUE),
        mean_predicted_event_probability_percent = 100 * mean(pred_prob, na.rm = TRUE),
        predicted_event_rate_percent = 100 * mean(pred_event, na.rm = TRUE),
        brier_score = brier,
        log_score = log_score,
        auc = safe_auc(truth_event, pred_prob),
        accuracy = (tp + tn) / (tp + fp + fn + tn),
        precision = precision,
        recall = recall,
        f1 = f1,
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
  do.call(rbind, out)
}

summarise_across_seeds <- function(df, id_cols = c("method", "split")) {
  numeric_cols <- names(df)[vapply(df, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, "random_seed")
  groups <- unique(df[id_cols])
  rows <- list()
  rr <- 1L
  for (i in seq_len(nrow(groups))) {
    idx <- rep(TRUE, nrow(df))
    for (cc in id_cols) idx <- idx & df[[cc]] == groups[[cc]][i]
    sub <- df[idx, , drop = FALSE]
    base <- groups[i, , drop = FALSE]
    for (col in numeric_cols) {
      vals <- sub[[col]]
      rows[[rr]] <- cbind(
        base,
        metric = col,
        mean = mean(vals, na.rm = TRUE),
        sd = sd(vals, na.rm = TRUE),
        min = min(vals, na.rm = TRUE),
        max = max(vals, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
      rr <- rr + 1L
    }
  }
  do.call(rbind, rows)
}

read_montana_inputs <- function(original_csv, order_csv) {
  original <- read.csv(original_csv, stringsAsFactors = FALSE, check.names = FALSE)
  orders <- read.csv(order_csv, stringsAsFactors = FALSE, check.names = FALSE)
  req_original <- c("lon", "lat", "log_ppt_2025_standardized")
  miss_original <- setdiff(req_original, names(original))
  if (length(miss_original) > 0L) stop("Original data missing columns: ", paste(miss_original, collapse = ", "))
  if (!("random_seed" %in% names(orders))) stop("Order table must contain random_seed.")
  original$lon <- as.numeric(original$lon)
  original$lat <- as.numeric(original$lat)
  original$y_obs <- as.numeric(original$log_ppt_2025_standardized)
  if (any(!is.finite(original$lon)) || any(!is.finite(original$lat)) || any(!is.finite(original$y_obs))) {
    stop("Original data contains non-finite lon/lat/y values.")
  }
  list(original = original, orders = orders)
}

make_seed_split <- function(original, orders, seed, train_size = 800L) {
  row <- orders[orders$random_seed == seed, , drop = FALSE]
  if (nrow(row) != 1L) stop("Could not find unique order row for seed ", seed)
  order_cols <- setdiff(names(orders), "random_seed")
  point_order <- as.integer(row[1, order_cols])
  n <- nrow(original)
  if (length(point_order) != n || !identical(sort(point_order), 0:(n - 1L))) {
    stop("Point order for seed ", seed, " is not a zero-based permutation of original row indices.")
  }
  dat <- original[point_order + 1L, , drop = FALSE]
  dat$point_index <- point_order
  dat$row_order <- seq_len(nrow(dat)) - 1L
  dat$split <- ifelse(dat$row_order < train_size, "S", "U")
  dat$x <- dat$lon
  dat$y <- dat$lat
  dat$x0 <- 1
  # Standardize covariates within this split-ordered full data, matching the Python split CSV workflow.
  dat$x1 <- standardize_safe(dat$lon)
  dat$x2 <- standardize_safe(abs(dat$lat))
  dat$w <- NA_real_
  dat[, c("row_order", "point_index", "split", "x", "y", "w", "y_obs", "x0", "x1", "x2", "lon", "lat", "log_ppt_2025_standardized")]
}

save_samples <- function(sample_matrix, path_prefix, save_csv = FALSE) {
  saveRDS(sample_matrix, paste0(path_prefix, ".rds"), compress = "xz")
  if (isTRUE(save_csv)) {
    write.csv(sample_matrix, paste0(path_prefix, ".csv"), row.names = FALSE)
  }
}


summarize_vector <- function(samples, parameter_name) {
  samples <- as.numeric(samples)
  data.frame(
    parameter = parameter_name,
    mean = mean(samples, na.rm = TRUE),
    sd = sd(samples, na.rm = TRUE),
    q025 = unname(quantile(samples, 0.025, na.rm = TRUE)),
    median = unname(quantile(samples, 0.5, na.rm = TRUE)),
    q975 = unname(quantile(samples, 0.975, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
}

flatten_numeric_object <- function(x, prefix = "") {
  output <- list()
  recurse <- function(value, current_name) {
    if (is.null(value)) return(invisible(NULL))
    if (is.list(value)) {
      child_names <- names(value)
      if (is.null(child_names)) child_names <- paste0("item_", seq_along(value))
      for (i in seq_along(value)) {
        child_name <- if (nzchar(current_name)) paste0(current_name, ".", child_names[i]) else child_names[i]
        recurse(value[[i]], child_name)
      }
      return(invisible(NULL))
    }
    value_vector <- as.vector(value)
    if (length(value_vector) == 0L) return(invisible(NULL))
    item_names <- names(value_vector)
    if (is.null(item_names) || any(!nzchar(item_names))) {
      item_names <- if (length(value_vector) == 1L) current_name else paste0(current_name, "[", seq_along(value_vector), "]")
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
    return(data.frame(item = character(), value = numeric(), note = character(), stringsAsFactors = FALSE))
  }
  do.call(rbind, output)
}
