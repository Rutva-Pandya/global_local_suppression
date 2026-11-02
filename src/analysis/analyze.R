#!/usr/bin/env Rscript
# One question per fit; the four named evaluated models are fixed effects.
MODELS <- c("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-72B-Instruct")
LABELS <- c("Llama 3.1 8B Instruct", "Llama 3.1 70B Instruct",
            "Qwen2.5 7B Instruct", "Qwen2.5 72B Instruct")
names(LABELS) <- MODELS
KEYS <- c("scenario_id", "evaluated_model")
# Describes the fit in fit_comparison: nlme::lme with a residual variance per model.
MODEL_FORMULA <- paste("comparison_score ~ 0 + evaluated_model + (1 | scenario_id);",
                       "residual variance: varIdent(form = ~1 | evaluated_model)")

require_that <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

figure_specs <- function() {
  global <- "figure1/global_knowledge/expert_novice"
  local <- "figure2/local_knowledge/use_ignore"
  anchors <- "figure5/theoretical_baseline_anchors/expert_novice"
  judge <- list(
    high_knowledge_minus_baseline = c(high_knowledge = 1, baseline = -1),
    suppression_minus_baseline = c(suppression = 1, baseline = -1),
    high_knowledge_minus_suppression = c(high_knowledge = 1, suppression = -1))
  global_judge <- c(judge, list(
    low_knowledge_minus_baseline = c(low_knowledge = 1, baseline = -1),
    suppression_minus_low_knowledge = c(suppression = 1, low_knowledge = -1)))
  local_lp <- list(high_knowledge = c(fact_provided_response = 1),
    suppression = c(fact_suppression_response = 1),
    high_minus_suppression = c(fact_provided_response = 1, fact_suppression_response = -1))
  global_lp <- list(high_knowledge = c(expert_response = 1),
    low_knowledge = c(novice_response = 1), suppression = c(expert_to_novice_response = 1),
    high_minus_suppression = c(expert_response = 1, expert_to_novice_response = -1),
    suppression_minus_low = c(expert_to_novice_response = 1, novice_response = -1))
  anchor_tests <- list(b_e = c(b_e = 1), b_n = c(b_n = 1), p_s = c(p_s = 1),
    b_e_minus_b_n = c(b_e = 1, b_n = -1),
    b_n_minus_p_s = c(b_n = 1, p_s = -1),
    b_e_minus_p_s = c(b_e = 1, p_s = -1))
  list(
    global_expertise = list(path = global, subdir = "judge", outcome = "expertise_score",
      level = "condition", levels = c("baseline", "high_knowledge", "suppression", "low_knowledge"),
      tests = global_judge, cross = c("high_knowledge_minus_baseline",
        "low_knowledge_minus_baseline", "suppression_minus_baseline")),
    global_delta_logprob = list(path = global, subdir = "delta_logprob", outcome = "delta_logprob",
      level = "response_condition", levels = c("expert_response", "expert_to_novice_response", "novice_response"),
      tests = global_lp, cross = c("high_knowledge", "low_knowledge", "suppression")),
    fact_usage = list(path = local, subdir = "judge", outcome = "fact_usage_score",
      level = "condition", levels = c("baseline", "high_knowledge", "suppression"),
      tests = judge, cross = c("high_knowledge_minus_baseline", "suppression_minus_baseline")),
    local_delta_logprob = list(path = local, subdir = "delta_logprob", outcome = "delta_logprob",
      level = "response_condition", levels = c("fact_provided_response", "fact_suppression_response"),
      tests = local_lp, cross = c("high_knowledge", "suppression")),
    context_anchor_diagnostics = list(path = anchors, subdir = "", outcome = "quantity_value",
      level = "quantity", levels = c("b_e", "b_n", "p_s"), tests = anchor_tests,
      cross = c("b_e", "b_n", "p_s")))
}

row_keys <- function(d, fields) do.call(paste, c(d[fields], sep = "\r"))

read_observations <- function(path, spec) {
  d <- read.csv(path, check.names = FALSE)
  fields <- c(KEYS, spec$level, spec$outcome, "source_file")
  require_that(all(fields %in% names(d)), paste("Missing observation columns:", path))
  require_that(!anyNA(d[fields]) && all(nzchar(d$scenario_id)) && all(nzchar(d$source_file)),
               paste("Blank observation values:", path))
  require_that(is.numeric(d[[spec$outcome]]) && all(is.finite(d[[spec$outcome]])), "Non-finite scores")
  require_that(setequal(unique(d$evaluated_model), MODELS), "Expected four evaluated models")
  require_that(length(unique(d$scenario_id)) == 25L, "Expected 25 scenarios")
  require_that(setequal(unique(d[[spec$level]]), spec$levels), "Unexpected conditions")
  fields <- c(KEYS, spec$level)
  grid <- expand.grid(sort(unique(d$scenario_id)), MODELS, spec$levels, stringsAsFactors = FALSE)
  names(grid) <- fields
  require_that(!anyDuplicated(row_keys(d, fields)) &&
    setequal(row_keys(d, fields), row_keys(grid, fields)), "Incomplete or duplicate observation grid")
  if (spec$subdir == "judge") {
    needed <- c("judge_model", "judge_run", "response_regime", "response_sample_count",
      "judge_sample_count", "n_valid_ratings", "response_score_sd", "response_score_se",
      "response_base_seed", "model_output_source", "suppression_type")
    require_that(all(needed %in% names(d)) && !anyNA(d[needed]), "Missing judge metadata")
    require_that(all(d$judge_model == "openai/gpt-oss-120b") && all(nzchar(d$judge_run)) &&
      all(d$response_regime == "sampled") && all(nzchar(d$model_output_source)), "Unexpected judge source")
    require_that(all(d$response_sample_count == 20) && all(d$judge_sample_count == 5) &&
      all(d$n_valid_ratings == 100), "Expected 20 responses and five valid ratings each")
    require_that(all(d[[spec$outcome]] >= 1 & d[[spec$outcome]] <= 7), "Judge scores outside 1-7")
    require_that(all(is.finite(d$response_score_sd)) && all(d$response_score_sd >= 0) &&
      all(is.finite(d$response_score_se)) &&
      max(abs(d$response_score_se - d$response_score_sd / sqrt(20))) < 1e-10,
      "Inconsistent response SD/SE")
    require_that(length(unique(d$response_base_seed)) == 1 &&
      all(is.finite(d$response_base_seed)) && all(d$response_base_seed == floor(d$response_base_seed)),
      "Expected one integer response base seed")
    expected_scope <- if (spec$outcome == "expertise_score") "global" else "local"
    require_that(all(ifelse(d$condition == "baseline", d$suppression_type == "default_baseline",
                           d$suppression_type == expected_scope)), "Unexpected suppression scope")
  } else {
    needed <- c("scoring_model", "is_self_scored", "n_samples", "source_log_base",
      "value_unit", "model_revision", "scoring_model_revision")
    require_that(all(needed %in% names(d)) && !anyNA(d[needed]), "Missing self-scoring metadata")
    require_that(all(d$evaluated_model == d$scoring_model) &&
      all(tolower(as.character(d$is_self_scored)) == "true"), "Expected self-scoring")
    require_that(all(d$n_samples == 20), "Expected 20 scored responses per cell")
    require_that(all(d$source_log_base == 2) && all(d$value_unit == "bits_per_scored_token"),
                 "Expected LP values in bits per scored token")
    require_that(all(nzchar(d$model_revision)) && all(d$model_revision == d$scoring_model_revision),
                 "Generator/scorer revision mismatch")
    require_that(all(vapply(split(d$model_revision, d$evaluated_model),
      function(x) length(unique(x)) == 1L, logical(1))), "Mixed model revisions")
  }
  d
}

comparison_scores <- function(d, spec, weights) {
  grid <- unique(d[KEYS])
  grid <- grid[do.call(order, grid[KEYS]), ]
  rownames(grid) <- NULL
  grid$comparison_score <- 0
  for (level in names(weights)) {
    values <- d[d[[spec$level]] == level, c(KEYS, spec$outcome)]
    index <- match(row_keys(grid, KEYS), row_keys(values, KEYS))
    require_that(!anyNA(index) && nrow(values) == nrow(grid) &&
      !anyDuplicated(row_keys(values, KEYS)), "Missing or duplicate comparison pair")
    grid$comparison_score <- grid$comparison_score + weights[[level]] * values[[spec$outcome]][index]
  }
  grid
}

subtract_pairs <- function(a, b) {
  index <- match(row_keys(a, KEYS), row_keys(b, KEYS))
  require_that(nrow(a) == nrow(b) && !anyNA(index) &&
    !anyDuplicated(row_keys(a, KEYS)) && !anyDuplicated(row_keys(b, KEYS)), "Critical/control pairing mismatch")
  a$comparison_score <- a$comparison_score - b$comparison_score[index]
  a
}

fit_comparison <- function(d, descriptive_floor = FALSE) {
  require_that(nrow(d) == 100 && !anyDuplicated(row_keys(d, KEYS)) &&
    all(is.finite(d$comparison_score)) && setequal(unique(d$evaluated_model), MODELS) &&
    all(table(d$scenario_id) == 4) && length(unique(d$scenario_id)) == 25, "Invalid comparison grid")
  if (descriptive_floor) {
    require_that(all(d$comparison_score >= 0 & d$comparison_score <= .25),
                 "The declared local-control floor case changed; review before analysis")
    return(list(fit = NULL, tau = NA_real_, error = "", status = "descriptive_floor"))
  }
  d$evaluated_model <- factor(d$evaluated_model, levels = MODELS)
  fit <- tryCatch(withCallingHandlers(nlme::lme(
    comparison_score ~ 0 + evaluated_model, random = ~1 | scenario_id,
    weights = nlme::varIdent(form = ~1 | evaluated_model), data = d, method = "REML",
    control = nlme::lmeControl(opt = "nlminb", maxIter = 200, msMaxIter = 200)),
    warning = function(w) stop(conditionMessage(w), call. = FALSE)), error = function(e) e)
  if (inherits(fit, "error")) return(list(fit = NULL, tau = NA_real_,
    error = conditionMessage(fit), status = "fit_failed"))
  tau <- as.numeric(nlme::getVarCov(fit, type = "random.effects"))
  residual_sd <- fit$sigma * coef(fit$modelStruct$varStruct, unconstrained = FALSE, allCoef = TRUE)
  if (!is.finite(tau) || tau <= 0 || any(!is.finite(residual_sd) | residual_sd <= 0)) {
    return(list(fit = NULL, tau = tau, error = "Non-positive fitted variance", status = "fit_failed"))
  }
  # A boundary scenario variance remains visible and blocks publication for review.
  if (tau / max(tau + residual_sd^2) < 1e-8) return(list(fit = NULL, tau = tau,
    error = "Scenario variance at boundary; review required", status = "fit_failed"))
  list(fit = fit, tau = tau, residual_sd = residual_sd, error = "", status = "ok")
}

summarize_comparison <- function(d, fitted) {
  if (fitted$status == "ok") {
    coefficients <- summary(fitted$fit)$tTable
    fixed_intervals <- nlme::intervals(fitted$fit, which = "fixed", level = .95)$fixed
  }
  rows <- lapply(MODELS, function(m) {
    x <- d$comparison_score[d$evaluated_model == m]
    estimate <- mean(x)
    se <- p <- lo <- hi <- stat <- df <- residual_sd <- NA_real_
    if (fitted$status == "ok") {
      coefficient <- paste0("evaluated_model", m)
      tt <- coefficients[coefficient, ]
      require_that(abs(tt[["Value"]] - estimate) < 1e-8, "Fitted mean differs from direct paired mean")
      estimate <- unname(tt[["Value"]])
      se <- unname(tt[["Std.Error"]])
      df <- unname(tt[["DF"]])
      stat <- unname(tt[["t-value"]])
      p <- unname(tt[["p-value"]])
      lo <- unname(fixed_intervals[coefficient, "lower"])
      hi <- unname(fixed_intervals[coefficient, "upper"])
      residual_sd <- unname(fitted$residual_sd[m])
    }
    reference <- if (sd(x) > 0) t.test(x) else NULL
    data.frame(evaluated_model = m, model_label = LABELS[[m]], estimate = estimate,
      standard_error = se, degrees_of_freedom = df, ci95_low = lo, ci95_high = hi,
      t_statistic = stat, p_value = p,
      significance = if (is.na(p)) "" else if (p < .001) "***" else if (p < .01) "**" else if (p < .05) "*" else "",
      status = fitted$status, n_scenarios = 25, n_models = 4,
      scenario_variance = fitted$tau, residual_sd = residual_sd,
      paired_standard_error = sd(x) / 5,
      paired_p_value = if (is.null(reference)) NA_real_ else reference$p.value,
      model_formula = MODEL_FORMULA, reml = TRUE,
      fixed_effect_inference = "nlme containment", p_value_adjustment = "none",
      confidence_interval_level = .95, error = fitted$error)
  })
  do.call(rbind, rows)
}

write_table <- function(d, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write.csv(d, path, row.names = FALSE, na = "")
}

analyze_figure <- function(name, spec, input_root, output_root, profiles = c("actual", "control")) {
  paths <- setNames(file.path(input_root, spec$path, profiles, spec$subdir, "observations.csv"),
                    profiles)
  raw <- lapply(paths, read_observations, spec = spec)
  if (length(raw) == 2) {
    require_that(setequal(unique(raw$actual$scenario_id), unique(raw$control$scenario_id)),
                 "Critical/control scenario mismatch")
    if (spec$subdir == "judge") {
      require_that(identical(unique(raw$actual$response_base_seed), unique(raw$control$response_base_seed)),
                   "Critical/control response base seeds differ")
    } else {
      revisions <- function(d) d[!duplicated(d$evaluated_model), c("evaluated_model", "model_revision")]
      a <- revisions(raw$actual); b <- revisions(raw$control)
      require_that(all(a$model_revision == b$model_revision[match(a$evaluated_model, b$evaluated_model)]),
                   "Critical/control revisions differ")
    }
    if (name == "global_expertise") {
      a <- comparison_scores(raw$actual, spec, c(baseline = 1))
      b <- comparison_scores(raw$control, spec, c(baseline = 1))
      require_that(all(subtract_pairs(a, b)$comparison_score == 0), "Global baseline is not shared")
    }
  }
  scores <- list(); result <- list()
  jobs <- c(lapply(names(raw), function(profile) lapply(names(spec$tests), function(id)
    list(profile = profile, id = id))), list(lapply(if (length(raw) == 2) spec$cross else character(), function(id)
      list(profile = "actual_vs_control", id = id))))
  jobs <- do.call(c, jobs)
  for (job in jobs) {
    weights <- spec$tests[[job$id]]
    if (job$profile == "actual_vs_control") {
      d <- subtract_pairs(comparison_scores(raw$actual, spec, weights),
                          comparison_scores(raw$control, spec, weights))
    } else d <- comparison_scores(raw[[job$profile]], spec, weights)
    floor_case <- name == "fact_usage" && job$profile == "control" &&
      job$id == "suppression_minus_baseline"
    fit <- fit_comparison(d, floor_case)
    rows <- summarize_comparison(d, fit)
    rows$figure <- name; rows$profile <- job$profile; rows$contrast_id <- job$id
    control <- raw$control
    novice_verified <- !is.null(control) &&
      "novice_wording_status" %in% names(control) &&
      all(!is.na(control$novice_wording_status) &
          control$novice_wording_status == "true_novice_verified")
    low_control <- job$profile != "actual" &&
      ((name == "global_expertise" && "low_knowledge" %in% names(weights)) ||
       name == "global_delta_logprob" ||
       (name == "context_anchor_diagnostics" &&
        any(names(weights) %in% c("b_n", "p_s"))))
    rows$interpretation_status <- if (low_control && !novice_verified)
      "control_low_wording_unresolved" else "as_defined"
    if (spec$subdir == "judge") {
      sources <- if (job$profile == "actual_vs_control") raw else raw[job$profile]
      rows$judge_model <- "openai/gpt-oss-120b"; rows$n_judges <- 1
      rows$judge_run <- paste(unique(unlist(lapply(sources, function(x) x$judge_run))), collapse = ";")
    } else {
      rows$scoring_model <- rows$evaluated_model
      rows$is_self_scored <- TRUE; rows$n_scoring_models <- 4
      rows$source_log_base <- 2; rows$value_unit <- "bits_per_scored_token"
    }
    result[[length(result) + 1L]] <- rows
    scores[[length(scores) + 1L]] <- data.frame(figure = name, profile = job$profile,
      contrast_id = job$id, d)
  }
  result <- do.call(rbind, result)
  folder <- file.path(output_root, name)
  write_table(result, file.path(folder, "all_contrasts.csv"))
  for (profile in unique(result$profile)) write_table(result[result$profile == profile, ],
    file.path(folder, profile, "contrasts.csv"))
  diagnostics <- result[c("figure", "profile", "contrast_id", "evaluated_model", "status",
    "scenario_variance", "residual_sd", "degrees_of_freedom", "error", "model_formula", "reml", "fixed_effect_inference")]
  write_table(diagnostics, file.path(folder, "diagnostics.csv"))
  write_table(do.call(rbind, scores), file.path(folder, "comparison_observations.csv"))
  current <- list()
  for (profile in names(raw)) {
    path <- file.path(dirname(paths[[profile]]), "contrasts.csv")
    if (!file.exists(path)) next
    old <- read.csv(path)
    fields <- c("contrast_id", "evaluated_model", "estimate", "ci95_low", "ci95_high", "p_value")
    old <- old[fields]; old$profile <- profile
    current[[profile]] <- old
  }
  if (length(current)) {
    comparison <- merge(result, do.call(rbind, current),
      by = c("profile", "contrast_id", "evaluated_model"), suffixes = c("", "_current"))
    comparison$significance_changed <- (comparison$p_value < .05) != (comparison$p_value_current < .05)
    write_table(comparison, file.path(folder, "current_vs_new.csv"))
  }
  inputs <- paths
  write_table(data.frame(path = unname(inputs), sha256 = vapply(inputs, function(p)
    digest::digest(file = p, algo = "sha256", serialize = FALSE), character(1))),
    file.path(folder, "input_hashes.csv"))
  require_that(all(result$status %in% c("ok", "descriptive_floor")),
    paste("A fit failed; inspect", file.path(folder, "diagnostics.csv")))
  cat(name, ":", nrow(result), "rows,", sum(result$status == "ok"), "inferential\n")
  invisible(result)
}

resolved_output_path <- function(path) {
  if (file.exists(path) || dir.exists(path)) return(normalizePath(path))
  parent <- dirname(path)
  require_that(parent != path, "Cannot resolve output directory")
  file.path(resolved_output_path(parent), basename(path))
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if ("--help" %in% args) {
    cat("Paper: --output-root DIR [--input-root analysis/paper] [--figures comma,separated] [--datasets critical,control]\n",
        "Supplemental: gap --output-root DIR [--input-root analysis/paper]\n")
    return(invisible(NULL))
  }
  require_that(length(args) > 0 && length(args) %% 2 == 0, "Expected --flag value pairs")
  require_that(!anyDuplicated(args[seq(1, length(args), by = 2)]),
               "Expected unique --flag value pairs")
  flags <- args[seq(1, length(args), by = 2)]
  require_that(all(flags %in% c("--output-root", "--input-root", "--figures", "--datasets")),
               "Unknown argument")
  opts <- setNames(as.list(args[seq(2, length(args), by = 2)]), flags)
  require_that(!is.null(opts[["--output-root"]]), "--output-root is required")
  input <- if (is.null(opts[["--input-root"]])) "analysis/paper" else opts[["--input-root"]]
  output <- opts[["--output-root"]]
  abs_output <- resolved_output_path(output)
  canonical <- normalizePath("analysis/paper")
  require_that(abs_output != canonical && !startsWith(abs_output, paste0(canonical, "/")),
               "Preview analysis cannot overwrite canonical analysis/paper")
  require_that(!dir.exists(output) || !length(list.files(output, all.files = TRUE, no.. = TRUE)),
               "Output directory must be empty; choose a new preview directory")
  specs <- figure_specs()
  selected <- if (is.null(opts[["--figures"]])) names(specs) else strsplit(opts[["--figures"]], ",", fixed = TRUE)[[1]]
  require_that(length(selected) > 0 && !anyDuplicated(selected) && all(selected %in% names(specs)), "Unknown or duplicate figure")
  datasets <- if (is.null(opts[["--datasets"]])) c("critical", "control") else
    strsplit(opts[["--datasets"]], ",", fixed = TRUE)[[1]]
  require_that(length(datasets) > 0 && !anyDuplicated(datasets) &&
    all(datasets %in% c("critical", "control")), "Unknown or duplicate dataset")
  profiles <- ifelse(datasets == "critical", "actual", datasets)
  results <- lapply(selected, function(name) analyze_figure(name, specs[[name]], input, output, profiles = profiles))
  fields <- c("figure", "profile", "contrast_id", "evaluated_model", "estimate", "standard_error",
    "degrees_of_freedom", "ci95_low", "ci95_high", "p_value", "status", "interpretation_status")
  all_results <- do.call(rbind, lapply(results, function(d) d[fields]))
  write_table(all_results, file.path(output, "all_contrasts.csv"))
  write_table(all_results[all_results$status == "descriptive_floor", ], file.path(output, "floor_cases.csv"))
  write_table(all_results[all_results$interpretation_status != "as_defined", ],
              file.path(output, "interpretation_flags.csv"))
  summaries <- changes <- list()
  for (name in selected) {
    path <- file.path(output, name, "current_vs_new.csv")
    if (!file.exists(path)) next
    d <- read.csv(path)
    finite <- is.finite(d$p_value) & is.finite(d$p_value_current)
    summaries[[name]] <- data.frame(figure = name, comparable_tests = sum(finite),
      significance_changes = sum(d$significance_changed, na.rm = TRUE),
      max_estimate_change = max(abs(d$estimate - d$estimate_current)))
    changes[[name]] <- d[finite & !is.na(d$significance_changed) & d$significance_changed,
      c(fields, "p_value_current", "ci95_low_current", "ci95_high_current")]
  }
  if (length(summaries)) {
    write_table(do.call(rbind, summaries), file.path(output, "audit_summary.csv"))
    write_table(do.call(rbind, changes), file.path(output, "changed_conclusions.csv"))
  }
  writeLines(capture.output(sessionInfo()), file.path(output, "session_info.txt"))
  source <- "src/analysis/analyze.R"
  write_table(data.frame(path = source, sha256 = digest::digest(file = source, algo = "sha256", serialize = FALSE)),
              file.path(output, "analysis_source_hash.csv"))
  writeLines(c("complete", paste(selected, collapse = ","), "REML; model-specific residual variance; nlme containment; no adjustment"),
              file.path(output, "COMPLETE"))
}



# Supplemental contrast uses the same pairing and fitter as the main analysis.
gap_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (identical(args, "--help")) {
    cat("--output-root DIR [--input-root analysis/paper]\n")
    return(invisible(NULL))
  }
  require_that(length(args) > 0 && length(args) %% 2 == 0,
               "Expected --flag value pairs")
  flags <- args[seq(1, length(args), by = 2)]
  require_that(!anyDuplicated(flags) &&
    all(flags %in% c("--input-root", "--output-root")), "Unknown or duplicate argument")
  opts <- setNames(as.list(args[seq(2, length(args), by = 2)]), flags)
  require_that(!is.null(opts[["--output-root"]]), "--output-root is required")
  require_that(all(nzchar(unlist(opts))), "Directory arguments cannot be blank")
  fitter_path <- "src/analysis/analyze.R"
  input <- if (is.null(opts[["--input-root"]])) "analysis/paper" else opts[["--input-root"]]
  output <- resolved_output_path(opts[["--output-root"]])
  require_that(!any(strsplit(output, "/", fixed = TRUE)[[1]] %in% c(".", "..")),
               "Output cannot contain unresolved dot segments")
  repo <- dirname(dirname(dirname(normalizePath(fitter_path))))
  protected <- vapply(c(input, file.path(repo, c("analysis/paper", "figures",
    "analysis/model_comparison/global_suppression_gap"))),
    resolved_output_path, character(1))
  overlaps <- function(path) output == path || startsWith(output, paste0(path, "/")) ||
    startsWith(path, paste0(output, "/"))
  require_that(!any(vapply(protected, overlaps, logical(1))),
               "Output cannot overlap input or canonical artifacts")
  require_that(!file.exists(output) || (dir.exists(output) &&
    !length(list.files(output, all.files = TRUE, no.. = TRUE))),
    "Output must be a new or empty directory")

  spec <- figure_specs()$global_delta_logprob
  paths <- setNames(file.path(input, spec$path, c("actual", "control"),
                              "delta_logprob/observations.csv"), c("actual", "control"))
  raw <- lapply(paths, read_observations, spec = spec)
  require_that("novice_wording_status" %in% names(raw$control) &&
    all(!is.na(raw$control$novice_wording_status) &
        raw$control$novice_wording_status == "true_novice_verified"),
    "Control novice wording is not fully verified")
  for (m in MODELS) require_that(
    identical(unique(raw$actual$model_revision[raw$actual$evaluated_model == m]),
              unique(raw$control$model_revision[raw$control$evaluated_model == m])),
    "Critical/control model revisions differ")

  weights <- spec$tests$suppression_minus_low
  gaps <- lapply(raw, comparison_scores, spec = spec, weights = weights)
  scores <- subtract_pairs(gaps$control, gaps$actual)
  scores$control_gap <- gaps$control$comparison_score
  index <- match(row_keys(scores, KEYS), row_keys(gaps$actual, KEYS))
  scores$critical_gap <- gaps$actual$comparison_score[index]
  require_that(all(abs(scores$comparison_score -
    (scores$control_gap - scores$critical_gap)) < 1e-12), "Contrast sign mismatch")

  fitted <- fit_comparison(scores)
  require_that(fitted$status == "ok", paste("Fit blocked:", fitted$error))
  result <- summarize_comparison(scores, fitted)
  result$contrast_id <- "control_minus_critical_suppression_minus_low"
  result$value_unit <- "bits_per_scored_token"
  result$interpretation_status <- "as_defined"

  # Record the inputs and both current implementation files, without old fits.
  sources <- c(unname(paths), fitter_path)
  hashes <- data.frame(path = sources, sha256 = vapply(sources, function(p)
    digest::digest(file = p, algo = "sha256", serialize = FALSE), character(1)))
  write_table(scores, file.path(output, "comparison_observations.csv"))
  write_table(result, file.path(output, "contrasts.csv"))
  writeLines(capture.output({
    print(summary(fitted$fit))
    print(nlme::VarCorr(fitted$fit))
    print(summary(residuals(fitted$fit, type = "normalized")))
  }), file.path(output, "diagnostics.txt"))
  writeLines(capture.output(sessionInfo()), file.path(output, "session_info.txt"))
  write_table(hashes, file.path(output, "source_hashes.csv"))
  writeLines(c("complete", "control_minus_critical_suppression_minus_low",
    "REML; model-specific residual variance; nlme containment; no adjustment"),
    file.path(output, "COMPLETE"))
  print(result[c("model_label", "estimate", "ci95_low", "ci95_high",
                 "degrees_of_freedom", "t_statistic", "p_value", "paired_p_value")],
        row.names = FALSE)
  cat("PASS: 100 paired scenario/model cells\n")
}



if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) && args[1] == "gap") gap_main(args[-1]) else main(args)
}
