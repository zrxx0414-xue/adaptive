source("drm_selection.R")
source("data_generating.R")
source("benchmark_aic_bic.R")
source("adaptive_basis.R")
library(dplyr)

run_linear_simulation <- function(structure = "Additional main effects", truth = "M2", snr = 1,
    n_train = 200L, n_test = 200L, repetitions = 500L, basis = "adaptive",
    kernel_bw = "adaptive", dim_basis = "adaptive", predictor_distribution = "Normal",
    error_distribution = "Normal", test_seed = 4100L, training_seed = 42L, alpha = 0.05, maxit = 2000L,
    additional_effect_size = NULL, save_simulated_data = FALSE) {
    truth <- match.arg(truth, truth_levels)
    structure <- match.arg(structure, structure_levels)
    if (!is.numeric(snr) || length(snr) != 1L || !is.finite(snr) || snr <= 0)
        stop("snr must be a positive finite number.")
    if (!is.null(additional_effect_size) &&
        (!is.numeric(additional_effect_size) || length(additional_effect_size) != 1L || !is.finite(additional_effect_size)))
        stop("additional_effect_size must be a finite number.")
    if (!is.logical(save_simulated_data) || length(save_simulated_data) != 1L || is.na(save_simulated_data))
        stop("save_simulated_data must be TRUE or FALSE.")
    simulation_start <- proc.time()[["elapsed"]]
    complexity <- candidate_parameter_count(structure)
    set.seed(test_seed)
    x_test <- generate_predictors(n_test, predictor_distribution)
    set.seed(training_seed)
    results <- parameters <- vector("list", repetitions)
    if (save_simulated_data)
        simulated_data <- vector("list", repetitions)

    for (i in seq_len(repetitions)) {
        x_train <- generate_predictors(n_train, predictor_distribution)
        data <- generate_experiment(x_train, x_test, structure, truth, snr, snr_by = "error_variance",
            error_distribution = error_distribution, additional_effect_size = additional_effect_size)
        parameters[[i]] <- data$parameters
        if (save_simulated_data)
            simulated_data[[i]] <- list(train = data$train, test = data$test)
        models <- fit_candidate_models(data$train, structure)
        prediction_1 <- as.numeric(predict(models$M1, newdata = data$test))
        prediction_2 <- as.numeric(predict(models$M2, newdata = data$test))
        drm_start <- proc.time()[["elapsed"]]
        drm <- safe_drm_model_selection(data$test$y, prediction_1, prediction_2, basis,
            complexity = complexity, alpha = alpha, maxit = maxit, kernel_bw = kernel_bw, dim_basis = dim_basis)
        drm$runtime_seconds <- proc.time()[["elapsed"]] - drm_start
        drm$method <- "DRM"
        benchmarks <- benchmark_aic_bic(models)
        reference <- data.frame(method = names(benchmarks), decision = unname(benchmarks), valid = TRUE)
        results[[i]] <- mutate(bind_rows(drm, reference), replicate = i)
    }

    parameters <- bind_rows(parameters) |> mutate(replicate = row_number())
    raw <- bind_rows(results) |> left_join(parameters, by = "replicate") |>
        mutate(structure = structure, truth = truth, snr = snr,
               n_train = n_train, n_test = n_test, error_sd = sqrt(error_variance), correct = decision == truth)
    summary <- raw |> group_by(method) |>
        summarise(valid_runs = sum(valid), invalid_runs = sum(!valid),
                  correct_selections = sum(correct, na.rm = TRUE),
                  correct_rate = if (any(valid)) mean(correct[valid]) else NA_real_, .groups = "drop")
    failures <- raw |> filter(method == "DRM") |>
        mutate(outcome = case_when(!valid ~ "Invalid DRM fit", correct ~ "Correct",
                                  !rejected ~ "Failed to reject equality", rejected ~ "Wrong Wald direction")) |>
        count(outcome, name = "runs") |> mutate(proportion = runs / sum(runs))
    error_summary <- raw |> filter(method == "DRM", !valid) |>
        count(failure_stage, error_type, error_message, name = "runs")
    settings <- list(structure = structure, truth = truth, snr = snr,
        coefficient = 1, predictor_variance = 1,
        snr_definition = "Full signal sample variance / error sample variance, conditional on training design",
        test_response = "Noiseless true signal; same DGP parameters as the corresponding training sample",
        n_train = n_train, n_test = n_test, repetitions = repetitions, basis = basis,
        kernel_bw = kernel_bw, dim_basis = dim_basis, predictor_distribution = predictor_distribution,
        error_distribution = error_distribution, test_seed = test_seed, training_seed = training_seed,
        alpha = alpha, maxit = maxit)
    if (!is.null(additional_effect_size))
        settings$additional_effect_size <- additional_effect_size
    runtime <- list(total_seconds = proc.time()[["elapsed"]] - simulation_start,
                    drm_total_seconds = sum(raw$runtime_seconds[raw$method == "DRM"]),
                    drm_mean_seconds = mean(raw$runtime_seconds[raw$method == "DRM"]))
    result <- list(settings = settings, parameters = parameters, raw = raw, summary = summary,
         drm_outcomes = failures, error_summary = error_summary, runtime = runtime)
    if (save_simulated_data)
        result$simulated_data <- simulated_data
    result
}
