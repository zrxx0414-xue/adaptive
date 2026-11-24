# Linear regression simulation and model selection.

structure_levels <- c("Additional main effects", "Additional quadratic term", "Additional interaction term")

truth_levels <- c("M1", "M2")

standardised_random <- function(n, distribution = "Normal") {
    values <- switch(distribution,
        "Normal" = rnorm(n), "Uniform" = runif(n, -sqrt(3), sqrt(3)),
        "Heavy-tailed t(3)" = rt(n, df = 3) / sqrt(3),
        "Bimodal" = (sample(c(-1, 1), n, replace = TRUE) + rnorm(n, sd = 0.5)) / sqrt(1.25),
        "Right-skewed" = rexp(n) - 1, stop("Unknown distribution: ", distribution))
    as.numeric(scale(values))
}

generate_predictors <- function(n, distribution = "Normal") {
    data.frame(x1 = standardised_random(n, distribution), x2 = standardised_random(n, distribution),
               x3 = standardised_random(n, distribution))
}

shared_signal <- function(data, structure) {
    switch(structure,
        "Additional main effects" = 1 + data$x1,
        "Additional quadratic term" = 1 + data$x1,
        "Additional interaction term" = 1 + data$x1 + data$x2)
}

additional_component <- function(data, structure) {
    switch(structure,
        "Additional main effects" = data$x2 + data$x3,
        "Additional quadratic term" = data$x1^2 - 1,
        "Additional interaction term" = data$x1 * data$x2)
}

# With an explicit effect size, coefficient scales only the shared signal.
generate_signal <- function(data, structure, truth, coefficient = 1, additional_effect_size = NULL) {
    extra <- if (truth == "M2") additional_component(data, structure) else 0
    shared <- shared_signal(data, structure) - 1
    if (is.null(additional_effect_size))
        return(1 + coefficient * (shared + extra))
    1 + coefficient * shared + additional_effect_size * extra
}

# Calibrate on the training design; apply the SAME parameters to the test design.
generate_experiment <- function(x_train, x_test, structure, truth, snr, snr_by,
                                coefficient = 1, predictor_variance = 1, error_variance = 1,
                                error_distribution = "Normal", additional_effect_size = NULL) {
    signal_at <- function(v) {
        generate_signal(x_train * sqrt(v), structure, truth, coefficient, additional_effect_size)
    }
    if (snr_by == "error_variance") {
        error_variance <- var(signal_at(predictor_variance)) / snr
    } else if (snr_by == "coefficient" && !is.null(additional_effect_size) && truth == "M2") {
        design <- x_train * sqrt(predictor_variance)
        shared <- shared_signal(design, structure)
        extra <- additional_effect_size * additional_component(design, structure)
        shared_variance <- var(shared)
        # Complete the square in Var(coefficient * shared + extra).
        offset <- cov(shared, extra) / shared_variance
        residual_variance <- var(extra - offset * shared)
        squared_distance <- (snr * error_variance - residual_variance) / shared_variance
        coefficient <- sqrt(max(0, squared_distance)) - offset
        if (!is.finite(coefficient) || squared_distance < 0 || coefficient <= 0)
            stop("The target snr is unattainable with this additional_effect_size and a positive shared coefficient.")
    } else if (snr_by == "coefficient") {
        unit_signal <- generate_signal(x_train * sqrt(predictor_variance), structure, truth, 1)
        coefficient <- sqrt(snr * error_variance / var(unit_signal))
    } else {
        # Nonlinear terms mean signal variance is not simply proportional to Var(X).
        target <- snr * error_variance
        upper <- 1
        while (var(signal_at(upper)) < target) upper <- 2 * upper
        predictor_variance <- uniroot(function(v) var(signal_at(v)) - target,
                                     interval = c(0, upper), tol = 1e-10)$root
    }
    train <- x_train * sqrt(predictor_variance)
    test <- x_test * sqrt(predictor_variance)
    mu <- generate_signal(train, structure, truth, coefficient, additional_effect_size)
    train$y <- mu + sqrt(error_variance) * standardised_random(nrow(train), error_distribution)
    test$y <- generate_signal(test, structure, truth, coefficient, additional_effect_size)
    parameters <- data.frame(coefficient = coefficient, predictor_variance = predictor_variance,
                             error_variance = error_variance, train_signal_variance = var(mu),
                             achieved_snr = var(mu) / error_variance)
    list(train = train, test = test, parameters = parameters)
}

fit_candidate_models <- function(data, structure) {
    switch(structure,
        "Additional main effects" = list(M1 = lm(y ~ x1, data), M2 = lm(y ~ x1 + x2 + x3, data)),
        "Additional quadratic term" = list(M1 = lm(y ~ x1, data), M2 = lm(y ~ x1 + I(x1^2), data)),
        "Additional interaction term" = list(M1 = lm(y ~ x1 + x2, data), M2 = lm(y ~ x1 * x2, data)))
}

candidate_parameter_count <- function(structure) {
    switch(structure,
        "Additional main effects" = c(M1 = 2L, M2 = 4L),
        "Additional quadratic term" = c(M1 = 2L, M2 = 3L),
        "Additional interaction term" = c(M1 = 3L, M2 = 4L))
}
