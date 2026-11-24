# Linear regression simulation and model selection.

structure_levels <- c("Additional main effects", "Additional quadratic term", "Additional interaction term")

truth_levels <- c("M1", "M2")
#distribution
standardised_random <- function(n, distribution = "Normal") {
    values <- switch(distribution,
        "Normal" = rnorm(n), "Uniform" = runif(n, -sqrt(3), sqrt(3)),
        "Heavy-tailed t(3)" = rt(n, df = 3) / sqrt(3),
        "Bimodal" = (sample(c(-1, 1), n, replace = TRUE) + rnorm(n, sd = 0.5)) / sqrt(1.25),
        "Right-skewed" = rexp(n) - 1, stop("Unknown distribution: ", distribution))
    as.numeric(scale(values))
}

#generate x
generate_predictors <- function(n, distribution = "Normal") {
    data.frame(x1 = standardised_random(n, distribution), x2 = standardised_random(n, distribution),
               x3 = standardised_random(n, distribution))
}
#M1
shared_signal <- function(data, structure) {
    switch(structure,
        "Additional main effects" = 1 + data$x1,
        "Additional quadratic term" = 1 + data$x1,
        "Additional interaction term" = 1 + data$x1 + data$x2)
}
#additional in M2
additional_component <- function(data, structure) {
    switch(structure,
        "Additional main effects" = data$x2 + data$x3,
        "Additional quadratic term" = data$x1^2 - 1,
        "Additional interaction term" = data$x1 * data$x2)
}

# With an explicit effect size, coefficient scales only the shared signal.
generate_signal <- function(data, structure, truth, additional_effect_size = 1) {
    extra <- if (truth == "M2") additional_component(data, structure) else 0
    1 +  shared + additional_effect_size * extra
}

# Calibrate on the training design; apply the SAME parameters to the test design.
generate_experiment <- function(train, test, structure, truth, snr,
                                error_variance = 1, error_distribution = "Normal", additional_effect_size = 1) {
    error_variance <- var(signal_at(generate_signal(train, structure, truth, additional_effect_size))) / snr
    mu <- generate_signal(train, structure, truth, additional_effect_size)
    train$y <- mu + sqrt(error_variance) * standardised_random(nrow(train), error_distribution)
    test$y <- generate_signal(test, structure, truth, additional_effect_size)
    list(train = train, test = test)
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
