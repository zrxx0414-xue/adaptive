source("run_simulation.R")

result <- run_linear_simulation(
    structure = "Additional main effects", truth = "M2", snr = 10,
    n_train = 200, n_test = 200, repetitions = 500,
    basis = "adaptive", kernel_bw = "adaptive", dim_basis = 1,
    predictor_distribution = "Normal", error_distribution = "Normal",
    test_seed = 4100, training_seed = 42, save_drm_data = FALSE
)

# For manual bandwidths: kernel_bw = 0.5 or c(0.5, 0.3, 0.5).
# For a fixed basis: basis = "q(y) = (y, y^2)".
result$summary
result$error_summary
result$runtime
head(result$parameters)
