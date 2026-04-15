#!/usr/bin/env Rscript

# Generate cross-language reference outputs from the R bayesiansurpriser package.
#
# Usage:
#   Rscript tools/generate_r_reference.R
#
# Set BAYESIANSURPRISER_R_PATH to override the default sibling package path.

args <- commandArgs(trailingOnly = TRUE)
out_path <- if (length(args) >= 1) args[[1]] else "tests/fixtures/r_reference.json"
r_path <- Sys.getenv("BAYESIANSURPRISER_R_PATH", unset = "../bayesian-surprise-r")

if (!requireNamespace("devtools", quietly = TRUE)) {
  stop("The devtools R package is required to load the R reference package.")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The jsonlite R package is required to write reference fixtures.")
}

suppressPackageStartupMessages(devtools::load_all(r_path, quiet = TRUE))

as_plain <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.matrix(x)) {
    return(lapply(seq_len(nrow(x)), function(i) unname(x[i, ])))
  }
  unname(x)
}

capture_result <- function(result) {
  list(
    surprise = as_plain(result$surprise),
    signed_surprise = as_plain(result$signed_surprise),
    posteriors = as_plain(result$posteriors),
    model_contributions = as_plain(result$model_contributions),
    prior = as_plain(result$model_space$prior),
    model_names = unname(names(result$model_space$models))
  )
}

count_observed <- c(50, 100, 150, 200)
count_expected <- c(10000, 50000, 100000, 25000)
count_space <- model_space(
  bs_model_uniform(),
  bs_model_baserate(count_expected),
  bs_model_funnel(count_expected)
)
count_result <- compute_surprise(
  count_space,
  observed = count_observed,
  expected = count_expected,
  return_posteriors = TRUE,
  return_contributions = TRUE
)

rates <- c(0.11, 0.14, 0.13, 0.28, 0.09, 0.12)
gaussian_space <- model_space(
  bs_model_uniform(),
  bs_model_gaussian()
)
gaussian_result <- compute_surprise(
  gaussian_space,
  observed = rates,
  return_posteriors = TRUE,
  return_contributions = TRUE
)

legacy_population <- c(100000, 100000, 100000)
legacy_counts <- c(1000, 1500, 3000)
legacy_space <- model_space(
  bs_model_uniform(),
  bs_model_funnel(legacy_population, formula = "paper"),
  prior = c(0.5, 0.5)
)
legacy_result <- compute_surprise(
  legacy_space,
  observed = legacy_counts,
  expected = legacy_population,
  return_posteriors = TRUE,
  return_contributions = TRUE,
  normalize_posterior = FALSE
)

df <- data.frame(
  region = c("a", "b", "c", "d"),
  events = count_observed,
  population = count_expected
)
df_result <- surprise(
  df,
  observed = "events",
  expected = "population",
  models = c("uniform", "baserate", "funnel")
)

reference <- list(
  metadata = list(
    generated_by = "bayesiansurpriser",
    r_package_path = normalizePath(r_path),
    note = "Reference values generated from the sibling R package for Python cross-validation."
  ),
  cases = list(
    count_default = c(
      list(observed = count_observed, expected = count_expected),
      capture_result(count_result)
    ),
    gaussian_rates = c(
      list(observed = rates, expected = NULL),
      capture_result(gaussian_result)
    ),
    legacy_unnormalized = c(
      list(observed = legacy_counts, expected = legacy_population),
      capture_result(legacy_result)
    ),
    dataframe_surprise = list(
      observed = count_observed,
      expected = count_expected,
      surprise = unname(df_result$surprise),
      signed_surprise = unname(df_result$signed_surprise)
    )
  )
)

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
writeLines(jsonlite::toJSON(reference, auto_unbox = TRUE, pretty = TRUE, digits = 16), out_path)
message("Wrote ", out_path)
