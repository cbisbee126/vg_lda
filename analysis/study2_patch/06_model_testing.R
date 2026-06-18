# ============================================================
# 06 MODEL TESTING — PATCH LEVERS → ENGAGEMENT
# ============================================================
#
# Purpose:
# Test whether developer emphasis on progression levers predicts
# player engagement lift and retention.
#
# Data structure:
# - Each row is one strict Steam update / patch note.
# - Updates are nested within games.
# - Models use random intercepts for game.
#
# Outcomes:
# - engagement_lift_0d / engagement_retention_0d
# - engagement_lift_1d / engagement_retention_1d
# - engagement_lift_2d / engagement_retention_2d
#
# Models:
# - Model 1: Main Linear HLM
# - Model 2: Exploratory Quadratic HLM
# ============================================================

rm(list = ls())

library(tidyverse)
library(readr)
library(lubridate)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(performance)

dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 0) HELPERS
# ============================================================

scale2 <- function(x) as.numeric(scale(x))

sig_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01  ~ "**",
    p < 0.05  ~ "*",
    p < 0.10  ~ "+",
    TRUE ~ ""
  )
}

# ============================================================
# 1) LOAD DATA
# ============================================================

df <- read_csv(
  "data_processed/final_patch_dataset.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date = as.Date(event_date),
    game = as.factor(game)
  )

cat("\nLoaded rows:", nrow(df), "\n")
cat("Games:", nlevels(df$game), "\n")

# ============================================================
# 2) DEFINE OUTCOME WINDOWS
# ============================================================

outcome_specs <- tibble(
  window = c("0d", "1d", "2d"),
  lift_dv = c(
    "engagement_lift_0d",
    "engagement_lift_1d",
    "engagement_lift_2d"
  ),
  retention_dv = c(
    "engagement_retention_0d",
    "engagement_retention_1d",
    "engagement_retention_2d"
  )
)

missing_outcomes <- setdiff(
  c(outcome_specs$lift_dv, outcome_specs$retention_dv),
  names(df)
)

if (length(missing_outcomes) > 0) {
  stop(
    "Missing outcome variables from final_patch_dataset.csv:\n",
    paste(missing_outcomes, collapse = ", "),
    "\n\nRun the updated Step 04 first."
  )
}

cat("\nOutcome windows found:\n")
print(outcome_specs)

# ============================================================
# 3) IDENTIFY LAGGED ENGAGEMENT CONTROL
# ============================================================

lag_candidates <- c(
  "log_lag_avg_players",
  "lag_log_avg_players",
  "lagged_engagement",
  "log_lag_engagement",
  "pre_log_avg_players",
  "pre_engagement"
)

lag_var <- lag_candidates[lag_candidates %in% names(df)][1]

if (is.na(lag_var)) {
  stop(
    "Could not find a lagged engagement variable.\n\n",
    "Expected one of:\n",
    paste(lag_candidates, collapse = ", "),
    "\n\nAvailable columns:\n",
    paste(names(df), collapse = ", ")
  )
}

cat("\nUsing lagged engagement control:", lag_var, "\n")

# ============================================================
# 4) REQUIRED VARIABLES
# ============================================================

required_vars <- c(
  outcome_specs$lift_dv,
  outcome_specs$retention_dv,

  "rel_competitive",
  "rel_cosmetic",
  "rel_seasonal",
  "rel_difficulty",

  "log_days_since_release",
  "log_total_chars",
  "new_season_patch",

  "game",
  "event_date",

  lag_var
)

missing_vars <- setdiff(required_vars, names(df))

if (length(missing_vars) > 0) {
  stop(
    "These required variables are missing from the dataset:\n",
    paste(missing_vars, collapse = ", ")
  )
}

# ============================================================
# 5) CLEAN MODEL SAMPLE
# ============================================================

df_model <- df %>%
  filter(
    if_all(all_of(required_vars), ~ !is.na(.x))
  ) %>%
  mutate(
    year = as.factor(format(event_date, "%Y")),

    # Standardized lever predictors
    z_competitive = scale2(rel_competitive),
    z_cosmetic    = scale2(rel_cosmetic),
    z_seasonal    = scale2(rel_seasonal),
    z_difficulty  = scale2(rel_difficulty),

    # Quadratic terms for Model 2
    z_competitive_sq = z_competitive^2,
    z_cosmetic_sq    = z_cosmetic^2,
    z_seasonal_sq    = z_seasonal^2,
    z_difficulty_sq  = z_difficulty^2,

    # Standardized controls
    z_lag_engagement = scale2(.data[[lag_var]]),
    z_days_since_release = scale2(log_days_since_release),
    z_patch_length = scale2(log_total_chars),

    # Binary season timing control
    new_season_patch = as.integer(new_season_patch)
  )

cat("\nModel sample:", nrow(df_model), "rows\n")

cat("\nRows by game:\n")
df_model %>%
  count(game, sort = TRUE) %>%
  print(n = Inf)

# ============================================================
# 6) LEVER CORRELATION CHECK
# ============================================================

cat("\n--- LEVER CORRELATION CHECK ---\n")

lever_cor <- df_model %>%
  select(
    rel_competitive,
    rel_cosmetic,
    rel_seasonal,
    rel_difficulty
  ) %>%
  cor(use = "pairwise.complete.obs")

print(lever_cor)

lever_cor_out <- as.data.frame(lever_cor) %>%
  rownames_to_column("lever")

write_csv(
  lever_cor_out,
  "results/study2_lever_correlations.csv"
)

# ============================================================
# 7) MODEL DEFINITIONS
# ============================================================

# Model 1: Main Linear HLM
model1_rhs <- paste(
  "z_lag_engagement",
  "z_competitive",
  "z_cosmetic",
  "z_seasonal",
  "z_difficulty",
  "new_season_patch",
  "z_days_since_release",
  "z_patch_length",
  sep = " + "
)

# Model 2: Exploratory Quadratic HLM
model2_rhs <- paste(
  "z_lag_engagement",
  "z_competitive + z_competitive_sq",
  "z_cosmetic + z_cosmetic_sq",
  "z_seasonal + z_seasonal_sq",
  "z_difficulty + z_difficulty_sq",
  "new_season_patch",
  "z_days_since_release",
  "z_patch_length",
  sep = " + "
)

# ============================================================
# 8) RUN MODEL FUNCTION
# ============================================================

run_lmer <- function(dv, rhs) {

  form <- as.formula(
    paste0(
      dv,
      " ~ ",
      rhs,
      " + (1 | game)"
    )
  )

  lmer(
    form,
    data = df_model,
    REML = FALSE,
    control = lmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 2e5)
    )
  )
}

# ============================================================
# 9) RUN MODELS ACROSS OUTCOME WINDOWS
# ============================================================

models <- list()

for (i in seq_len(nrow(outcome_specs))) {

  window_i <- outcome_specs$window[i]
  lift_i <- outcome_specs$lift_dv[i]
  retention_i <- outcome_specs$retention_dv[i]

  cat("\nRunning window:", window_i, "\n")

  models[[paste0("model1_lift_", window_i)]] <- run_lmer(
    dv = lift_i,
    rhs = model1_rhs
  )

  models[[paste0("model2_lift_", window_i)]] <- run_lmer(
    dv = lift_i,
    rhs = model2_rhs
  )

  models[[paste0("model1_retention_", window_i)]] <- run_lmer(
    dv = retention_i,
    rhs = model1_rhs
  )

  models[[paste0("model2_retention_", window_i)]] <- run_lmer(
    dv = retention_i,
    rhs = model2_rhs
  )
}

# ============================================================
# 10) MODEL INDEX
# ============================================================

model_index <- tibble(
  object_name = names(models)
) %>%
  mutate(
    model_number = case_when(
      str_detect(object_name, "model1") ~ "Model 1",
      str_detect(object_name, "model2") ~ "Model 2",
      TRUE ~ object_name
    ),
    model = case_when(
      str_detect(object_name, "model1") ~ "Main Linear HLM",
      str_detect(object_name, "model2") ~ "Exploratory Quadratic HLM",
      TRUE ~ object_name
    ),
    outcome = case_when(
      str_detect(object_name, "lift") ~ "Lift",
      str_detect(object_name, "retention") ~ "Retention",
      TRUE ~ NA_character_
    ),
    window = str_extract(object_name, "(0d|1d|2d)$")
  )

# ============================================================
# 11) COLLECT FULL TIDY RESULTS
# ============================================================

extract_results <- function(model_obj, object_name) {

  info <- model_index %>%
    filter(object_name == !!object_name)

  broom.mixed::tidy(
    model_obj,
    effects = "fixed"
  ) %>%
    mutate(
      object_name = object_name,
      model_number = info$model_number,
      model = info$model,
      outcome = info$outcome,
      window = info$window
    )
}

results_all <- map2_dfr(
  models,
  names(models),
  extract_results
) %>%
  mutate(
    term_clean = case_when(
      term == "(Intercept)" ~ "Intercept",
      term == "z_lag_engagement" ~ "Lagged Engagement",
      term == "z_competitive" ~ "Competitive",
      term == "z_cosmetic" ~ "Cosmetic",
      term == "z_seasonal" ~ "Seasonal",
      term == "z_difficulty" ~ "Difficulty",
      term == "z_competitive_sq" ~ "Competitive Squared",
      term == "z_cosmetic_sq" ~ "Cosmetic Squared",
      term == "z_seasonal_sq" ~ "Seasonal Squared",
      term == "z_difficulty_sq" ~ "Difficulty Squared",
      term == "new_season_patch" ~ "Near Season Launch",
      term == "z_days_since_release" ~ "Days Since Release",
      term == "z_patch_length" ~ "Patch Length",
      TRUE ~ term
    ),
    sig = sig_stars(p.value),
    display = sprintf("%.3f (%.3f)%s", estimate, std.error, sig)
  )

# ============================================================
# 12) CLEAN COEFFICIENT TABLE
# ============================================================

clean_terms <- c(
  "Competitive",
  "Cosmetic",
  "Seasonal",
  "Difficulty",
  "Competitive Squared",
  "Cosmetic Squared",
  "Seasonal Squared",
  "Difficulty Squared",
  "Near Season Launch",
  "Patch Length",
  "Days Since Release",
  "Lagged Engagement"
)

clean_results <- results_all %>%
  filter(term_clean %in% clean_terms) %>%
  select(
    window,
    outcome,
    model_number,
    model,
    term = term_clean,
    estimate,
    std_error = std.error,
    statistic,
    p_value = p.value,
    sig,
    display
  ) %>%
  arrange(
    window,
    outcome,
    model_number,
    factor(term, levels = clean_terms)
  )

# Wide version for easier viewing
clean_results_wide <- clean_results %>%
  select(window, outcome, model_number, term, display) %>%
  pivot_wider(
    names_from = c(window, outcome, model_number),
    values_from = display
  )

# ============================================================
# 13) MODEL FIT TABLE
# ============================================================

get_fit <- function(model_obj, object_name) {

  info <- model_index %>%
    filter(object_name == !!object_name)

  r2_vals <- performance::r2_nakagawa(model_obj)

  tibble(
    object_name = object_name,
    model_number = info$model_number,
    model = info$model,
    outcome = info$outcome,
    window = info$window,
    n_obs = nobs(model_obj),
    aic = AIC(model_obj),
    bic = BIC(model_obj),
    r2_marginal = r2_vals$R2_marginal,
    r2_conditional = r2_vals$R2_conditional
  )
}

fit_compare <- map2_dfr(
  models,
  names(models),
  get_fit
) %>%
  arrange(window, outcome, model_number)

# ============================================================
# 14) SAVE RESULTS
# ============================================================

write_csv(
  results_all,
  "results/study2_hlm_models_results_full.csv",
  na = ""
)

write_csv(
  clean_results,
  "results/study2_hlm_models_clean_results.csv",
  na = ""
)

write_csv(
  clean_results_wide,
  "results/study2_hlm_models_clean_results_wide.csv",
  na = ""
)

write_csv(
  fit_compare,
  "results/study2_hlm_models_fit.csv",
  na = ""
)

cat("\nSaved:\n")
cat(" - results/study2_hlm_models_results_full.csv\n")
cat(" - results/study2_hlm_models_clean_results.csv\n")
cat(" - results/study2_hlm_models_clean_results_wide.csv\n")
cat(" - results/study2_hlm_models_fit.csv\n")
cat(" - results/study2_lever_correlations.csv\n")

cat("\nModel guide:\n")
cat(" - Model 1: Main Linear HLM\n")
cat(" - Model 2: Exploratory Quadratic HLM\n")

cat("\nOutcome windows:\n")
cat(" - 0d: immediate = day 0\n")
cat(" - 1d: immediate = days 0 to +1\n")
cat(" - 2d: immediate = days 0 to +2\n")