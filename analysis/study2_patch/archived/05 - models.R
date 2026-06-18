# ============================================================
# FINAL MODELS — ENGAGEMENT LEVEL, LIFT, RETENTION
# ============================================================

rm(list = ls())

library(tidyverse)
library(readr)
library(broom)

dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD FINAL DATA
# ============================================================

df <- read_csv(
  "data_processed/final_patch_dataset.csv",
  show_col_types = FALSE
)

# ============================================================
# 2) CREATE OUTCOME VARIABLES
# ============================================================

df <- df %>%
  arrange(game, event_date) %>%
  group_by(game) %>%
  mutate(
    # immediate lift = log difference in average players from previous day
    log_lift = log1p(avg_players) - log1p(lag_avg_players),

    # next 7 days of engagement for retention
    # A week is chosen to control for weekend and day of the week biases
    lead_1 = lead(avg_players, 1),
    lead_2 = lead(avg_players, 2),
    lead_3 = lead(avg_players, 3),
    lead_4 = lead(avg_players, 4),
    lead_5 = lead(avg_players, 5),
    lead_6 = lead(avg_players, 6),
    lead_7 = lead(avg_players, 7)
  ) %>%
  ungroup() %>%
  mutate(
    avg_next_7 = rowMeans(
      select(., lead_1:lead_7),
      na.rm = TRUE
    ),
    avg_next_7 = if_else(is.nan(avg_next_7), NA_real_, avg_next_7),

    retention_7d = avg_next_7 / avg_players,
    log_retention_7d = log1p(avg_next_7) - log1p(avg_players),

    game = as.factor(game)
  )

# ============================================================
# 3) MODEL DATASETS
# ============================================================

df_level <- df %>%
  drop_na(
    log_avg_players,
    log_lag_avg_players,
    rel_competitive,
    rel_cosmetic,
    rel_seasonal,
    rel_difficulty,
    new_season_patch,
    log_days_since_release,
    log_total_chars,
    game
  )

df_lift <- df %>%
  drop_na(
    log_lift,
    log_lag_avg_players,
    rel_competitive,
    rel_cosmetic,
    rel_seasonal,
    rel_difficulty,
    new_season_patch,
    log_days_since_release,
    log_total_chars,
    game
  )

df_retention <- df %>%
  drop_na(
    log_retention_7d,
    log_lag_avg_players,
    rel_competitive,
    rel_cosmetic,
    rel_seasonal,
    rel_difficulty,
    new_season_patch,
    log_days_since_release,
    log_total_chars,
    game
  )

# ============================================================
# 4) RUN MODELS
# ============================================================

model_level <- lm(
  log_avg_players ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars +
    game,
  data = df_level
)

model_lift <- lm(
  log_lift ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars +
    game,
  data = df_lift
)

model_retention <- lm(
  log_retention_7d ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars +
    game,
  data = df_retention
)

# ============================================================
# 5) VIEW RESULTS
# ============================================================

cat("\n--- MODEL 1: ENGAGEMENT LEVEL ---\n")
print(summary(model_level))

cat("\n--- MODEL 2: IMMEDIATE LIFT ---\n")
print(summary(model_lift))

cat("\n--- MODEL 3: 7-DAY RETENTION ---\n")
print(summary(model_retention))

# ============================================================
# 6) SAVE TIDY OUTPUT
# ============================================================

results_level <- tidy(model_level) %>%
  mutate(model = "Engagement Level")

results_lift <- tidy(model_lift) %>%
  mutate(model = "Immediate Lift")

results_retention <- tidy(model_retention) %>%
  mutate(model = "7-Day Retention")

all_results <- bind_rows(
  results_level,
  results_lift,
  results_retention
)

write_csv(
  all_results,
  "results/final_model_results.csv"
)

# ============================================================
# 7) SAVE MODEL FIT SUMMARY
# ============================================================

fit_summary <- tibble(
  model = c("Engagement Level", "Immediate Lift", "7-Day Retention"),
  n = c(nobs(model_level), nobs(model_lift), nobs(model_retention)),
  r_squared = c(summary(model_level)$r.squared,
                summary(model_lift)$r.squared,
                summary(model_retention)$r.squared),
  adj_r_squared = c(summary(model_level)$adj.r.squared,
                    summary(model_lift)$adj.r.squared,
                    summary(model_retention)$adj.r.squared)
)

write_csv(
  fit_summary,
  "results/final_model_fit_summary.csv"
)

cat("\n✅ DONE — final models estimated and saved\n")