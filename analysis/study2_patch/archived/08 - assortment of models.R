# ============================================================
# VIDEO GAME PATCH ANALYSIS — FINAL MODEL NOTEBOOK
# ============================================================

rm(list = ls())

# ============================================================
# PACKAGE SETUP (AUTO INSTALL + LOAD)
# ============================================================

packages <- c(
  "tidyverse",
  "readr",
  "broom",
  "lme4",
  "plm",
  "lmtest",
  "sandwich"
)

installed <- packages %in% installed.packages()

if (any(!installed)) {
  install.packages(packages[!installed])
}

lapply(packages, library, character.only = TRUE)

# 🔥 CRITICAL: force correct lag/lead (avoids plm/zoo conflicts)
lag  <- dplyr::lag
lead <- dplyr::lead

# ============================================================
# 1) LOAD DATA
# ============================================================

df <- read_csv(
  "data_processed/final_patch_dataset.csv",
  show_col_types = FALSE
)

# ============================================================
# 2) CREATE VARIABLES
# ============================================================

df <- df %>%
  arrange(game, event_date) %>%
  group_by(game) %>%
  mutate(
    # Lagged engagement
    log_lag_avg_players = lag(log_avg_players),

    # Immediate lift
    log_lift = log1p(avg_players) - log1p(lag_avg_players),

    # Forward retention (7 days)
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
    avg_next_7 = rowMeans(select(., lead_1:lead_7), na.rm = TRUE),
    avg_next_7 = if_else(is.nan(avg_next_7), NA_real_, avg_next_7),
    log_retention_7d = log1p(avg_next_7) - log1p(avg_players),
    game = as.factor(game)
  )

# ============================================================
# 3) CLEAN MODEL DATA
# ============================================================

df_model <- df %>%
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

# ============================================================
# 4) BASELINE OLS MODEL
# ============================================================

model_ols <- lm(
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
  data = df_model
)

cat("\n--- OLS MODEL ---\n")
print(summary(model_ols))

# ============================================================
# 5) MIXED EFFECTS MODEL (HLM) — MAIN MODEL
# ============================================================

model_mixed <- lmer(
  log_avg_players ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars +
    (1 | game),
  data = df_model
)

cat("\n--- MIXED MODEL (RANDOM INTERCEPT) ---\n")
print(summary(model_mixed))

# ============================================================
# 6) MIXED MODEL WITH RANDOM SLOPES (OPTIONAL BUT STRONG)
# ============================================================

model_mixed_slopes <- lmer(
  log_avg_players ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars +
    (1 + rel_competitive + rel_seasonal | game),
  data = df_model
)

cat("\n--- MIXED MODEL (RANDOM SLOPES) ---\n")
print(summary(model_mixed_slopes))

# ============================================================
# 7) FIXED EFFECTS PANEL MODEL
# ============================================================

df_panel <- df_model %>%
  mutate(event_id_numeric = as.numeric(as.factor(event_date)))

model_fe <- plm(
  log_avg_players ~
    log_lag_avg_players +
    rel_competitive +
    rel_cosmetic +
    rel_seasonal +
    rel_difficulty +
    new_season_patch +
    log_days_since_release +
    log_total_chars,
  data = df_panel,
  index = c("game", "event_id_numeric"),
  model = "within"
)

cat("\n--- FIXED EFFECTS MODEL ---\n")
print(summary(model_fe))

# Clustered standard errors (important!)
cat("\n--- FIXED EFFECTS (CLUSTERED SE) ---\n")
print(coeftest(model_fe, vcov = vcovHC(model_fe, type = "HC1")))

# ============================================================
# 8) NOTE ON RANDOM EFFECTS
# ============================================================

cat("\n--- NOTE ---\n")
cat("Random effects model not estimated due to small number of groups (games).\n")
cat("Mixed-effects and fixed-effects models are used instead.\n")


# ============================================================
# 10) SAVE RESULTS
# ============================================================

results <- bind_rows(
  tidy(model_ols) %>% mutate(model = "OLS"),
  tidy(model_fe) %>% mutate(model = "Fixed Effects")
)

write_csv(results, "results/model_comparison_results.csv")

cat("\n✅ ALL MODELS RUN SUCCESSFULLY\n")