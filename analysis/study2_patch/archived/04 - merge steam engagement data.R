# ============================================================
# FINAL PIPELINE — DAILY AVERAGE PLAYERS + LAG + MERGE
# ============================================================

rm(list = ls())

library(tidyverse)
library(lubridate)
library(readr)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD PATCH DATA
# ============================================================

patch_data <- read_csv(
  "data_processed/patch_levers_with_controls.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date = as.Date(event_date)
  )

# ============================================================
# 2) LOAD STEAM FILES
# ============================================================

apex <- read_csv("data_raw/apex_steam_data.csv", show_col_types = FALSE) %>%
  mutate(game = "Apex Legends")

marvel <- read_csv("data_raw/marvel_steam_data.csv", show_col_types = FALSE) %>%
  mutate(game = "Marvel Rivals")

overwatch <- read_csv("data_raw/overwatch_steam_data.csv", show_col_types = FALSE) %>%
  mutate(game = "Overwatch 2")

# ============================================================
# 3) COMBINE + STANDARDIZE
# ============================================================

steam_raw <- bind_rows(apex, marvel, overwatch) %>%
  rename(
    datetime = `DateTime`,
    players = `Players`,
    avg_players_raw = `Average Players`
  ) %>%
  mutate(
    datetime = as.POSIXct(datetime, tz = "UTC"),
    event_date = as.Date(datetime)
  )

# ============================================================
# 4) COLLAPSE TO DAILY (USING AVERAGE PLAYERS FIELD)
# ============================================================

steam_daily <- steam_raw %>%
  group_by(game, event_date) %>%
  summarise(
    n_rows = n(),
    
    avg_players = case_when(
      
      # ✅ OLD DATA: already daily → keep it
      n_rows == 1 ~ first(avg_players_raw),
      
      # ✅ NEW DATA: multiple rows → average available avg values
      TRUE ~ mean(avg_players_raw, na.rm = TRUE)
    ),
    
    # clean NaN → NA
    avg_players = if_else(is.nan(avg_players), NA_real_, avg_players),
    
    # always safe
    peak_players = max(players, na.rm = TRUE),
    
    # diagnostics
    n_avg_obs_day = sum(!is.na(avg_players_raw)),
    
    .groups = "drop"
  )

# ============================================================
# 5) ADD LAG (CRITICAL)
# ============================================================

steam_daily <- steam_daily %>%
  arrange(game, event_date) %>%
  group_by(game) %>%
  mutate(
    lag_avg_players = lag(avg_players, 1)
  ) %>%
  ungroup()

# ============================================================
# 6) COLLAPSE PATCH DUPLICATES (IMPORTANT)
# ============================================================

patch_daily <- patch_data %>%
  group_by(game, event_date) %>%
  summarise(
    rel_competitive = mean(rel_competitive, na.rm = TRUE),
    rel_cosmetic = mean(rel_cosmetic, na.rm = TRUE),
    rel_seasonal = mean(rel_seasonal, na.rm = TRUE),
    rel_difficulty = mean(rel_difficulty, na.rm = TRUE),
    
    total_chars = sum(total_chars, na.rm = TRUE),
    log_total_chars = log1p(sum(total_chars, na.rm = TRUE)),
    
    new_season_patch = max(new_season_patch, na.rm = TRUE),
    
    days_since_release = first(days_since_release),
    log_days_since_release = first(log_days_since_release),
    
    patch_number = first(patch_number),
    
    .groups = "drop"
  )

# ============================================================
# 7) MERGE
# ============================================================

merged_data <- patch_daily %>%
  left_join(
    steam_daily,
    by = c("game", "event_date")
  )

# ============================================================
# 8) CLEAN FOR MODELING
# ============================================================

merged_data <- merged_data %>%
  mutate(
    log_avg_players = log1p(avg_players),
    log_lag_avg_players = log1p(lag_avg_players),
    log_peak_players = log1p(peak_players)
  )

# ============================================================
# 9) VALIDATION
# ============================================================

cat("\n--- STEAM DAILY CHECK ---\n")

steam_daily %>%
  group_by(game) %>%
  summarise(
    min_date = min(event_date),
    max_date = max(event_date),
    min_avg_obs = min(n_avg_obs_day, na.rm = TRUE),
    max_avg_obs = max(n_avg_obs_day, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- MERGE CHECK ---\n")

merged_data %>%
  summarise(
    total_patch_days = n(),
    missing_avg_players = sum(is.na(avg_players)),
    missing_lag = sum(is.na(lag_avg_players))
  ) %>%
  print()

cat("\n--- SAMPLE ---\n")

merged_data %>%
  select(game, event_date, avg_players, lag_avg_players, peak_players, n_avg_obs_day) %>%
  arrange(game, event_date) %>%
  print(n = 20)

# ============================================================
# 10) FINAL MODEL DATASET
# ============================================================

final_data <- merged_data %>%
  drop_na(avg_players, lag_avg_players)

# ============================================================
# 11) SAVE
# ============================================================

write_csv(
  final_data,
  "data_processed/final_patch_dataset.csv"
)

cat("\n✅ DONE — FINAL DATASET READY FOR MODELING\n")