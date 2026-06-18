# ============================================================
# 04 MERGE PATCH-DAY FEATURES WITH STEAM DAILY ENGAGEMENT
# Creates 0-day, 1-day, and 2-day lift/retention outcomes
# ============================================================

rm(list = ls())

library(tidyverse)
library(lubridate)
library(readr)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# HELPER: SAFE DATETIME PARSER
# ============================================================

parse_steam_datetime <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x[x == ""] <- NA_character_

  out <- rep(as.POSIXct(NA, tz = "UTC"), length(x))

  formats <- c(
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%OS",
    "%Y-%m-%dT%H:%M:%OSZ",
    "%Y-%m-%dT%H:%M:%SZ"
  )

  for (fmt in formats) {
    needs_parse <- is.na(out) & !is.na(x)
    if (!any(needs_parse)) break

    parsed <- suppressWarnings(
      as.POSIXct(x[needs_parse], format = fmt, tz = "UTC")
    )

    idx <- which(needs_parse)
    out[idx[!is.na(parsed)]] <- parsed[!is.na(parsed)]
  }

  needs_parse <- is.na(out) & !is.na(x)

  if (any(needs_parse)) {
    parsed <- suppressWarnings(
      as.POSIXct(x[needs_parse], tz = "UTC")
    )

    idx <- which(needs_parse)
    out[idx[!is.na(parsed)]] <- parsed[!is.na(parsed)]
  }

  out
}

# ============================================================
# HELPER: LOAD STEAM FILE
# ============================================================

load_steam_file <- function(path, game_name) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(
      DateTime = as.character(DateTime),
      Players = as.numeric(Players),
      `Average Players` = as.numeric(`Average Players`),
      game = game_name
    )
}

# ============================================================
# HELPER: WINDOW STATS
# ============================================================

get_window_stats <- function(game_i, date_i, start_offset, end_offset, steam_daily_df) {
  window_df <- steam_daily_df %>%
    filter(
      game == game_i,
      event_date >= date_i + start_offset,
      event_date <= date_i + end_offset
    )

  tibble(
    mean_log_avg_players = mean(window_df$log_avg_players_daily, na.rm = TRUE),
    mean_avg_players = mean(window_df$avg_players, na.rm = TRUE),
    n_days = sum(!is.na(window_df$log_avg_players_daily))
  ) %>%
    mutate(
      mean_log_avg_players = if_else(
        is.nan(mean_log_avg_players),
        NA_real_,
        mean_log_avg_players
      ),
      mean_avg_players = if_else(
        is.nan(mean_avg_players),
        NA_real_,
        mean_avg_players
      )
    )
}

# ============================================================
# 1) LOAD PATCH-DAY FEATURES
# ============================================================

patch_data <- read_csv(
  "data_processed/patch_levers_with_controls.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date = as.Date(event_date),
    game = as.character(game)
  )

cat("\n🎮 Loaded patch-level rows:", nrow(patch_data), "\n")
cat("🎮 Unique games:", n_distinct(patch_data$game), "\n")

# ============================================================
# 2) LOAD STEAM FILES
# ============================================================

apex <- load_steam_file(
  "data_raw/apex_steam_data.csv",
  "Apex Legends"
)

marvel <- load_steam_file(
  "data_raw/marvel_steam_data.csv",
  "Marvel Rivals"
)

overwatch <- load_steam_file(
  "data_raw/overwatch_steam_data.csv",
  "Overwatch 2"
)

brawlhalla <- load_steam_file(
  "data_raw/brawlhalla_steam_data.csv",
  "Brawlhalla"
)

finals <- load_steam_file(
  "data_raw/the_finals_steam_data.csv",
  "THE FINALS"
)

war_thunder <- load_steam_file(
  "data_raw/war_thunder_steam_data.csv",
  "War Thunder"
)

pubg <- load_steam_file(
  "data_raw/pubg_steam_data.csv",
  "PUBG: BATTLEGROUNDS"
)

counter_strike <- load_steam_file(
  "data_raw/counter_strike_steam_data.csv",
  "Counter-Strike 2"
)

# ============================================================
# 3) COMBINE + STANDARDIZE STEAM
# ============================================================

steam_raw <- bind_rows(
  apex,
  marvel,
  overwatch,
  brawlhalla,
  finals,
  war_thunder,
  pubg,
  counter_strike
) %>%
  rename(
    datetime = `DateTime`,
    players = `Players`,
    avg_players_raw = `Average Players`
  ) %>%
  mutate(
    datetime = parse_steam_datetime(datetime),
    event_date = as.Date(datetime)
  ) %>%
  filter(
    !is.na(game),
    !is.na(event_date)
  )

cat("\n🎮 Loaded raw Steam rows:", nrow(steam_raw), "\n")
cat("🎮 Steam games:", n_distinct(steam_raw$game), "\n")

# ============================================================
# 4) COLLAPSE TO STEAM DAILY
# ============================================================

steam_daily <- steam_raw %>%
  group_by(game, event_date) %>%
  summarise(
    n_rows = n(),

    avg_players = case_when(
      n_rows == 1 & !is.na(first(avg_players_raw)) ~ first(avg_players_raw),
      sum(!is.na(avg_players_raw)) > 0 ~ mean(avg_players_raw, na.rm = TRUE),
      TRUE ~ mean(players, na.rm = TRUE)
    ),

    avg_players = if_else(is.nan(avg_players), NA_real_, avg_players),

    peak_players = if (all(is.na(players))) {
      NA_real_
    } else {
      max(players, na.rm = TRUE)
    },

    n_avg_obs_day = sum(!is.na(avg_players_raw)),

    .groups = "drop"
  ) %>%
  arrange(game, event_date) %>%
  group_by(game) %>%
  mutate(
    lag_avg_players = lag(avg_players, 1),
    log_avg_players_daily = log1p(avg_players),
    log_lag_avg_players_daily = log1p(lag_avg_players),
    log_peak_players_daily = log1p(peak_players)
  ) %>%
  ungroup()

# ============================================================
# 5) COLLAPSE PATCH COMMUNICATION TO GAME-DAY
# ============================================================

patch_daily <- patch_data %>%
  filter(game %in% unique(steam_daily$game)) %>%
  group_by(game, event_date) %>%
  summarise(
    abs_competitive = sum(abs_competitive, na.rm = TRUE),
    abs_cosmetic    = sum(abs_cosmetic, na.rm = TRUE),
    abs_seasonal    = sum(abs_seasonal, na.rm = TRUE),
    abs_difficulty  = sum(abs_difficulty, na.rm = TRUE),

    total_chars = sum(total_chars, na.rm = TRUE),
    total_chars = if_else(is.infinite(total_chars), NA_real_, total_chars),

    rel_competitive = if_else(
      total_chars > 0,
      abs_competitive / total_chars,
      NA_real_
    ),

    rel_cosmetic = if_else(
      total_chars > 0,
      abs_cosmetic / total_chars,
      NA_real_
    ),

    rel_seasonal = if_else(
      total_chars > 0,
      abs_seasonal / total_chars,
      NA_real_
    ),

    rel_difficulty = if_else(
      total_chars > 0,
      abs_difficulty / total_chars,
      NA_real_
    ),

    log_total_chars = log1p(total_chars),

    new_season_patch = max(new_season_patch, na.rm = TRUE),
    days_since_release = first(days_since_release),
    log_days_since_release = first(log_days_since_release),

    n_patch_posts_day = n(),

    patch_titles = paste(unique(patch_title), collapse = " | "),
    event_ids = paste(unique(event_id), collapse = " | "),

    .groups = "drop"
  ) %>%
  mutate(
    new_season_patch = if_else(
      is.infinite(new_season_patch),
      0L,
      as.integer(new_season_patch)
    )
  )

# ============================================================
# 6) CREATE ENGAGEMENT OUTCOME WINDOWS
# ============================================================

# Outcome logic:
#
# Pre-window:
#   days -7 to -1
#
# 0-day version:
#   immediate window = day 0
#   post window      = days +1 to +7
#
# 1-day version:
#   immediate window = days 0 to +1
#   post window      = days +2 to +7
#
# 2-day version:
#   immediate window = days 0 to +2
#   post window      = days +3 to +7
#
# For each version:
#   lift      = immediate window - pre-window
#   retention = post-window - immediate window

outcome_windows <- patch_daily %>%
  select(game, event_date) %>%
  distinct() %>%
  rowwise() %>%
  mutate(
    # -------------------------
    # Baseline / pre-window
    # -------------------------
    pre_window = list(
      get_window_stats(game, event_date, -7, -1, steam_daily)
    ),

    pre_log_avg_players = pre_window$mean_log_avg_players,
    pre_avg_players = pre_window$mean_avg_players,
    n_pre_days = pre_window$n_days,

    # -------------------------
    # 0-day immediate version
    # -------------------------
    immediate_0d_window = list(
      get_window_stats(game, event_date, 0, 0, steam_daily)
    ),

    post_0d_window = list(
      get_window_stats(game, event_date, 1, 7, steam_daily)
    ),

    immediate_0d_log_avg_players = immediate_0d_window$mean_log_avg_players,
    post_0d_log_avg_players = post_0d_window$mean_log_avg_players,

    immediate_0d_avg_players = immediate_0d_window$mean_avg_players,
    post_0d_avg_players = post_0d_window$mean_avg_players,

    n_immediate_0d_days = immediate_0d_window$n_days,
    n_post_0d_days = post_0d_window$n_days,

    engagement_lift_0d =
      immediate_0d_log_avg_players - pre_log_avg_players,

    engagement_retention_0d =
      post_0d_log_avg_players - immediate_0d_log_avg_players,

    raw_lift_players_0d =
      immediate_0d_avg_players - pre_avg_players,

    raw_retention_players_0d =
      post_0d_avg_players - immediate_0d_avg_players,

    pct_lift_players_0d = if_else(
      pre_avg_players > 0,
      (immediate_0d_avg_players - pre_avg_players) / pre_avg_players,
      NA_real_
    ),

    pct_retention_players_0d = if_else(
      immediate_0d_avg_players > 0,
      (post_0d_avg_players - immediate_0d_avg_players) / immediate_0d_avg_players,
      NA_real_
    ),

    # -------------------------
    # 1-day immediate version
    # -------------------------
    immediate_1d_window = list(
      get_window_stats(game, event_date, 0, 1, steam_daily)
    ),

    post_1d_window = list(
      get_window_stats(game, event_date, 2, 7, steam_daily)
    ),

    immediate_1d_log_avg_players = immediate_1d_window$mean_log_avg_players,
    post_1d_log_avg_players = post_1d_window$mean_log_avg_players,

    immediate_1d_avg_players = immediate_1d_window$mean_avg_players,
    post_1d_avg_players = post_1d_window$mean_avg_players,

    n_immediate_1d_days = immediate_1d_window$n_days,
    n_post_1d_days = post_1d_window$n_days,

    engagement_lift_1d =
      immediate_1d_log_avg_players - pre_log_avg_players,

    engagement_retention_1d =
      post_1d_log_avg_players - immediate_1d_log_avg_players,

    raw_lift_players_1d =
      immediate_1d_avg_players - pre_avg_players,

    raw_retention_players_1d =
      post_1d_avg_players - immediate_1d_avg_players,

    pct_lift_players_1d = if_else(
      pre_avg_players > 0,
      (immediate_1d_avg_players - pre_avg_players) / pre_avg_players,
      NA_real_
    ),

    pct_retention_players_1d = if_else(
      immediate_1d_avg_players > 0,
      (post_1d_avg_players - immediate_1d_avg_players) / immediate_1d_avg_players,
      NA_real_
    ),

    # -------------------------
    # 2-day immediate version
    # -------------------------
    immediate_2d_window = list(
      get_window_stats(game, event_date, 0, 2, steam_daily)
    ),

    post_2d_window = list(
      get_window_stats(game, event_date, 3, 7, steam_daily)
    ),

    immediate_2d_log_avg_players = immediate_2d_window$mean_log_avg_players,
    post_2d_log_avg_players = post_2d_window$mean_log_avg_players,

    immediate_2d_avg_players = immediate_2d_window$mean_avg_players,
    post_2d_avg_players = post_2d_window$mean_avg_players,

    n_immediate_2d_days = immediate_2d_window$n_days,
    n_post_2d_days = post_2d_window$n_days,

    engagement_lift_2d =
      immediate_2d_log_avg_players - pre_log_avg_players,

    engagement_retention_2d =
      post_2d_log_avg_players - immediate_2d_log_avg_players,

    raw_lift_players_2d =
      immediate_2d_avg_players - pre_avg_players,

    raw_retention_players_2d =
      post_2d_avg_players - immediate_2d_avg_players,

    pct_lift_players_2d = if_else(
      pre_avg_players > 0,
      (immediate_2d_avg_players - pre_avg_players) / pre_avg_players,
      NA_real_
    ),

    pct_retention_players_2d = if_else(
      immediate_2d_avg_players > 0,
      (post_2d_avg_players - immediate_2d_avg_players) / immediate_2d_avg_players,
      NA_real_
    )
  ) %>%
  ungroup() %>%
  select(
    game,
    event_date,

    pre_log_avg_players,
    pre_avg_players,
    n_pre_days,

    immediate_0d_log_avg_players,
    post_0d_log_avg_players,
    immediate_0d_avg_players,
    post_0d_avg_players,
    n_immediate_0d_days,
    n_post_0d_days,
    engagement_lift_0d,
    engagement_retention_0d,
    raw_lift_players_0d,
    raw_retention_players_0d,
    pct_lift_players_0d,
    pct_retention_players_0d,

    immediate_1d_log_avg_players,
    post_1d_log_avg_players,
    immediate_1d_avg_players,
    post_1d_avg_players,
    n_immediate_1d_days,
    n_post_1d_days,
    engagement_lift_1d,
    engagement_retention_1d,
    raw_lift_players_1d,
    raw_retention_players_1d,
    pct_lift_players_1d,
    pct_retention_players_1d,

    immediate_2d_log_avg_players,
    post_2d_log_avg_players,
    immediate_2d_avg_players,
    post_2d_avg_players,
    n_immediate_2d_days,
    n_post_2d_days,
    engagement_lift_2d,
    engagement_retention_2d,
    raw_lift_players_2d,
    raw_retention_players_2d,
    pct_lift_players_2d,
    pct_retention_players_2d
  )

# ============================================================
# 7) MERGE PATCH FEATURES + SAME-DAY ENGAGEMENT + OUTCOMES
# ============================================================

merged_data <- patch_daily %>%
  left_join(
    steam_daily,
    by = c("game", "event_date")
  ) %>%
  left_join(
    outcome_windows,
    by = c("game", "event_date")
  ) %>%
  mutate(
    log_avg_players = log1p(avg_players),
    log_lag_avg_players = log1p(lag_avg_players),
    log_peak_players = log1p(peak_players)
  )

# ============================================================
# 8) VALIDATION
# ============================================================

cat("\n--- STEAM DAILY CHECK ---\n")

steam_daily %>%
  group_by(game) %>%
  summarise(
    min_date = min(event_date, na.rm = TRUE),
    max_date = max(event_date, na.rm = TRUE),
    min_avg_obs = min(n_avg_obs_day, na.rm = TRUE),
    max_avg_obs = max(n_avg_obs_day, na.rm = TRUE),
    total_days = n(),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- PATCH-DAY CHECK ---\n")

patch_daily %>%
  group_by(game) %>%
  summarise(
    patch_days = n(),
    mean_patch_posts_day = mean(n_patch_posts_day, na.rm = TRUE),
    max_patch_posts_day = max(n_patch_posts_day, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- MERGE CHECK ---\n")

merge_check <- merged_data %>%
  summarise(
    total_patch_days = n(),
    missing_avg_players = sum(is.na(avg_players)),
    missing_lag = sum(is.na(lag_avg_players)),

    missing_lift_0d = sum(is.na(engagement_lift_0d)),
    missing_retention_0d = sum(is.na(engagement_retention_0d)),

    missing_lift_1d = sum(is.na(engagement_lift_1d)),
    missing_retention_1d = sum(is.na(engagement_retention_1d)),

    missing_lift_2d = sum(is.na(engagement_lift_2d)),
    missing_retention_2d = sum(is.na(engagement_retention_2d))
  )

print(merge_check)

cat("\n--- OUTCOME WINDOW DAY COUNTS ---\n")

window_check <- merged_data %>%
  summarise(
    min_pre_days = min(n_pre_days, na.rm = TRUE),
    mean_pre_days = mean(n_pre_days, na.rm = TRUE),
    max_pre_days = max(n_pre_days, na.rm = TRUE),

    min_immediate_0d_days = min(n_immediate_0d_days, na.rm = TRUE),
    mean_immediate_0d_days = mean(n_immediate_0d_days, na.rm = TRUE),
    max_immediate_0d_days = max(n_immediate_0d_days, na.rm = TRUE),
    min_post_0d_days = min(n_post_0d_days, na.rm = TRUE),
    mean_post_0d_days = mean(n_post_0d_days, na.rm = TRUE),
    max_post_0d_days = max(n_post_0d_days, na.rm = TRUE),

    min_immediate_1d_days = min(n_immediate_1d_days, na.rm = TRUE),
    mean_immediate_1d_days = mean(n_immediate_1d_days, na.rm = TRUE),
    max_immediate_1d_days = max(n_immediate_1d_days, na.rm = TRUE),
    min_post_1d_days = min(n_post_1d_days, na.rm = TRUE),
    mean_post_1d_days = mean(n_post_1d_days, na.rm = TRUE),
    max_post_1d_days = max(n_post_1d_days, na.rm = TRUE),

    min_immediate_2d_days = min(n_immediate_2d_days, na.rm = TRUE),
    mean_immediate_2d_days = mean(n_immediate_2d_days, na.rm = TRUE),
    max_immediate_2d_days = max(n_immediate_2d_days, na.rm = TRUE),
    min_post_2d_days = min(n_post_2d_days, na.rm = TRUE),
    mean_post_2d_days = mean(n_post_2d_days, na.rm = TRUE),
    max_post_2d_days = max(n_post_2d_days, na.rm = TRUE)
  )

print(window_check)

cat("\n--- OUTCOME CHECK ---\n")

outcome_check <- merged_data %>%
  summarise(
    rows = n(),

    mean_lift_0d = mean(engagement_lift_0d, na.rm = TRUE),
    sd_lift_0d = sd(engagement_lift_0d, na.rm = TRUE),
    mean_retention_0d = mean(engagement_retention_0d, na.rm = TRUE),
    sd_retention_0d = sd(engagement_retention_0d, na.rm = TRUE),

    mean_lift_1d = mean(engagement_lift_1d, na.rm = TRUE),
    sd_lift_1d = sd(engagement_lift_1d, na.rm = TRUE),
    mean_retention_1d = mean(engagement_retention_1d, na.rm = TRUE),
    sd_retention_1d = sd(engagement_retention_1d, na.rm = TRUE),

    mean_lift_2d = mean(engagement_lift_2d, na.rm = TRUE),
    sd_lift_2d = sd(engagement_lift_2d, na.rm = TRUE),
    mean_retention_2d = mean(engagement_retention_2d, na.rm = TRUE),
    sd_retention_2d = sd(engagement_retention_2d, na.rm = TRUE)
  )

print(outcome_check)

cat("\n--- OUTCOME CHECK BY GAME ---\n")

outcome_game_check <- merged_data %>%
  group_by(game) %>%
  summarise(
    patch_days = n(),

    mean_lift_0d = mean(engagement_lift_0d, na.rm = TRUE),
    mean_retention_0d = mean(engagement_retention_0d, na.rm = TRUE),

    mean_lift_1d = mean(engagement_lift_1d, na.rm = TRUE),
    mean_retention_1d = mean(engagement_retention_1d, na.rm = TRUE),

    mean_lift_2d = mean(engagement_lift_2d, na.rm = TRUE),
    mean_retention_2d = mean(engagement_retention_2d, na.rm = TRUE),

    mean_pre_days = mean(n_pre_days, na.rm = TRUE),

    .groups = "drop"
  ) %>%
  arrange(desc(patch_days))

print(outcome_game_check, n = Inf)

cat("\n--- SAMPLE FINAL DATA ---\n")

merged_data %>%
  select(
    game,
    event_date,
    avg_players,
    lag_avg_players,
    pre_log_avg_players,

    immediate_0d_log_avg_players,
    post_0d_log_avg_players,
    engagement_lift_0d,
    engagement_retention_0d,

    immediate_1d_log_avg_players,
    post_1d_log_avg_players,
    engagement_lift_1d,
    engagement_retention_1d,

    immediate_2d_log_avg_players,
    post_2d_log_avg_players,
    engagement_lift_2d,
    engagement_retention_2d,

    n_pre_days,
    n_immediate_0d_days,
    n_post_0d_days,
    n_immediate_1d_days,
    n_post_1d_days,
    n_immediate_2d_days,
    n_post_2d_days,
    n_patch_posts_day
  ) %>%
  arrange(game, event_date) %>%
  print(n = 20)

# ============================================================
# 9) FINAL MODEL DATASET
# ============================================================

# Minimum window requirements:
#
# 0-day version:
# - at least 3 pre-days
# - at least 1 immediate day
# - at least 3 post-days
#
# 1-day version:
# - at least 3 pre-days
# - at least 1 immediate day
# - at least 3 post-days
#
# 2-day version:
# - at least 3 pre-days
# - at least 2 immediate days
# - at least 3 post-days
#
# We keep rows that support at least the 0-day version.
# The model script can filter further depending on which outcome version is used.

final_data <- merged_data %>%
  filter(
    !is.na(avg_players),
    !is.na(lag_avg_players),
    n_pre_days >= 3,

    !is.na(engagement_lift_0d),
    !is.na(engagement_retention_0d),
    n_immediate_0d_days >= 1,
    n_post_0d_days >= 3
  )

cat("\n--- FINAL DATASET CHECK ---\n")

final_check <- final_data %>%
  summarise(
    final_rows = n(),
    games = n_distinct(game),
    min_date = min(event_date, na.rm = TRUE),
    max_date = max(event_date, na.rm = TRUE),

    mean_lift_0d = mean(engagement_lift_0d, na.rm = TRUE),
    mean_retention_0d = mean(engagement_retention_0d, na.rm = TRUE),

    mean_lift_1d = mean(engagement_lift_1d, na.rm = TRUE),
    mean_retention_1d = mean(engagement_retention_1d, na.rm = TRUE),

    mean_lift_2d = mean(engagement_lift_2d, na.rm = TRUE),
    mean_retention_2d = mean(engagement_retention_2d, na.rm = TRUE)
  )

print(final_check)

cat("\n--- FINAL ROWS BY GAME ---\n")

final_game_check <- final_data %>%
  count(game, sort = TRUE)

print(final_game_check, n = Inf)

# ============================================================
# 10) SAVE
# ============================================================

write_csv(
  final_data,
  "data_processed/final_patch_dataset.csv"
)

write_csv(
  steam_daily,
  "data_processed/steam_daily_engagement.csv"
)

write_csv(
  patch_daily,
  "data_processed/patch_daily_features.csv"
)

write_csv(
  outcome_windows,
  "data_processed/patch_engagement_outcomes.csv"
)

write_csv(
  merge_check,
  "results/step4_merge_check.csv"
)

write_csv(
  window_check,
  "results/step4_window_check.csv"
)

write_csv(
  outcome_check,
  "results/step4_outcome_check.csv"
)

write_csv(
  outcome_game_check,
  "results/step4_outcome_game_check.csv"
)

write_csv(
  final_check,
  "results/step4_final_check.csv"
)

write_csv(
  final_game_check,
  "results/step4_final_rows_by_game.csv"
)

cat("\n✅ DONE — FINAL DATASET READY FOR MODELING\n")
cat("📁 Main modeling file: data_processed/final_patch_dataset.csv\n")
cat("📁 Supporting files:\n")
cat("   - data_processed/steam_daily_engagement.csv\n")
cat("   - data_processed/patch_daily_features.csv\n")
cat("   - data_processed/patch_engagement_outcomes.csv\n")
cat("📁 Diagnostics:\n")
cat("   - results/step4_merge_check.csv\n")
cat("   - results/step4_window_check.csv\n")
cat("   - results/step4_outcome_check.csv\n")
cat("   - results/step4_outcome_game_check.csv\n")
cat("   - results/step4_final_check.csv\n")
cat("   - results/step4_final_rows_by_game.csv\n")