# ============================================================
# 03 ADD PATCH-LEVEL CONTROLS
# Updated for sentence-level lever pipeline
# Input:
#   data_processed/patch_levers_sentence.csv
#
# Output:
#   data_processed/patch_levers_with_controls.csv
# ============================================================

library(tidyverse)
library(lubridate)
library(readr)
library(stringr)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD DATA
# ============================================================

patch_features <- read_csv(
  "data_processed/patch_levers_sentence.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date  = as.Date(event_date),
    patch_title = as.character(patch_title),
    source_url  = as.character(source_url),
    event_id    = as.character(event_id),
    game        = as.character(game),

    # Main length control from sentence-level pipeline
    total_chars = as.numeric(total_sentence_chars),

    # Safety fallback if total_sentence_chars is missing for any reason
    total_chars = if_else(
      is.na(total_chars) | total_chars <= 0,
      as.numeric(char_count),
      total_chars
    )
  )

cat("\n🎮 Loaded patch-level rows:", nrow(patch_features), "\n")
cat("🎮 Unique games:", n_distinct(patch_features$game), "\n")

# ============================================================
# 2) RELEASE DATES
# ============================================================

release_dates <- tibble(
  game = c(
    "Apex Legends",
    "Brawlhalla",
    "Counter-Strike 2",
    "Marvel Rivals",
    "Overwatch 2",
    "PUBG: BATTLEGROUNDS",
    "THE FINALS",
    "War Thunder"
  ),
  release_date = as.Date(c(
    "2019-02-04",  # Apex Legends
    "2017-10-17",  # Brawlhalla full release
    "2023-09-27",  # Counter-Strike 2
    "2024-12-06",  # Marvel Rivals
    "2022-10-04",  # Overwatch 2
    "2017-12-20",  # PUBG full release
    "2023-12-07",  # THE FINALS
    "2013-08-15"   # War Thunder
  ))
)

patch_features <- patch_features %>%
  left_join(release_dates, by = "game")

# ============================================================
# 3) BASIC TIME + LENGTH CONTROLS
# ============================================================

patch_features <- patch_features %>%
  mutate(
    days_since_release = as.numeric(event_date - release_date),
    days_since_release = if_else(
      days_since_release < 0,
      NA_real_,
      days_since_release
    ),
    log_days_since_release = log1p(days_since_release),
    log_total_chars = log1p(total_chars)
  )

# ============================================================
# 4) DATE-BASED SEASON / RESET DATES
# ============================================================

# -------- APEX LEGENDS --------
apex_starts <- as.Date(c(
  "2019-03-19","2019-07-02","2019-10-01","2020-02-04",
  "2020-05-12","2020-08-18","2020-11-04","2021-02-02",
  "2021-05-04","2021-08-03","2021-11-02","2022-02-08",
  "2022-05-10","2022-08-09","2022-11-01","2023-02-14",
  "2023-05-09","2023-08-08","2023-10-31","2024-02-13",
  "2024-05-07","2024-08-06","2024-11-05","2025-02-11"
))

# -------- OVERWATCH 2 --------
ow_starts <- as.Date(c(
  "2022-10-04","2022-12-06","2023-02-07","2023-04-11",
  "2023-06-13","2023-08-10","2023-10-10","2023-12-05",
  "2024-02-13","2024-04-16","2024-06-20","2024-08-20",
  "2024-10-15","2024-12-10","2025-02-18","2025-04-22",
  "2025-06-24","2025-08-26","2025-10-14","2025-12-09"
))

# -------- MARVEL RIVALS --------
marvel_starts <- as.Date(c(
  "2025-01-10","2025-02-21","2025-04-11","2025-05-30",
  "2025-07-11","2025-08-08","2025-09-12","2025-10-10",
  "2025-11-14","2025-12-12","2026-01-16","2026-02-13",
  "2026-03-20","2026-04-17"
))

# -------- PUBG: BATTLEGROUNDS --------
pubg_starts <- as.Date(c(
  "2018-10-03","2018-12-19","2019-03-28","2019-07-24","2019-10-23",
  "2020-01-22","2020-04-21","2020-07-22","2020-10-21","2020-12-16",
  "2021-03-31","2021-06-02","2021-08-04","2021-10-06","2021-11-30",
  "2022-02-16","2022-04-13","2022-06-08","2022-08-09","2022-10-12",
  "2022-12-06","2023-02-15","2023-04-12","2023-06-14","2023-08-09",
  "2023-10-11","2023-12-06","2024-02-07","2024-04-09","2024-06-12",
  "2024-08-07","2024-10-09","2024-12-05","2025-02-12","2025-04-09",
  "2025-06-11","2025-08-13","2025-10-15","2025-12-03","2026-02-04",
  "2026-04-08"
))

# -------- BRAWLHALLA --------
brawlhalla_starts <- as.Date(c(
  "2015-02-19","2015-09-24","2016-03-23","2016-09-21","2016-12-14",
  "2017-03-15","2017-06-21","2017-09-13","2017-12-13","2018-03-14",
  "2018-06-13","2018-09-12","2018-12-12","2019-04-03","2019-07-03",
  "2019-09-26","2020-01-08","2020-04-14","2020-07-15","2020-10-07",
  "2021-01-20"
))

# -------- THE FINALS --------
finals_starts <- as.Date(c(
  "2023-12-07","2024-03-14","2024-06-13","2024-09-26","2024-12-12",
  "2025-03-20","2025-06-12","2025-09-10","2025-12-10","2026-03-26"
))

# -------- WAR THUNDER --------
warthunder_starts <- as.Date(c(
  "2020-12-02","2021-02-24","2021-05-12","2021-07-28","2021-10-27",
  "2022-01-26","2022-04-27","2022-07-27","2022-10-26","2023-01-25",
  "2023-04-26","2023-07-26","2023-10-25","2024-01-24","2024-04-24",
  "2024-07-24","2024-10-23"
))

# ============================================================
# 5) BUILD DATE-BASED NEW SEASON FLAG
# Universal rule: within +/- 2 days of listed season/reset date
# ============================================================

patch_features <- patch_features %>%
  rowwise() %>%
  mutate(
    new_season_patch = case_when(
      game == "Apex Legends" ~ as.integer(any(abs(event_date - apex_starts) <= 2)),
      game == "Overwatch 2" ~ as.integer(any(abs(event_date - ow_starts) <= 2)),
      game == "Marvel Rivals" ~ as.integer(any(abs(event_date - marvel_starts) <= 2)),
      game == "PUBG: BATTLEGROUNDS" ~ as.integer(any(abs(event_date - pubg_starts) <= 2)),
      game == "Brawlhalla" ~ as.integer(any(abs(event_date - brawlhalla_starts) <= 2)),
      game == "THE FINALS" ~ as.integer(any(abs(event_date - finals_starts) <= 2)),
      game == "War Thunder" ~ as.integer(any(abs(event_date - warthunder_starts) <= 2)),
      TRUE ~ 0L
    )
  ) %>%
  ungroup()

# ============================================================
# 6) VALIDATION
# ============================================================

cat("\n--- NEW SEASON CHECK ---\n")

season_check <- patch_features %>%
  group_by(game) %>%
  summarise(
    season_patches = sum(new_season_patch, na.rm = TRUE),
    total_patches  = n(),
    pct_season     = mean(new_season_patch, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(season_patches))

print(season_check, n = Inf)

cat("\n--- RELEASE TIME CHECK ---\n")

release_time_check <- patch_features %>%
  group_by(game) %>%
  summarise(
    min_days = min(days_since_release, na.rm = TRUE),
    max_days = max(days_since_release, na.rm = TRUE),
    .groups = "drop"
  )

print(release_time_check, n = Inf)

cat("\n--- LENGTH CHECK ---\n")

length_check <- patch_features %>%
  summarise(
    min_total_chars  = min(total_chars, na.rm = TRUE),
    mean_total_chars = mean(total_chars, na.rm = TRUE),
    max_total_chars  = max(total_chars, na.rm = TRUE)
  )

print(length_check)

cat("\n--- SAMPLE SEASON PATCHES ---\n")

sample_season_patches <- patch_features %>%
  filter(new_season_patch == 1) %>%
  select(
    game,
    event_date,
    patch_title,
    rel_seasonal,
    new_season_patch
  ) %>%
  arrange(game, event_date)

print(sample_season_patches, n = 60)

# ============================================================
# 7) SAVE
# ============================================================

write_csv(
  patch_features,
  "data_processed/patch_levers_with_controls.csv"
)

write_csv(
  season_check,
  "results/step3_new_season_check.csv"
)

write_csv(
  release_time_check,
  "results/step3_release_time_check.csv"
)

write_csv(
  length_check,
  "results/step3_length_check.csv"
)

write_csv(
  sample_season_patches,
  "results/step3_sample_season_patches.csv"
)

cat("\n✅ DONE — Controls added to sentence-level lever dataset\n")
cat("📁 Main output: data_processed/patch_levers_with_controls.csv\n")