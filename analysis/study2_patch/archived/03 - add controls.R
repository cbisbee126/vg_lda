# ============================================================
# 03 ADD PATCH-LEVEL CONTROLS (FINAL — ALL GAMES, FIXED)
# ============================================================

library(tidyverse)
library(lubridate)
library(readr)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD DATA
# ============================================================

patch_features <- read_csv(
  "data_processed/patch_levers.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date = as.Date(event_date)
  )

# ============================================================
# 2) GAME RELEASE DATES
# ============================================================

release_dates <- tibble(
  game = c("Apex Legends", "Marvel Rivals", "Overwatch 2"),
  release_date = as.Date(c(
    "2019-02-04",
    "2024-12-06",
    "2022-10-04"
  ))
)

patch_features <- patch_features %>%
  left_join(release_dates, by = "game")

# ============================================================
# 3) SEASON START DATES
# ============================================================
# -------- OVERWATCH 2 --------
ow_starts <- as.Date(c(
  "2022-10-04","2022-12-06","2023-02-07","2023-04-11",
  "2023-06-13","2023-08-10","2023-10-10","2023-12-05",
  "2024-02-13","2024-04-16","2024-06-20","2024-08-20",
  "2024-10-15","2024-12-10","2025-02-18","2025-04-22",
  "2025-06-24","2025-08-26","2025-10-14","2025-12-09"
))

# -------- MARVEL RIVALS --------
# Use BOTH .0 and .5 starts
marvel_starts <- as.Date(c(
  "2025-01-10", # 1.0
  "2025-02-21", # 1.5
  "2025-04-11", # 2.0
  "2025-05-30", # 2.5
  "2025-07-11", # 3.0
  "2025-08-08", # 3.5
  "2025-09-12", # 4.0
  "2025-10-10", # 4.5
  "2025-11-14", # 5.0
  "2025-12-12", # 5.5
  "2026-01-16", # 6.0
  "2026-02-13", # 6.5
  "2026-03-20", # 7.0
  "2026-04-17"  # 7.5
))

# -------- APEX LEGENDS --------
apex_starts <- as.Date(c(
  "2019-03-19","2019-07-02","2019-10-01","2020-02-04",
  "2020-05-12","2020-08-18","2020-11-04","2021-02-02",
  "2021-05-04","2021-08-03","2021-11-02","2022-02-08",
  "2022-05-10","2022-08-09","2022-11-01","2023-02-14",
  "2023-05-09","2023-08-08","2023-10-31","2024-02-13",
  "2024-05-07","2024-08-06","2024-11-05","2025-02-11"
))

# ============================================================
# 4) FEATURE ENGINEERING
# ============================================================

patch_features <- patch_features %>%
  arrange(game, event_date, event_id) %>%
  group_by(game) %>%
  mutate(
    patch_number = row_number(),
    days_since_release = as.numeric(event_date - release_date)
  ) %>%
  ungroup()

# ============================================================
# 5) NEW SEASON PATCH (FIXED — USING any())
# ============================================================

patch_features <- patch_features %>%
  rowwise() %>%
  mutate(
    new_season_patch = case_when(
      
      game == "Overwatch 2" ~ as.integer(
        any(abs(event_date - ow_starts) <= 1)
      ),
      
      game == "Marvel Rivals" ~ as.integer(
        any(abs(event_date - marvel_starts) <= 5)
      ),
      
      game == "Apex Legends" ~ as.integer(
        any(abs(event_date - apex_starts) <= 1)
      ),
      
      TRUE ~ 0L
    )
  ) %>%
  ungroup()

# ============================================================
# 6) OPTIONAL TRANSFORMS
# ============================================================

patch_features <- patch_features %>%
  mutate(
    log_days_since_release = log1p(days_since_release),
    log_total_chars = log1p(total_chars)
  )

# ============================================================
# 7) VALIDATION
# ============================================================

cat("\n--- NEW SEASON CHECK ---\n")

patch_features %>%
  group_by(game) %>%
  summarise(
    season_patches = sum(new_season_patch),
    total_patches = n(),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- PATCH NUMBER CHECK ---\n")

patch_features %>%
  group_by(game) %>%
  summarise(
    max_patch_number = max(patch_number),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- RELEASE TIME CHECK ---\n")

patch_features %>%
  group_by(game) %>%
  summarise(
    min_days = min(days_since_release, na.rm = TRUE),
    max_days = max(days_since_release, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# ============================================================
# 8) SAVE
# ============================================================

write_csv(
  patch_features,
  "data_processed/patch_levers_with_controls.csv"
)

cat("\n✅ DONE — Controls added (ALL GAMES, FIXED)\n")