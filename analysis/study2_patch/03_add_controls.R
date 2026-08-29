# ============================================================
# 03 ADD COMMUNICATION CONTROLS
# ============================================================
#
# PURPOSE
# -------
# Add the communication-level controls used in Study 2.
#
# INPUT
# -----
# data/interim/study2/patch_levers_sentence.csv
#
# OUTPUT
# ------
# data/interim/study2/patch_levers_with_controls.csv
#
# CONTROLS
# --------
# total_chars / log_total_chars
# avg_sentence_chars / log_avg_sentence_chars
# season_related_communication =
#   title explicitly references season/midseason
#
# Game age is created in Step 04 from the observed Steam series.
# ============================================================

rm(list = ls())

library(tidyverse)
library(readr)
library(stringr)
library(lubridate)

dir.create(
  "data/interim/study2",
  showWarnings = FALSE,
  recursive = TRUE
)

# ============================================================
# 1) LOAD EMPHASIS DATA
# ============================================================

patch_features <- read_csv(
  "data/interim/study2/patch_levers_sentence.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),
    event_id = as.character(event_id),
    patch_title = as.character(patch_title),
    game = as.character(game),
    full_text = as.character(full_text),

    total_chars = as.numeric(char_count),
    total_chars = if_else(
      is.na(total_chars) | total_chars <= 0,
      as.numeric(total_sentence_chars),
      total_chars
    ),

    avg_sentence_chars = as.numeric(avg_sentence_chars),
    avg_sentence_chars = if_else(
      is.na(avg_sentence_chars) & total_sentences > 0,
      as.numeric(total_sentence_chars) / as.numeric(total_sentences),
      avg_sentence_chars
    )
  ) |>
  arrange(game, event_date)

cat("\nLoaded rows:", nrow(patch_features), "\n")
cat("Games:", n_distinct(patch_features$game), "\n")

if (
  nrow(
    patch_features |>
      count(game, event_date) |>
      filter(n > 1)
  ) > 0
) {
  stop("Duplicate game-day communications found in Step 03.")
}

if (
  nrow(
    patch_features |>
      count(event_id) |>
      filter(n > 1)
  ) > 0
) {
  stop("Duplicate event_id values found in Step 03.")
}

# ============================================================
# 2) CREATE CONTROLS
# ============================================================

patch_features <- patch_features |>
  mutate(
    log_total_chars = log1p(total_chars),

    log_avg_sentence_chars = log1p(avg_sentence_chars),

    # Indicates whether the communication title explicitly
    # references a season or midseason.
    # This is NOT a verified season-launch indicator.
    season_related_communication = as.integer(
      str_detect(
        str_to_lower(coalesce(patch_title, "")),
        "\\b(season|midseason|mid-season)\\b"
      )
    ),

    year = year(event_date),
    month = month(event_date),

    weekday = wday(
      event_date,
      label = TRUE,
      abbr = TRUE,
      week_start = 1
    )
  )

# ============================================================
# 3) CONSOLE DIAGNOSTICS
# ============================================================

cat("\n--- CONTROL COMPLETENESS ---\n")

patch_features |>
  summarise(
    rows = n(),

    missing_total_chars =
      sum(is.na(total_chars)),

    missing_log_total_chars =
      sum(is.na(log_total_chars)),

    missing_avg_sentence_chars =
      sum(is.na(avg_sentence_chars)),

    missing_log_avg_sentence_chars =
      sum(is.na(log_avg_sentence_chars)),

    missing_season_related =
      sum(is.na(season_related_communication))
  ) |>
  print(width = Inf)

cat("\n--- SEASON-RELATED COMMUNICATIONS ---\n")

patch_features |>
  group_by(game) |>
  summarise(
    communication_days = n(),

    season_related_days =
      sum(season_related_communication),

    pct_season_related =
      mean(season_related_communication),

    mean_rel_seasonal_season =
      mean(
        rel_seasonal[
          season_related_communication == 1
        ],
        na.rm = TRUE
      ),

    mean_rel_seasonal_other =
      mean(
        rel_seasonal[
          season_related_communication == 0
        ],
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>
  arrange(desc(communication_days)) |>
  print(
    n = Inf,
    width = Inf
  )

cat("\n--- LENGTH / EMPHASIS CORRELATIONS ---\n")

patch_features |>
  select(
    log_total_chars,
    log_avg_sentence_chars,
    rel_competitive,
    rel_cosmetic,
    rel_seasonal,
    rel_difficulty
  ) |>
  cor(
    use = "pairwise.complete.obs"
  ) |>
  round(3) |>
  print()

# ============================================================
# 4) SAVE ONE FILE
# ============================================================

output_file <-
  "data/interim/study2/patch_levers_with_controls.csv"

write_csv(
  patch_features,
  output_file
)

cat("\nDONE - Step 03\n")
cat(
  "Update-communication days:",
  nrow(patch_features),
  "\n"
)
cat("Saved:", output_file, "\n")