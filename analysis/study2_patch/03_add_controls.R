# ============================================================
# 03 ADD COMMUNICATION CONTROLS
# ============================================================

rm(list = ls())

library(dplyr)
library(stringr)
library(readr)


# Load emphasis data

updates <- read_csv(
  "data/interim/study2/update_progression_emphasis.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date)
  )


# Add communication controls

updates <- updates |>
  mutate(

    # Communication length
    log_char_count = log1p(char_count),

    # Title explicitly references a season or midseason
    season_title = as.integer(
      str_detect(
        titles,
        regex(
          "\\bseason\\b|\\bmid[- ]?season\\b",
          ignore_case = TRUE
        )
      )
    )
  )


# Inspect controls

updates |>
  summarise(
    updates = n(),
    mean_chars = mean(char_count),
    median_chars = median(char_count),
    season_titles = sum(season_title),
    pct_season_titles = mean(season_title)
  ) |>
  print(width = Inf)


# Inspect by game

updates |>
  group_by(game) |>
  summarise(
    updates = n(),
    mean_chars = mean(char_count),
    season_titles = sum(season_title),
    pct_season_titles = mean(season_title),
    .groups = "drop"
  ) |>
  print(n = Inf)


# Check relationship with seasonal emphasis

updates |>
  group_by(season_title) |>
  summarise(
    updates = n(),
    mean_rel_seasonal = mean(rel_seasonal),
    .groups = "drop"
  ) |>
  print()


# Basic checks

stopifnot(
  !anyDuplicated(updates$update_id),
  !anyDuplicated(updates[c("game", "event_date")]),
  !any(is.na(updates$log_char_count)),
  !any(is.na(updates$season_title))
)


# Save

write_csv(
  updates,
  "data/interim/study2/update_progression_with_controls.csv"
)