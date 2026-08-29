# ============================================================
# 03b CHECK COMMUNICATION SPACING
# ============================================================
#
# PURPOSE
# -------
# Examine how frequently qualifying official update-related
# communications occur for each game.
#
# TWO RELATED DIAGNOSTICS
# -----------------------
# 1) Consecutive communication gaps
#    - Mean days between consecutive communications
#    - Median days between consecutive communications
#
# 2) Nearest communication spacing
#    - Distance to the closest qualifying communication
#      before OR after each focal communication
#    - Percent with another communication within
#      3, 5, and 7 days
#
# Nothing is saved.
# Results print to the console only.
# ============================================================

rm(list = ls())

library(tidyverse)
library(readr)

# ============================================================
# 1) LOAD COMMUNICATION DATA
# ============================================================

communication_data <- read_csv(
  "data/interim/study2/patch_levers_with_controls.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),
    game = as.character(game)
  )

cat("\n============================================================\n")
cat("COMMUNICATION SPACING CHECK\n")
cat("============================================================\n")

cat(
  "\nRaw communication rows:",
  nrow(communication_data),
  "\n"
)

cat(
  "Games:",
  n_distinct(communication_data$game),
  "\n"
)

# ============================================================
# 2) UNIQUE GAME-COMMUNICATION DAYS
# ============================================================

# Step 01 should already contain one row per game-day,
# but this ensures the spacing calculation uses unique dates.

communication_daily <- communication_data |>
  distinct(
    game,
    event_date
  ) |>
  arrange(
    game,
    event_date
  )

cat(
  "Unique game-communication days:",
  nrow(communication_daily),
  "\n"
)

# ============================================================
# 3) CALCULATE PREVIOUS, NEXT, AND NEAREST GAPS
# ============================================================

communication_gaps <- communication_daily |>
  group_by(game) |>
  arrange(
    event_date,
    .by_group = TRUE
  ) |>
  mutate(
    previous_communication =
      lag(event_date),

    next_communication =
      lead(event_date),

    days_since_previous =
      as.numeric(
        event_date - previous_communication
      ),

    days_until_next =
      as.numeric(
        next_communication - event_date
      ),

    nearest_communication_gap = case_when(
      !is.na(days_since_previous) &
        !is.na(days_until_next) ~
        pmin(
          days_since_previous,
          days_until_next
        ),

      !is.na(days_since_previous) ~
        days_since_previous,

      !is.na(days_until_next) ~
        days_until_next,

      TRUE ~
        NA_real_
    )
  ) |>
  ungroup()

# ============================================================
# 4) CONSECUTIVE GAP SUMMARY BY GAME
# ============================================================

spacing_by_game <- communication_gaps |>
  group_by(game) |>
  summarise(
    communications = n(),

    observed_consecutive_gaps =
      sum(!is.na(days_since_previous)),

    mean_days_between =
      mean(
        days_since_previous,
        na.rm = TRUE
      ),

    median_days_between =
      median(
        days_since_previous,
        na.rm = TRUE
      ),

    min_days_between =
      min(
        days_since_previous,
        na.rm = TRUE
      ),

    max_days_between =
      max(
        days_since_previous,
        na.rm = TRUE
      ),

    mean_nearest_gap =
      mean(
        nearest_communication_gap,
        na.rm = TRUE
      ),

    median_nearest_gap =
      median(
        nearest_communication_gap,
        na.rm = TRUE
      ),

    pct_another_within_3_days =
      mean(
        nearest_communication_gap <= 3,
        na.rm = TRUE
      ),

    pct_another_within_5_days =
      mean(
        nearest_communication_gap <= 5,
        na.rm = TRUE
      ),

    pct_another_within_7_days =
      mean(
        nearest_communication_gap <= 7,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>
  arrange(
    median_days_between
  )

cat("\n============================================================\n")
cat("COMMUNICATION SPACING BY GAME\n")
cat("============================================================\n")

print(
  spacing_by_game,
  n = Inf,
  width = Inf
)

# ============================================================
# 5) OVERALL CONSECUTIVE GAP SUMMARY
# ============================================================

overall_consecutive_spacing <- communication_gaps |>
  filter(
    !is.na(days_since_previous)
  ) |>
  summarise(
    observed_consecutive_gaps = n(),

    mean_days_between =
      mean(days_since_previous),

    median_days_between =
      median(days_since_previous),

    min_days_between =
      min(days_since_previous),

    max_days_between =
      max(days_since_previous),

    pct_previous_within_3_days =
      mean(days_since_previous <= 3),

    pct_previous_within_5_days =
      mean(days_since_previous <= 5),

    pct_previous_within_7_days =
      mean(days_since_previous <= 7)
  )

cat("\n============================================================\n")
cat("OVERALL CONSECUTIVE COMMUNICATION GAPS\n")
cat("============================================================\n")

print(
  overall_consecutive_spacing,
  width = Inf
)

# ============================================================
# 6) OVERALL NEAREST-COMMUNICATION SUMMARY
# ============================================================

overall_nearest_spacing <- communication_gaps |>
  filter(
    !is.na(nearest_communication_gap)
  ) |>
  summarise(
    communication_days = n(),

    mean_nearest_gap =
      mean(nearest_communication_gap),

    median_nearest_gap =
      median(nearest_communication_gap),

    min_nearest_gap =
      min(nearest_communication_gap),

    max_nearest_gap =
      max(nearest_communication_gap),

    pct_another_within_3_days =
      mean(nearest_communication_gap <= 3),

    pct_another_within_5_days =
      mean(nearest_communication_gap <= 5),

    pct_another_within_7_days =
      mean(nearest_communication_gap <= 7)
  )

cat("\n============================================================\n")
cat("OVERALL NEAREST-COMMUNICATION SPACING\n")
cat("============================================================\n")

print(
  overall_nearest_spacing,
  width = Inf
)

# ============================================================
# 7) DISTRIBUTION OF CONSECUTIVE GAPS
# ============================================================

cat("\n============================================================\n")
cat("CONSECUTIVE GAP DISTRIBUTION\n")
cat("============================================================\n")

communication_gaps |>
  filter(
    !is.na(days_since_previous)
  ) |>
  mutate(
    gap_group = case_when(
      days_since_previous <= 1 ~
        "1 day",

      days_since_previous <= 3 ~
        "2-3 days",

      days_since_previous <= 5 ~
        "4-5 days",

      days_since_previous <= 7 ~
        "6-7 days",

      days_since_previous <= 14 ~
        "8-14 days",

      TRUE ~
        "15+ days"
    ),

    gap_group = factor(
      gap_group,
      levels = c(
        "1 day",
        "2-3 days",
        "4-5 days",
        "6-7 days",
        "8-14 days",
        "15+ days"
      )
    )
  ) |>
  count(
    gap_group
  ) |>
  mutate(
    pct = n / sum(n)
  ) |>
  print(
    n = Inf,
    width = Inf
  )

# ============================================================
# 8) DISTRIBUTION OF NEAREST COMMUNICATION GAPS
# ============================================================

cat("\n============================================================\n")
cat("NEAREST-COMMUNICATION GAP DISTRIBUTION\n")
cat("============================================================\n")

communication_gaps |>
  filter(
    !is.na(nearest_communication_gap)
  ) |>
  mutate(
    gap_group = case_when(
      nearest_communication_gap <= 1 ~
        "1 day",

      nearest_communication_gap <= 3 ~
        "2-3 days",

      nearest_communication_gap <= 5 ~
        "4-5 days",

      nearest_communication_gap <= 7 ~
        "6-7 days",

      nearest_communication_gap <= 14 ~
        "8-14 days",

      TRUE ~
        "15+ days"
    ),

    gap_group = factor(
      gap_group,
      levels = c(
        "1 day",
        "2-3 days",
        "4-5 days",
        "6-7 days",
        "8-14 days",
        "15+ days"
      )
    )
  ) |>
  count(
    gap_group
  ) |>
  mutate(
    pct = n / sum(n)
  ) |>
  print(
    n = Inf,
    width = Inf
  )

# ============================================================
# 9) KEY WINDOW DIAGNOSTICS
# ============================================================

shortest_game_median <- spacing_by_game |>
  slice_min(
    median_days_between,
    n = 1,
    with_ties = TRUE
  ) |>
  select(
    game,
    median_days_between
  )

cat("\n============================================================\n")
cat("KEY WINDOW DIAGNOSTICS\n")
cat("============================================================\n")

cat("\nShortest game-level median spacing:\n")

print(
  shortest_game_median,
  n = Inf,
  width = Inf
)

cat("\nOverall consecutive spacing:\n")

overall_consecutive_spacing |>
  select(
    mean_days_between,
    median_days_between
  ) |>
  print(
    width = Inf
  )

cat("\nPercent with another communication nearby:\n")

overall_nearest_spacing |>
  select(
    pct_another_within_3_days,
    pct_another_within_5_days,
    pct_another_within_7_days
  ) |>
  print(
    width = Inf
  )

# ============================================================
# 10) DONE
# ============================================================

cat("\n============================================================\n")
cat("DONE\n")
cat("============================================================\n")

cat(
  "\nUse consecutive-gap mean/median statistics to describe ",
  "the communication cadence. Use nearest-communication ",
  "statistics to describe how often focal communications ",
  "have another qualifying communication nearby.\n",
  sep = ""
)