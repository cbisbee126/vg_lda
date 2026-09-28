# ============================================================
# 04 BUILD STUDY 2 ENGAGEMENT DATA
# ============================================================

rm(list = ls())

library(dplyr)
library(tidyr)
library(readr)
library(lubridate)

dir.create(
  "data/interim/study2",
  showWarnings = FALSE,
  recursive = TRUE
)


# ============================================================
# 1) LOAD COMMUNICATION DATA
# ============================================================

communications <- read_csv(
  "data/interim/study2/update_progression_with_controls.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),
    f2p_start = as.Date(f2p_start)
  ) |>
  arrange(
    game,
    event_date
  )


# Communication spacing

communications <- communications |>
  group_by(game) |>
  mutate(
    days_since_previous = as.numeric(
      event_date - lag(event_date)
    ),

    days_until_next = as.numeric(
      lead(event_date) - event_date
    ),

    overlap_7d =
      coalesce(days_since_previous <= 7, FALSE) |
      coalesce(days_until_next <= 7, FALSE),

    overlap_14d =
      coalesce(days_since_previous <= 14, FALSE) |
      coalesce(days_until_next <= 14, FALSE)
  ) |>
  ungroup()


# Basic checks

stopifnot(
  !anyDuplicated(communications$update_id),
  !anyDuplicated(
    communications[c("game", "event_date")]
  )
)


# ============================================================
# 2) LOAD STEAM ENGAGEMENT DATA
# ============================================================

analysis_end <- as.Date("2026-09-25")


# Handles both already-parsed Steam dates and character dates

parse_steam_date <- function(x) {

  if (inherits(x, "POSIXt") || inherits(x, "Date")) {
    return(as.Date(x))
  }

  as.Date(
    parse_date_time(
      as.character(x),
      orders = c(
        "ymd HMS",
        "ymd HM",
        "ymd",
        "mdy HMS",
        "mdy HM",
        "mdy"
      ),
      tz = "UTC"
    )
  )
}


read_steam <- function(path, game_name) {

  read_csv(
    path,
    show_col_types = FALSE
  ) |>
    transmute(
      game = game_name,

      calendar_date =
        parse_steam_date(DateTime),

      avg_players_raw =
        parse_number(
          as.character(`Average Players`)
        ),

      players =
        parse_number(
          as.character(Players)
        )
    ) |>
    mutate(
      engagement = coalesce(
        avg_players_raw,
        players
      )
    ) |>
    filter(
      !is.na(calendar_date),
      !is.na(engagement),
      calendar_date <= analysis_end
    )
}


steam_raw <- bind_rows(

  read_steam(
    "data/raw/study2/apex_steam_data.csv",
    "Apex Legends"
  ),

  read_steam(
    "data/raw/study2/marvel_steam_data.csv",
    "Marvel Rivals"
  ),

  read_steam(
    "data/raw/study2/overwatch_steam_data.csv",
    "Overwatch 2"
  ),

  read_steam(
    "data/raw/study2/brawlhalla_steam_data.csv",
    "Brawlhalla"
  ),

  read_steam(
    "data/raw/study2/the_finals_steam_data.csv",
    "THE FINALS"
  ),

  read_steam(
    "data/raw/study2/war_thunder_steam_data.csv",
    "War Thunder"
  ),

  read_steam(
    "data/raw/study2/warframe_steam_data.csv",
    "Warframe"
  ),

  read_steam(
    "data/raw/study2/counter_strike_steam_data.csv",
    "Counter-Strike 2"
  )
)


# ============================================================
# 3) CREATE DAILY STEAM ENGAGEMENT
# ============================================================

steam_daily <- steam_raw |>
  group_by(
    game,
    calendar_date
  ) |>
  summarise(
    avg_players = mean(
      engagement,
      na.rm = TRUE
    ),

    .groups = "drop"
  ) |>
  mutate(
    log_avg_players =
      log1p(avg_players)
  ) |>
  arrange(
    game,
    calendar_date
  )


# Steam coverage

steam_coverage <- steam_daily |>
  group_by(game) |>
  summarise(
    steam_start =
      min(calendar_date),

    steam_end =
      max(calendar_date),

    steam_days =
      n(),

    .groups = "drop"
  )


cat("\n--- STEAM COVERAGE ---\n")

steam_coverage |>
  print(
    n = Inf,
    width = Inf
  )


stopifnot(
  n_distinct(steam_daily$game) == 8
)


# ============================================================
# 4) KEEP COMMUNICATIONS WITH STEAM COVERAGE
# ============================================================

communications_model <- communications |>
  left_join(
    steam_coverage,
    by = "game"
  ) |>
  filter(
    event_date >= steam_start,
    event_date <= steam_end
  ) |>
  mutate(
    game_age_days =
      as.numeric(
        event_date - f2p_start
      ),

    log_game_age_days =
      log1p(game_age_days)
  )


cat(
  "\nCommunications loaded:",
  nrow(communications),
  "\n"
)

cat(
  "Within Steam coverage:",
  nrow(communications_model),
  "\n"
)

cat(
  "Dropped outside Steam coverage:",
  nrow(communications) -
    nrow(communications_model),
  "\n"
)


# ============================================================
# 5) BUILD EVENT WINDOWS
# ============================================================

window_rows <- communications_model |>
  select(
    game,
    update_id,
    event_date,
    f2p_start
  ) |>
  crossing(
    day = c(
      -14:-1,
      1:14
    )
  ) |>
  mutate(
    calendar_date =
      event_date + day
  ) |>
  left_join(
    steam_daily,
    by = c(
      "game",
      "calendar_date"
    )
  )


# ============================================================
# 6) CREATE 7-DAY AND 14-DAY OUTCOMES
# ============================================================

window_summary <- window_rows |>
  group_by(
    game,
    update_id,
    event_date,
    f2p_start
  ) |>
  summarise(

    # 7-day window

    n_pre_7d =
      sum(
        day >= -7 &
          day <= -1 &
          !is.na(avg_players)
      ),

    n_post_7d =
      sum(
        day >= 1 &
          day <= 7 &
          !is.na(avg_players)
      ),

    pre_avg_players_7d =
      mean(
        avg_players[
          day >= -7 &
            day <= -1
        ],
        na.rm = TRUE
      ),

    post_avg_players_7d =
      mean(
        avg_players[
          day >= 1 &
            day <= 7
        ],
        na.rm = TRUE
      ),

    pre_log_players_7d =
      mean(
        log_avg_players[
          day >= -7 &
            day <= -1
        ],
        na.rm = TRUE
      ),

    post_log_players_7d =
      mean(
        log_avg_players[
          day >= 1 &
            day <= 7
        ],
        na.rm = TRUE
      ),


    # 14-day window

    n_pre_14d =
      sum(
        day >= -14 &
          day <= -1 &
          !is.na(avg_players)
      ),

    n_post_14d =
      sum(
        day >= 1 &
          day <= 14 &
          !is.na(avg_players)
      ),

    pre_avg_players_14d =
      mean(
        avg_players[
          day >= -14 &
            day <= -1
        ],
        na.rm = TRUE
      ),

    post_avg_players_14d =
      mean(
        avg_players[
          day >= 1 &
            day <= 14
        ],
        na.rm = TRUE
      ),

    pre_log_players_14d =
      mean(
        log_avg_players[
          day >= -14 &
            day <= -1
        ],
        na.rm = TRUE
      ),

    post_log_players_14d =
      mean(
        log_avg_players[
          day >= 1 &
            day <= 14
        ],
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>
  mutate(

    # Replace NaN from entirely missing windows

    across(
      c(
        pre_avg_players_7d,
        post_avg_players_7d,
        pre_log_players_7d,
        post_log_players_7d,
        pre_avg_players_14d,
        post_avg_players_14d,
        pre_log_players_14d,
        post_log_players_14d
      ),
      ~ if_else(
        is.nan(.x),
        NA_real_,
        .x
      )
    ),


    # Make sure pre-update windows stay within F2P period

    valid_f2p_7d =
      event_date - 7 >= f2p_start,

    valid_f2p_14d =
      event_date - 14 >= f2p_start,


    # Complete windows

    complete_7d =
      n_pre_7d == 7 &
      n_post_7d == 7 &
      valid_f2p_7d,

    complete_14d =
      n_pre_14d == 14 &
      n_post_14d == 14 &
      valid_f2p_14d,


    # Primary outcome

    engagement_change_7d =
      post_log_players_7d -
      pre_log_players_7d,


    # Robustness outcome

    engagement_change_14d =
      post_log_players_14d -
      pre_log_players_14d,


    # Easier-to-interpret raw percentage changes

    pct_player_change_7d =
      if_else(
        pre_avg_players_7d > 0,
        post_avg_players_7d /
          pre_avg_players_7d - 1,
        NA_real_
      ),

    pct_player_change_14d =
      if_else(
        pre_avg_players_14d > 0,
        post_avg_players_14d /
          pre_avg_players_14d - 1,
        NA_real_
      )
  )


# ============================================================
# 7) FINAL EVENT DATA
# ============================================================

event_data <- communications_model |>
  left_join(
    window_summary,
    by = c(
      "game",
      "update_id",
      "event_date",
      "f2p_start"
    )
  ) |>
  select(
    -clean_text,
    -announcement_ids
  )


# ============================================================
# 8) INSPECT
# ============================================================

cat("\n--- ENGAGEMENT WINDOWS ---\n")

event_data |>
  summarise(
    updates = n(),

    n_complete_7d =
      sum(
        complete_7d,
        na.rm = TRUE
      ),

    pct_complete_7d =
      mean(
        complete_7d,
        na.rm = TRUE
      ),

    n_complete_14d =
      sum(
        complete_14d,
        na.rm = TRUE
      ),

    pct_complete_14d =
      mean(
        complete_14d,
        na.rm = TRUE
      ),

    pct_overlap_7d =
      mean(
        overlap_7d,
        na.rm = TRUE
      ),

    pct_overlap_14d =
      mean(
        overlap_14d,
        na.rm = TRUE
      ),

    mean_change_7d =
      mean(
        engagement_change_7d[
          complete_7d
        ],
        na.rm = TRUE
      ),

    mean_change_14d =
      mean(
        engagement_change_14d[
          complete_14d
        ],
        na.rm = TRUE
      )
  ) |>
  print(width = Inf)


cat("\n--- BY GAME ---\n")

event_data |>
  group_by(game) |>
  summarise(
    updates = n(),

    n_complete_7d =
      sum(
        complete_7d,
        na.rm = TRUE
      ),

    pct_complete_7d =
      mean(
        complete_7d,
        na.rm = TRUE
      ),

    n_complete_14d =
      sum(
        complete_14d,
        na.rm = TRUE
      ),

    pct_complete_14d =
      mean(
        complete_14d,
        na.rm = TRUE
      ),

    pct_overlap_7d =
      mean(
        overlap_7d,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 9) BASIC CHECKS
# ============================================================

stopifnot(
  !anyDuplicated(
    event_data$update_id
  ),

  !anyDuplicated(
    event_data[
      c(
        "game",
        "event_date"
      )
    ]
  ),

  all(
    event_data$game_age_days >= 0
  ),

  all(
    steam_daily$calendar_date <= analysis_end
  )
)


# ============================================================
# 10) SAVE
# ============================================================

write_csv(
  event_data,
  "data/interim/study2/study2_event_data.csv"
)