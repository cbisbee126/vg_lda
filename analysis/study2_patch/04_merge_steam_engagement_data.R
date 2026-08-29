# ============================================================
# 04 BUILD STUDY 2 ENGAGEMENT DATA
# ============================================================
#
# PURPOSE
# -------
# Merge communication features with daily Steam engagement and
# create the analysis datasets used in Step 06.
#
# ANALYSES SUPPORTED
# ------------------
#
# 1) PRIMARY PRE/POST MODEL - EXCLUDES DAY 0
#
#    Pre  = mean logged engagement on Days -3 to -1
#    Post = mean logged engagement on Days +1 to +3
#
#    Change = Post - Pre
#
#    Day 0 is excluded because communications can occur at
#    different times within the calendar day.
#
#
# 2) DAY 0 SENSITIVITY MODEL
#
#    Pre  = mean logged engagement on Days -3 to -1
#    Post = mean logged engagement on Days 0 to +2
#
#    Change = Post - Pre
#
#    This keeps the pre- and post-periods the same length while
#    testing whether results depend on excluding Day 0.
#
#
# 3) DAILY DISTRIBUTED-LAG ROBUSTNESS MODEL
#
#    One row per actual game-date.
#
#    Step 06 will estimate communication exposure on:
#      Day 0
#      Day +1
#      Day +2
#      Day +3
#
#    Recent communications enter simultaneously so nearby
#    communications do not need to be treated as isolated events.
#
#
# IMPORTANT
# ---------
# The complete calendar is retained so a one-day lag always
# represents one actual calendar day rather than the previous
# observed Steam row.
#
#
# OUTPUT
# ------
# data/interim/study2/study2_analysis_data.rds
#
# The saved object contains:
#   event_data
#   daily_panel
#   steam_coverage
#
# ============================================================


rm(list = ls())


# ============================================================
# 0) PACKAGES
# ============================================================

library(tidyverse)
library(lubridate)
library(readr)


# ============================================================
# 1) OUTPUT DIRECTORY + HELPERS
# ============================================================

dir.create(
  "data/interim/study2",
  showWarnings = FALSE,
  recursive = TRUE
)


# ------------------------------------------------------------
# Standardize a numeric variable
# ------------------------------------------------------------

scale2 <- function(x) {

  x_mean <- mean(
    x,
    na.rm = TRUE
  )

  x_sd <- sd(
    x,
    na.rm = TRUE
  )

  if (
    is.na(x_sd) ||
    x_sd == 0
  ) {

    return(
      rep(
        0,
        length(x)
      )
    )
  }

  as.numeric(
    (x - x_mean) / x_sd
  )
}


# ------------------------------------------------------------
# Safe Steam datetime parser
# ------------------------------------------------------------

parse_steam_datetime <- function(x) {

  x <- trimws(
    as.character(x)
  )

  x[x == ""] <- NA_character_


  out <- rep(
    as.POSIXct(
      NA,
      tz = "UTC"
    ),
    length(x)
  )


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

    needs_parse <-
      is.na(out) &
      !is.na(x)

    if (!any(needs_parse)) {
      break
    }


    parsed <- suppressWarnings(
      as.POSIXct(
        x[needs_parse],
        format = fmt,
        tz = "UTC"
      )
    )


    idx <- which(
      needs_parse
    )


    out[
      idx[
        !is.na(parsed)
      ]
    ] <-
      parsed[
        !is.na(parsed)
      ]
  }


  # Final generic parse attempt

  needs_parse <-
    is.na(out) &
    !is.na(x)


  if (any(needs_parse)) {

    parsed <- suppressWarnings(
      as.POSIXct(
        x[needs_parse],
        tz = "UTC"
      )
    )


    idx <- which(
      needs_parse
    )


    out[
      idx[
        !is.na(parsed)
      ]
    ] <-
      parsed[
        !is.na(parsed)
      ]
  }


  out
}


# ------------------------------------------------------------
# Load one Steam engagement file
# ------------------------------------------------------------

load_steam_file <- function(
  path,
  game_name
) {

  read_csv(
    path,
    show_col_types = FALSE
  ) |>
    mutate(
      DateTime = as.character(DateTime),
      Players = as.numeric(Players),
      `Average Players` = as.numeric(`Average Players`),
      game = game_name
    )
}


# ============================================================
# 2) LOAD COMMUNICATION FEATURES
# ============================================================

communications <- read_csv(
  "data/interim/study2/patch_levers_with_controls.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),
    event_id = as.character(event_id),
    game = as.character(game)
  ) |>
  arrange(
    game,
    event_date
  )


cat(
  "\n============================================================\n"
)

cat(
  "04 BUILD STUDY 2 ENGAGEMENT DATA\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nCommunication days loaded:",
  nrow(communications),
  "\n"
)


cat(
  "Games:",
  n_distinct(communications$game),
  "\n"
)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

duplicate_days <- communications |>
  count(
    game,
    event_date
  ) |>
  filter(
    n > 1
  )


if (nrow(duplicate_days) > 0) {

  stop(
    "Duplicate communication game-days found in Step 04 input."
  )
}


duplicate_ids <- communications |>
  count(
    event_id
  ) |>
  filter(
    n > 1
  )


if (nrow(duplicate_ids) > 0) {

  stop(
    "Duplicate event_id values found in Step 04 input."
  )
}


# ============================================================
# 3) LOAD + COMBINE STEAM ENGAGEMENT DATA
# ============================================================

steam_raw <- bind_rows(

  load_steam_file(
    "data/raw/study2/apex_steam_data.csv",
    "Apex Legends"
  ),

  load_steam_file(
    "data/raw/study2/marvel_steam_data.csv",
    "Marvel Rivals"
  ),

  load_steam_file(
    "data/raw/study2/overwatch_steam_data.csv",
    "Overwatch 2"
  ),

  load_steam_file(
    "data/raw/study2/brawlhalla_steam_data.csv",
    "Brawlhalla"
  ),

  load_steam_file(
    "data/raw/study2/the_finals_steam_data.csv",
    "THE FINALS"
  ),

  load_steam_file(
    "data/raw/study2/war_thunder_steam_data.csv",
    "War Thunder"
  ),

  load_steam_file(
    "data/raw/study2/pubg_steam_data.csv",
    "PUBG: BATTLEGROUNDS"
  ),

  load_steam_file(
    "data/raw/study2/counter_strike_steam_data.csv",
    "Counter-Strike 2"
  )

) |>
  rename(
    datetime = DateTime,
    players = Players,
    avg_players_raw = `Average Players`
  ) |>
  mutate(
    datetime =
      parse_steam_datetime(
        datetime
      ),

    calendar_date =
      as.Date(
        datetime
      )
  ) |>
  filter(
    !is.na(game),
    !is.na(calendar_date)
  )


cat(
  "\nRaw Steam rows:",
  nrow(steam_raw),
  "\n"
)


# ============================================================
# 4) COLLAPSE STEAM DATA TO ONE ROW PER GAME-DAY
# ============================================================

steam_daily <- steam_raw |>
  group_by(
    game,
    calendar_date
  ) |>
  summarise(

    n_steam_rows =
      n(),

    # Prefer the supplied Average Players measure when
    # available. Fall back to Players otherwise.

    avg_players =
      if (
        any(
          !is.na(avg_players_raw)
        )
      ) {

        mean(
          avg_players_raw,
          na.rm = TRUE
        )

      } else {

        mean(
          players,
          na.rm = TRUE
        )
      },

    .groups = "drop"
  ) |>
  mutate(

    avg_players =
      if_else(
        is.nan(avg_players),
        NA_real_,
        avg_players
      ),

    log_avg_players_daily =
      log1p(
        avg_players
      )
  ) |>
  arrange(
    game,
    calendar_date
  )


cat(
  "Observed Steam game-days:",
  nrow(steam_daily),
  "\n"
)


if (
  nrow(
    steam_daily |>
      count(
        game,
        calendar_date
      ) |>
      filter(
        n > 1
      )
  ) > 0
) {

  stop(
    "Duplicate game-day rows remain after Steam aggregation."
  )
}


# ============================================================
# 5) STEAM COVERAGE
# ============================================================

steam_coverage <- steam_daily |>
  group_by(
    game
  ) |>
  summarise(

    steam_start_date =
      min(
        calendar_date,
        na.rm = TRUE
      ),

    steam_end_date =
      max(
        calendar_date,
        na.rm = TRUE
      ),

    observed_steam_days =
      n(),

    .groups = "drop"
  )


cat(
  "\n--- STEAM COVERAGE ---\n"
)


print(
  steam_coverage,
  n = Inf,
  width = Inf
)


# ============================================================
# 6) CREATE COMPLETE GAME-DAY CALENDAR
# ============================================================

# Missing calendar dates are retained as rows with missing
# engagement. This ensures that lagging by one row later in
# the analysis corresponds to exactly one calendar day.

steam_calendar <- steam_daily |>
  group_by(
    game
  ) |>
  complete(
    calendar_date =
      seq.Date(
        min(
          calendar_date,
          na.rm = TRUE
        ),
        max(
          calendar_date,
          na.rm = TRUE
        ),
        by = "day"
      )
  ) |>
  ungroup() |>
  arrange(
    game,
    calendar_date
  )


cat(
  "\nComplete calendar rows:",
  nrow(steam_calendar),
  "\n"
)


# ============================================================
# 7) RETAIN COMMUNICATIONS WITHIN STEAM COVERAGE
# ============================================================

communications_model <- communications |>
  left_join(
    steam_coverage,
    by = "game"
  ) |>
  filter(
    event_date >= steam_start_date,
    event_date <= steam_end_date
  ) |>
  mutate(

    game_age_days =
      as.numeric(
        event_date -
          steam_start_date
      ),

    log_game_age_days =
      log1p(
        game_age_days
      )
  )


cat(
  "\nCommunication days within Steam coverage:",
  nrow(communications_model),
  "\n"
)


cat(
  "Dropped outside Steam coverage:",
  nrow(communications) -
    nrow(communications_model),
  "\n"
)


if (
  any(
    communications_model$game_age_days < 0,
    na.rm = TRUE
  )
) {

  stop(
    "Negative game-age values found after Steam-coverage restriction."
  )
}


# ============================================================
# 8) STANDARDIZE COMMUNICATION-LEVEL PREDICTORS
# ============================================================
#
# Standardization occurs across actual communication days only.
# The same standardized values are used in both pre/post models
# and the daily robustness model.
# ============================================================

communications_model <- communications_model |>
  mutate(

    z_competitive =
      scale2(
        rel_competitive
      ),

    z_cosmetic =
      scale2(
        rel_cosmetic
      ),

    z_seasonal =
      scale2(
        rel_seasonal
      ),

    z_difficulty =
      scale2(
        rel_difficulty
      ),

    z_log_total_chars =
      scale2(
        log_total_chars
      ),

    z_log_avg_sentence_chars =
      scale2(
        log_avg_sentence_chars
      ),

    z_log_game_age_days_event =
      scale2(
        log_game_age_days
      )
  )


# ============================================================
# 9) BUILD 3-DAY PRE/POST EVENT WINDOWS
# ============================================================
#
# Seven calendar dates are retrieved for each communication:
#
#   -3, -2, -1, 0, +1, +2, +3
#
# These rows are used to construct:
#
# Model 1:
#   -3:-1 versus +1:+3
#
# Model 2:
#   -3:-1 versus 0:+2
#
# ============================================================

window_rows <- communications_model |>
  select(
    game,
    event_id,
    event_date
  ) |>
  crossing(
    window_offset =
      c(
        -3L,
        -2L,
        -1L,
         0L,
         1L,
         2L,
         3L
      )
  ) |>
  mutate(
    calendar_date =
      event_date +
      window_offset
  ) |>
  left_join(
    steam_daily |>
      select(
        game,
        calendar_date,
        avg_players,
        log_avg_players_daily
      ),
    by = c(
      "game",
      "calendar_date"
    )
  )


# ============================================================
# 10) CREATE BOTH PRE/POST OUTCOMES
# ============================================================

window_summary <- window_rows |>
  group_by(
    game,
    event_id,
    event_date
  ) |>
  summarise(

    # --------------------------------------------------------
    # Shared pre-period: Days -3 to -1
    # --------------------------------------------------------

    pre_log_avg_players_3d =
      mean(
        log_avg_players_daily[
          window_offset %in%
            c(-3L, -2L, -1L)
        ],
        na.rm = TRUE
      ),

    pre_avg_players_3d =
      mean(
        avg_players[
          window_offset %in%
            c(-3L, -2L, -1L)
        ],
        na.rm = TRUE
      ),

    n_pre_days =
      sum(
        window_offset %in%
          c(-3L, -2L, -1L) &
          !is.na(
            log_avg_players_daily
          )
      ),


    # --------------------------------------------------------
    # Primary post-period: Days +1 to +3
    # --------------------------------------------------------

    post_log_avg_players_excl_day0 =
      mean(
        log_avg_players_daily[
          window_offset %in%
            c(1L, 2L, 3L)
        ],
        na.rm = TRUE
      ),

    post_avg_players_excl_day0 =
      mean(
        avg_players[
          window_offset %in%
            c(1L, 2L, 3L)
        ],
        na.rm = TRUE
      ),

    n_post_days_excl_day0 =
      sum(
        window_offset %in%
          c(1L, 2L, 3L) &
          !is.na(
            log_avg_players_daily
          )
      ),


    # --------------------------------------------------------
    # Day 0 sensitivity post-period: Days 0 to +2
    # --------------------------------------------------------

    post_log_avg_players_incl_day0 =
      mean(
        log_avg_players_daily[
          window_offset %in%
            c(0L, 1L, 2L)
        ],
        na.rm = TRUE
      ),

    post_avg_players_incl_day0 =
      mean(
        avg_players[
          window_offset %in%
            c(0L, 1L, 2L)
        ],
        na.rm = TRUE
      ),

    n_post_days_incl_day0 =
      sum(
        window_offset %in%
          c(0L, 1L, 2L) &
          !is.na(
            log_avg_players_daily
          )
      ),

    .groups = "drop"
  ) |>

  mutate(

    # Convert NaN means from entirely missing windows to NA

    across(
      c(
        pre_log_avg_players_3d,
        pre_avg_players_3d,
        post_log_avg_players_excl_day0,
        post_avg_players_excl_day0,
        post_log_avg_players_incl_day0,
        post_avg_players_incl_day0
      ),
      ~ if_else(
        is.nan(.x),
        NA_real_,
        .x
      )
    ),


    # --------------------------------------------------------
    # PRIMARY OUTCOME
    # Days +1:+3 minus Days -3:-1
    # --------------------------------------------------------

    engagement_change_excl_day0 =
      post_log_avg_players_excl_day0 -
      pre_log_avg_players_3d,


    # --------------------------------------------------------
    # DAY 0 SENSITIVITY OUTCOME
    # Days 0:+2 minus Days -3:-1
    # --------------------------------------------------------

    engagement_change_incl_day0 =
      post_log_avg_players_incl_day0 -
      pre_log_avg_players_3d,


    # --------------------------------------------------------
    # Raw-player diagnostics
    # --------------------------------------------------------

    raw_player_change_excl_day0 =
      post_avg_players_excl_day0 -
      pre_avg_players_3d,

    raw_player_change_incl_day0 =
      post_avg_players_incl_day0 -
      pre_avg_players_3d,


    pct_player_change_excl_day0 =
      if_else(
        pre_avg_players_3d > 0,
        (
          post_avg_players_excl_day0 -
            pre_avg_players_3d
        ) /
          pre_avg_players_3d,
        NA_real_
      ),

    pct_player_change_incl_day0 =
      if_else(
        pre_avg_players_3d > 0,
        (
          post_avg_players_incl_day0 -
            pre_avg_players_3d
        ) /
          pre_avg_players_3d,
        NA_real_
      ),


    # --------------------------------------------------------
    # Complete-window indicators
    # --------------------------------------------------------

    complete_excl_day0_window =
      n_pre_days == 3 &
      n_post_days_excl_day0 == 3,

    complete_incl_day0_window =
      n_pre_days == 3 &
      n_post_days_incl_day0 == 3
  )


# ============================================================
# 11) BUILD EVENT-LEVEL ANALYSIS DATA
# ============================================================

event_data <- communications_model |>
  left_join(
    window_summary,
    by = c(
      "game",
      "event_id",
      "event_date"
    )
  ) |>
  mutate(

    weekday_f =
      factor(
        wday(
          event_date,
          label = TRUE,
          abbr = TRUE,
          week_start = 1
        )
      ),

    month_f =
      factor(
        month(
          event_date
        )
      ),

    year_f =
      factor(
        year(
          event_date
        )
      ),

    game_month =
      interaction(
        game,
        format(
          event_date,
          "%Y-%m"
        ),
        drop = TRUE
      )
  )


# ============================================================
# 12) PRE/POST WINDOW DIAGNOSTICS
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "PRE/POST WINDOW CHECK\n"
)

cat(
  "============================================================\n"
)


event_data |>
  summarise(

    communication_days =
      n(),


    # Primary model

    complete_excl_day0 =
      sum(
        complete_excl_day0_window,
        na.rm = TRUE
      ),

    incomplete_excl_day0 =
      sum(
        !complete_excl_day0_window,
        na.rm = TRUE
      ),

    pct_complete_excl_day0 =
      mean(
        complete_excl_day0_window,
        na.rm = TRUE
      ),

    mean_change_excl_day0 =
      mean(
        engagement_change_excl_day0[
          complete_excl_day0_window
        ],
        na.rm = TRUE
      ),

    sd_change_excl_day0 =
      sd(
        engagement_change_excl_day0[
          complete_excl_day0_window
        ],
        na.rm = TRUE
      ),


    # Day 0 sensitivity model

    complete_incl_day0 =
      sum(
        complete_incl_day0_window,
        na.rm = TRUE
      ),

    incomplete_incl_day0 =
      sum(
        !complete_incl_day0_window,
        na.rm = TRUE
      ),

    pct_complete_incl_day0 =
      mean(
        complete_incl_day0_window,
        na.rm = TRUE
      ),

    mean_change_incl_day0 =
      mean(
        engagement_change_incl_day0[
          complete_incl_day0_window
        ],
        na.rm = TRUE
      ),

    sd_change_incl_day0 =
      sd(
        engagement_change_incl_day0[
          complete_incl_day0_window
        ],
        na.rm = TRUE
      )
  ) |>
  print(
    width = Inf
  )


# ============================================================
# 13) PRIMARY/SENSITIVITY SAMPLE BY GAME
# ============================================================

cat(
  "\n--- COMPLETE WINDOWS BY GAME ---\n"
)


event_data |>
  group_by(
    game
  ) |>
  summarise(

    communication_days =
      n(),

    complete_excl_day0 =
      sum(
        complete_excl_day0_window,
        na.rm = TRUE
      ),

    complete_incl_day0 =
      sum(
        complete_incl_day0_window,
        na.rm = TRUE
      ),

    pct_complete_excl_day0 =
      mean(
        complete_excl_day0_window,
        na.rm = TRUE
      ),

    pct_complete_incl_day0 =
      mean(
        complete_incl_day0_window,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>
  arrange(
    desc(
      communication_days
    )
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 14) BUILD DAILY PANEL FOR DISTRIBUTED-LAG ROBUSTNESS
# ============================================================
#
# Each game-date appears exactly once.
#
# Communication characteristics are nonzero only on actual
# communication dates. Step 06 will create Day 0 through
# Day +3 distributed lags from these variables.
# ============================================================

daily_panel <- steam_calendar |>

  left_join(
    communications_model |>
      select(
        game,
        event_date,
        event_id,
        patch_title,
        z_competitive,
        z_cosmetic,
        z_seasonal,
        z_difficulty,
        z_log_total_chars,
        z_log_avg_sentence_chars,
        season_related_communication
      ),
    by = c(
      "game",
      "calendar_date" = "event_date"
    )
  ) |>

  mutate(

    communication_day =
      as.integer(
        !is.na(event_id)
      ),


    # --------------------------------------------------------
    # Communication emphasis variables
    # --------------------------------------------------------

    x_competitive =
      if_else(
        communication_day == 1,
        z_competitive,
        0
      ),

    x_cosmetic =
      if_else(
        communication_day == 1,
        z_cosmetic,
        0
      ),

    x_seasonal =
      if_else(
        communication_day == 1,
        z_seasonal,
        0
      ),

    x_difficulty =
      if_else(
        communication_day == 1,
        z_difficulty,
        0
      ),


    # --------------------------------------------------------
    # Communication controls
    # --------------------------------------------------------

    x_length =
      if_else(
        communication_day == 1,
        z_log_total_chars,
        0
      ),

    x_sentence_length =
      if_else(
        communication_day == 1,
        z_log_avg_sentence_chars,
        0
      ),

    x_season_related =
      if_else(
        communication_day == 1,
        as.numeric(
          season_related_communication
        ),
        0
      )
  ) |>

  left_join(
    steam_coverage,
    by = "game"
  ) |>

  mutate(

    # --------------------------------------------------------
    # Daily observed game-age control
    # --------------------------------------------------------

    game_age_days =
      as.numeric(
        calendar_date -
          steam_start_date
      ),

    log_game_age_days =
      log1p(
        game_age_days
      ),

    z_log_game_age_days =
      scale2(
        log_game_age_days
      ),


    # --------------------------------------------------------
    # Daily fixed-effect variables
    # --------------------------------------------------------

    weekday_f =
      factor(
        wday(
          calendar_date,
          label = TRUE,
          abbr = TRUE,
          week_start = 1
        )
      ),

    month_f =
      factor(
        month(
          calendar_date
        )
      ),

    year_f =
      factor(
        year(
          calendar_date
        )
      ),

    game_month =
      interaction(
        game,
        format(
          calendar_date,
          "%Y-%m"
        ),
        drop = TRUE
      )
  ) |>

  arrange(
    game,
    calendar_date
  )


# ============================================================
# 15) DAILY PANEL VALIDATION
# ============================================================

duplicate_daily_rows <- daily_panel |>
  count(
    game,
    calendar_date
  ) |>
  filter(
    n > 1
  )


if (nrow(duplicate_daily_rows) > 0) {

  stop(
    "Duplicate game-day rows found in the daily panel."
  )
}


cat(
  "\n============================================================\n"
)

cat(
  "DAILY PANEL CHECK\n"
)

cat(
  "============================================================\n"
)


daily_panel |>
  summarise(

    calendar_rows =
      n(),

    observed_engagement_days =
      sum(
        !is.na(
          log_avg_players_daily
        )
      ),

    communication_days =
      sum(
        communication_day,
        na.rm = TRUE
      ),

    games =
      n_distinct(
        game
      )
  ) |>
  print(
    width = Inf
  )


# ============================================================
# 16) FINAL DATA VALIDATION
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "FINAL ANALYSIS DATA CHECK\n"
)

cat(
  "============================================================\n"
)


event_data |>
  summarise(

    communication_events =
      n(),

    games =
      n_distinct(
        game
      ),

    missing_competitive =
      sum(
        is.na(
          z_competitive
        )
      ),

    missing_cosmetic =
      sum(
        is.na(
          z_cosmetic
        )
      ),

    missing_seasonal =
      sum(
        is.na(
          z_seasonal
        )
      ),

    missing_difficulty =
      sum(
        is.na(
          z_difficulty
        )
      ),

    missing_length =
      sum(
        is.na(
          z_log_total_chars
        )
      ),

    missing_sentence_length =
      sum(
        is.na(
          z_log_avg_sentence_chars
        )
      ),

    missing_season_related =
      sum(
        is.na(
          season_related_communication
        )
      ),

    missing_game_age =
      sum(
        is.na(
          z_log_game_age_days_event
        )
      ),

    complete_primary_windows =
      sum(
        complete_excl_day0_window,
        na.rm = TRUE
      ),

    complete_day0_windows =
      sum(
        complete_incl_day0_window,
        na.rm = TRUE
      )
  ) |>
  print(
    width = Inf
  )


# ============================================================
# 17) SAVE ONE COMPACT ANALYSIS OBJECT
# ============================================================

analysis_data <- list(

  event_data =
    event_data,

  daily_panel =
    daily_panel,

  steam_coverage =
    steam_coverage
)


output_file <-
  "data/interim/study2/study2_analysis_data.rds"


saveRDS(
  analysis_data,
  output_file,
  compress = "xz"
)


# ============================================================
# 18) FINAL MESSAGE
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "DONE - STEP 04\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nPrimary model:\n"
)

cat(
  " - Pre:  Days -3 to -1\n"
)

cat(
  " - Post: Days +1 to +3\n"
)

cat(
  " - Day 0 excluded\n"
)

cat(
  " - Outcome: engagement_change_excl_day0\n"
)


cat(
  "\nDay 0 sensitivity model:\n"
)

cat(
  " - Pre:  Days -3 to -1\n"
)

cat(
  " - Post: Days 0 to +2\n"
)

cat(
  " - Outcome: engagement_change_incl_day0\n"
)


cat(
  "\nDaily robustness model:\n"
)

cat(
  " - One row per actual game-date\n"
)

cat(
  " - Day 0 through Day +3 lags created in Step 06\n"
)

cat(
  " - Nearby communications can enter simultaneously\n"
)


cat(
  "\nSaved:\n"
)

cat(
  " -",
  output_file,
  "\n"
)