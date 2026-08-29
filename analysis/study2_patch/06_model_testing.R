# ============================================================
# 06 STUDY 2 MODEL TESTING
# ============================================================
#
# PURPOSE
# -------
# Estimate the final Study 2 engagement models.
#
#
# MODEL 1: PRIMARY PRE/POST MODEL
# --------------------------------
#
# Pre  = average logged Steam engagement, Days -3 to -1
# Post = average logged Steam engagement, Days +1 to +3
#
# DV = Post - Pre
#
# Day 0 is excluded because communications can occur at
# different times within the calendar day.
#
#
# MODEL 2: DAY 0 SENSITIVITY MODEL
# ---------------------------------
#
# Pre  = average logged Steam engagement, Days -3 to -1
# Post = average logged Steam engagement, Days 0 to +2
#
# DV = Post - Pre
#
# This keeps the pre- and post-periods the same length while
# testing whether the results depend on excluding Day 0.
#
#
# MODEL 3: DAILY DISTRIBUTED-LAG ROBUSTNESS MODEL
# ------------------------------------------------
#
# One row per actual game-date.
#
# Communication occurrence, progression emphasis, and
# communication controls enter simultaneously for:
#
#   Day 0
#   Day +1
#   Day +2
#   Day +3
#
# This allows nearby communications to be represented jointly.
#
#
# FOCAL PROGRESSION-EMPHASIS MEASURES
# ------------------------------------
# Competitive progression
# Cosmetics / identity
# Seasonal progression
# Difficulty / balance
#
#
# COMMUNICATION CONTROLS
# ----------------------
# Communication length
# Average sentence length
# Season-related communication
#
#
# ADDITIONAL CONTROL
# ------------------
# Logged observed game age
#
#
# FIXED EFFECTS
# -------------
# Game
# Weekday
# Calendar month
# Calendar year
#
#
# STANDARD ERRORS
# ---------------
# Clustered by game-month.
#
#
# OUTPUTS
# -------
# output/tables/study2/study2_model_results.csv
# output/tables/study2/study2_vif_results.csv
# output/figures/study2/study2_daily_lag_emphasis.png
#
# ============================================================


rm(list = ls())


# ============================================================
# 0) PACKAGES
# ============================================================

library(tidyverse)
library(fixest)
library(stringr)


# ============================================================
# 1) OUTPUT DIRECTORIES
# ============================================================

dir.create(
  "output/tables/study2",
  showWarnings = FALSE,
  recursive = TRUE
)

dir.create(
  "output/figures/study2",
  showWarnings = FALSE,
  recursive = TRUE
)


# ============================================================
# 2) HELPERS
# ============================================================


# ------------------------------------------------------------
# Extract coefficients, standard errors, p-values, and CIs
# ------------------------------------------------------------

extract_fixest <- function(
  model,
  model_name
) {

  model_n <- nobs(model)

  coef_matrix <- as.data.frame(
    coeftable(model)
  )

  out <- tibble(
    term = rownames(coef_matrix),
    estimate = coef_matrix[[1]],
    std_error = coef_matrix[[2]],
    statistic = coef_matrix[[3]],
    p_value = coef_matrix[[4]]
  )

  ci_matrix <- as.data.frame(
    confint(model)
  )

  ci <- tibble(
    term = rownames(ci_matrix),
    conf_low = ci_matrix[[1]],
    conf_high = ci_matrix[[2]]
  )

  out |>
    left_join(
      ci,
      by = "term"
    ) |>
    mutate(
      model = model_name,
      n = model_n,
      display = sprintf(
        "%.3f (%.3f)",
        estimate,
        std_error
      ),
      .before = 1
    )
}


# ------------------------------------------------------------
# Calculate VIF after removing the fixed effects
# ------------------------------------------------------------

calculate_vif_table <- function(
  data,
  vars,
  fe_list,
  model_name
) {

  X <- data |>
    select(
      all_of(vars)
    ) |>
    as.matrix()


  X_within <- fixest::demean(
    X = X,
    f = fe_list
  )


  colnames(X_within) <- vars

  vif_data <- as.data.frame(
    X_within
  )


  map_dfr(
    vars,
    function(variable) {

      other_vars <- setdiff(
        vars,
        variable
      )


      variable_variance <- var(
        vif_data[[variable]],
        na.rm = TRUE
      )


      if (
        length(other_vars) == 0 ||
        is.na(variable_variance) ||
        variable_variance == 0
      ) {

        return(
          tibble(
            variable = variable,
            auxiliary_r2 = NA_real_,
            tolerance = NA_real_,
            vif = NA_real_
          )
        )
      }


      aux_model <- lm(
        reformulate(
          other_vars,
          response = variable
        ),
        data = vif_data
      )


      r2 <- summary(
        aux_model
      )$r.squared


      vif_value <- 1 / (1 - r2)


      tibble(
        variable = variable,
        auxiliary_r2 = r2,
        tolerance = 1 / vif_value,
        vif = vif_value
      )
    }
  ) |>
    mutate(
      model = model_name,
      .before = 1
    ) |>
    arrange(
      desc(vif)
    )
}


# ============================================================
# 3) LOAD ANALYSIS DATA
# ============================================================

analysis_data <- readRDS(
  "data/interim/study2/study2_analysis_data.rds"
)


event_data <- analysis_data$event_data

daily_panel <- analysis_data$daily_panel


cat(
  "\n============================================================\n"
)

cat(
  "06 STUDY 2 MODEL TESTING\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nEvent rows:",
  nrow(event_data),
  "\n"
)


cat(
  "Daily calendar rows:",
  nrow(daily_panel),
  "\n"
)


# ============================================================
# 4) COMMON EVENT-LEVEL PREDICTORS
# ============================================================

event_predictors <- c(

  "z_competitive",

  "z_cosmetic",

  "z_seasonal",

  "z_difficulty",

  "z_log_total_chars",

  "z_log_avg_sentence_chars",

  "season_related_communication",

  "z_log_game_age_days_event"
)


event_design_vars <- c(

  event_predictors,

  "game",

  "weekday_f",

  "month_f",

  "year_f",

  "game_month"
)


# ============================================================
# 5) MODEL 1 SAMPLE:
#    PRIMARY PRE/POST, DAY 0 EXCLUDED
# ============================================================

primary_df <- event_data |>
  filter(
    complete_excl_day0_window,
    !is.na(
      engagement_change_excl_day0
    ),
    if_all(
      all_of(event_design_vars),
      ~ !is.na(.x)
    )
  )


cat(
  "\n============================================================\n"
)

cat(
  "MODEL 1 SAMPLE - DAY 0 EXCLUDED\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nComplete communication events:",
  nrow(primary_df),
  "\n"
)


cat(
  "Games:",
  n_distinct(primary_df$game),
  "\n"
)


cat(
  "Mean engagement change:",
  round(
    mean(
      primary_df$engagement_change_excl_day0
    ),
    4
  ),
  "\n"
)


# ============================================================
# 6) MODEL 1A:
#    OVERALL PRE/POST CHANGE, DAY 0 EXCLUDED
# ============================================================

model_mean_excl_day0 <- feols(

  engagement_change_excl_day0 ~ 1,

  data =
    primary_df,

  vcov =
    ~ game_month
)


cat(
  "\n============================================================\n"
)

cat(
  "OVERALL PRE/POST CHANGE - DAY 0 EXCLUDED\n"
)

cat(
  "============================================================\n"
)


print(
  summary(
    model_mean_excl_day0
  )
)


# ============================================================
# 7) MODEL 1B:
#    PRIMARY CONTENT MODEL, DAY 0 EXCLUDED
# ============================================================

model_primary <- feols(

  engagement_change_excl_day0 ~

    z_competitive +

    z_cosmetic +

    z_seasonal +

    z_difficulty +

    z_log_total_chars +

    z_log_avg_sentence_chars +

    season_related_communication +

    z_log_game_age_days_event

  |

    game +

    weekday_f +

    month_f +

    year_f,

  data =
    primary_df,

  vcov =
    ~ game_month
)


cat(
  "\n============================================================\n"
)

cat(
  "PRIMARY CONTENT MODEL - DAY 0 EXCLUDED\n"
)

cat(
  "============================================================\n"
)


print(
  summary(
    model_primary
  )
)


etable(

  model_primary,

  digits = 3,

  fitstat =
    ~ n + r2 + ar2 + wr2
)


# ============================================================
# 8) MODEL 1 VIF
# ============================================================

primary_vif <- calculate_vif_table(

  data =
    primary_df,

  vars =
    event_predictors,

  fe_list =
    list(
      primary_df$game,
      primary_df$weekday_f,
      primary_df$month_f,
      primary_df$year_f
    ),

  model_name =
    "Primary pre/post - Day 0 excluded"
)


cat(
  "\n--- PRIMARY MODEL VIF ---\n"
)


print(
  primary_vif,
  n = Inf,
  width = Inf
)


# ============================================================
# 9) MODEL 2 SAMPLE:
#    DAY 0 INCLUDED
# ============================================================

sensitivity_df <- event_data |>
  filter(
    complete_incl_day0_window,
    !is.na(
      engagement_change_incl_day0
    ),
    if_all(
      all_of(event_design_vars),
      ~ !is.na(.x)
    )
  )


cat(
  "\n============================================================\n"
)

cat(
  "MODEL 2 SAMPLE - DAY 0 INCLUDED\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nComplete communication events:",
  nrow(sensitivity_df),
  "\n"
)


cat(
  "Games:",
  n_distinct(sensitivity_df$game),
  "\n"
)


cat(
  "Mean engagement change:",
  round(
    mean(
      sensitivity_df$engagement_change_incl_day0
    ),
    4
  ),
  "\n"
)


# ============================================================
# 10) MODEL 2A:
#     OVERALL PRE/POST CHANGE, DAY 0 INCLUDED
# ============================================================

model_mean_incl_day0 <- feols(

  engagement_change_incl_day0 ~ 1,

  data =
    sensitivity_df,

  vcov =
    ~ game_month
)


cat(
  "\n============================================================\n"
)

cat(
  "OVERALL PRE/POST CHANGE - DAY 0 INCLUDED\n"
)

cat(
  "============================================================\n"
)


print(
  summary(
    model_mean_incl_day0
  )
)


# ============================================================
# 11) MODEL 2B:
#     CONTENT MODEL, DAY 0 INCLUDED
# ============================================================

model_sensitivity <- feols(

  engagement_change_incl_day0 ~

    z_competitive +

    z_cosmetic +

    z_seasonal +

    z_difficulty +

    z_log_total_chars +

    z_log_avg_sentence_chars +

    season_related_communication +

    z_log_game_age_days_event

  |

    game +

    weekday_f +

    month_f +

    year_f,

  data =
    sensitivity_df,

  vcov =
    ~ game_month
)


cat(
  "\n============================================================\n"
)

cat(
  "SENSITIVITY CONTENT MODEL - DAY 0 INCLUDED\n"
)

cat(
  "============================================================\n"
)


print(
  summary(
    model_sensitivity
  )
)


etable(

  model_sensitivity,

  digits = 3,

  fitstat =
    ~ n + r2 + ar2 + wr2
)


# ============================================================
# 12) MODEL 2 VIF
# ============================================================

sensitivity_vif <- calculate_vif_table(

  data =
    sensitivity_df,

  vars =
    event_predictors,

  fe_list =
    list(
      sensitivity_df$game,
      sensitivity_df$weekday_f,
      sensitivity_df$month_f,
      sensitivity_df$year_f
    ),

  model_name =
    "Sensitivity pre/post - Day 0 included"
)


cat(
  "\n--- DAY 0 SENSITIVITY MODEL VIF ---\n"
)


print(
  sensitivity_vif,
  n = Inf,
  width = Inf
)


# ============================================================
# 13) COMPARE PRIMARY AND DAY 0 SAMPLES
# ============================================================

same_event_sample <-

  nrow(primary_df) ==
    nrow(sensitivity_df) &&

  setequal(
    primary_df$event_id,
    sensitivity_df$event_id
  )


cat(
  "\n============================================================\n"
)

cat(
  "PRE/POST SAMPLE COMPARISON\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nPrimary N:",
  nrow(primary_df),
  "\n"
)


cat(
  "Day 0 sensitivity N:",
  nrow(sensitivity_df),
  "\n"
)


cat(
  "Same event sample:",
  same_event_sample,
  "\n"
)


# ============================================================
# 14) CREATE DAY 0 THROUGH DAY +3 DISTRIBUTED LAGS
# ============================================================
#
# For engagement observed on calendar date t:
#
# *_t0 = communication occurring today
# *_t1 = communication occurring one day ago
# *_t2 = communication occurring two days ago
# *_t3 = communication occurring three days ago
#
# All four horizons enter the model simultaneously.
# ============================================================

lag_df <- daily_panel |>

  arrange(
    game,
    calendar_date
  ) |>

  group_by(
    game
  ) |>

  mutate(


    # --------------------------------------------------------
    # Communication occurrence
    # --------------------------------------------------------

    comm_t0 =
      communication_day,

    comm_t1 =
      lag(
        communication_day,
        1,
        default = 0
      ),

    comm_t2 =
      lag(
        communication_day,
        2,
        default = 0
      ),

    comm_t3 =
      lag(
        communication_day,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Competitive progression
    # --------------------------------------------------------

    comp_t0 =
      x_competitive,

    comp_t1 =
      lag(
        x_competitive,
        1,
        default = 0
      ),

    comp_t2 =
      lag(
        x_competitive,
        2,
        default = 0
      ),

    comp_t3 =
      lag(
        x_competitive,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Cosmetics / identity
    # --------------------------------------------------------

    cos_t0 =
      x_cosmetic,

    cos_t1 =
      lag(
        x_cosmetic,
        1,
        default = 0
      ),

    cos_t2 =
      lag(
        x_cosmetic,
        2,
        default = 0
      ),

    cos_t3 =
      lag(
        x_cosmetic,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Seasonal progression
    # --------------------------------------------------------

    seas_t0 =
      x_seasonal,

    seas_t1 =
      lag(
        x_seasonal,
        1,
        default = 0
      ),

    seas_t2 =
      lag(
        x_seasonal,
        2,
        default = 0
      ),

    seas_t3 =
      lag(
        x_seasonal,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Difficulty / balance
    # --------------------------------------------------------

    diff_t0 =
      x_difficulty,

    diff_t1 =
      lag(
        x_difficulty,
        1,
        default = 0
      ),

    diff_t2 =
      lag(
        x_difficulty,
        2,
        default = 0
      ),

    diff_t3 =
      lag(
        x_difficulty,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Communication length
    # --------------------------------------------------------

    length_t0 =
      x_length,

    length_t1 =
      lag(
        x_length,
        1,
        default = 0
      ),

    length_t2 =
      lag(
        x_length,
        2,
        default = 0
      ),

    length_t3 =
      lag(
        x_length,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Average sentence length
    # --------------------------------------------------------

    sent_length_t0 =
      x_sentence_length,

    sent_length_t1 =
      lag(
        x_sentence_length,
        1,
        default = 0
      ),

    sent_length_t2 =
      lag(
        x_sentence_length,
        2,
        default = 0
      ),

    sent_length_t3 =
      lag(
        x_sentence_length,
        3,
        default = 0
      ),


    # --------------------------------------------------------
    # Season-related communication
    # --------------------------------------------------------

    season_related_t0 =
      x_season_related,

    season_related_t1 =
      lag(
        x_season_related,
        1,
        default = 0
      ),

    season_related_t2 =
      lag(
        x_season_related,
        2,
        default = 0
      ),

    season_related_t3 =
      lag(
        x_season_related,
        3,
        default = 0
      )
  ) |>

  ungroup() |>

  filter(

    # Full three-day communication history must be available.

    game_age_days >= 3,

    !is.na(
      log_avg_players_daily
    ),

    !is.na(
      z_log_game_age_days
    )
  )


# ============================================================
# 15) DAILY ROBUSTNESS SAMPLE CHECK
# ============================================================

duplicate_model_days <- lag_df |>
  count(
    game,
    calendar_date
  ) |>
  filter(
    n > 1
  )


if (nrow(duplicate_model_days) > 0) {

  stop(
    "Duplicate game-day rows found in the robustness model sample."
  )
}


cat(
  "\n============================================================\n"
)

cat(
  "MODEL 3 SAMPLE - DAILY DISTRIBUTED LAGS\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nDaily observations:",
  nrow(lag_df),
  "\n"
)


cat(
  "Unique game-dates:",
  n_distinct(
    paste(
      lag_df$game,
      lag_df$calendar_date
    )
  ),
  "\n"
)


cat(
  "Games:",
  n_distinct(
    lag_df$game
  ),
  "\n"
)


cat(
  "Communication days:",
  sum(
    lag_df$communication_day,
    na.rm = TRUE
  ),
  "\n"
)


# ============================================================
# 16) MODEL 3:
#     DAILY DISTRIBUTED-LAG ROBUSTNESS
# ============================================================

model_lag <- feols(

  log_avg_players_daily ~


    # Communication occurrence

    comm_t0 +
    comm_t1 +
    comm_t2 +
    comm_t3 +


    # Competitive progression

    comp_t0 +
    comp_t1 +
    comp_t2 +
    comp_t3 +


    # Cosmetics / identity

    cos_t0 +
    cos_t1 +
    cos_t2 +
    cos_t3 +


    # Seasonal progression

    seas_t0 +
    seas_t1 +
    seas_t2 +
    seas_t3 +


    # Difficulty / balance

    diff_t0 +
    diff_t1 +
    diff_t2 +
    diff_t3 +


    # Communication length

    length_t0 +
    length_t1 +
    length_t2 +
    length_t3 +


    # Average sentence length

    sent_length_t0 +
    sent_length_t1 +
    sent_length_t2 +
    sent_length_t3 +


    # Season-related communication

    season_related_t0 +
    season_related_t1 +
    season_related_t2 +
    season_related_t3 +


    # Observed game age

    z_log_game_age_days

  |

    game +

    weekday_f +

    month_f +

    year_f,

  data =
    lag_df,

  vcov =
    ~ game_month
)


cat(
  "\n============================================================\n"
)

cat(
  "DAILY DISTRIBUTED-LAG ROBUSTNESS MODEL\n"
)

cat(
  "============================================================\n"
)


print(
  summary(
    model_lag
  )
)


etable(

  model_lag,

  digits = 3,

  fitstat =
    ~ n + r2 + ar2 + wr2
)


# ============================================================
# 17) DISTRIBUTED-LAG VIF
# ============================================================

lag_vif_vars <- c(

  # Communication occurrence

  "comm_t0",
  "comm_t1",
  "comm_t2",
  "comm_t3",


  # Competitive progression

  "comp_t0",
  "comp_t1",
  "comp_t2",
  "comp_t3",


  # Cosmetics / identity

  "cos_t0",
  "cos_t1",
  "cos_t2",
  "cos_t3",


  # Seasonal progression

  "seas_t0",
  "seas_t1",
  "seas_t2",
  "seas_t3",


  # Difficulty / balance

  "diff_t0",
  "diff_t1",
  "diff_t2",
  "diff_t3",


  # Communication length

  "length_t0",
  "length_t1",
  "length_t2",
  "length_t3",


  # Average sentence length

  "sent_length_t0",
  "sent_length_t1",
  "sent_length_t2",
  "sent_length_t3",


  # Season-related communication

  "season_related_t0",
  "season_related_t1",
  "season_related_t2",
  "season_related_t3",


  # Game age

  "z_log_game_age_days"
)


lag_vif <- calculate_vif_table(

  data =
    lag_df,

  vars =
    lag_vif_vars,

  fe_list =
    list(
      lag_df$game,
      lag_df$weekday_f,
      lag_df$month_f,
      lag_df$year_f
    ),

  model_name =
    "Distributed-lag robustness"
)


cat(
  "\n--- DISTRIBUTED-LAG VIF ---\n"
)


print(
  lag_vif,
  n = Inf,
  width = Inf
)


# ============================================================
# 18) VIF SUMMARY
# ============================================================

all_vif <- bind_rows(

  primary_vif,

  sensitivity_vif,

  lag_vif
)


cat(
  "\n============================================================\n"
)

cat(
  "VIF SUMMARY\n"
)

cat(
  "============================================================\n"
)


all_vif |>
  group_by(
    model
  ) |>
  summarise(

    predictors =
      n(),

    mean_vif =
      mean(
        vif,
        na.rm = TRUE
      ),

    median_vif =
      median(
        vif,
        na.rm = TRUE
      ),

    max_vif =
      max(
        vif,
        na.rm = TRUE
      ),

    vif_5_or_more =
      sum(
        vif >= 5,
        na.rm = TRUE
      ),

    vif_10_or_more =
      sum(
        vif >= 10,
        na.rm = TRUE
      ),

    .groups = "drop"
  ) |>

  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 19) EXTRACT ALL MODEL RESULTS
# ============================================================

mean_excl_results <- extract_fixest(

  model_mean_excl_day0,

  "Overall pre/post - Day 0 excluded"
) |>

  mutate(

    term =
      if_else(
        term == "(Intercept)",
        "Average engagement change",
        term
      )
  )


primary_results <- extract_fixest(

  model_primary,

  "Primary pre/post - Day 0 excluded"
)


mean_incl_results <- extract_fixest(

  model_mean_incl_day0,

  "Overall pre/post - Day 0 included"
) |>

  mutate(

    term =
      if_else(
        term == "(Intercept)",
        "Average engagement change",
        term
      )
  )


sensitivity_results <- extract_fixest(

  model_sensitivity,

  "Sensitivity pre/post - Day 0 included"
)


lag_results <- extract_fixest(

  model_lag,

  "Distributed-lag robustness"
)


model_results <- bind_rows(

  mean_excl_results,

  primary_results,

  mean_incl_results,

  sensitivity_results,

  lag_results
)


# ============================================================
# 20) PRINT PRE/POST CONTENT COMPARISON
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "PRIMARY VS DAY 0 SENSITIVITY CONTENT RESULTS\n"
)

cat(
  "============================================================\n"
)


bind_rows(

  primary_results |>
    select(
      model,
      term,
      estimate,
      std_error,
      p_value
    ),

  sensitivity_results |>
    select(
      model,
      term,
      estimate,
      std_error,
      p_value
    )
) |>

  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 21) FOCAL DISTRIBUTED-LAG RESULTS
# ============================================================

lag_focal <- lag_results |>

  filter(
    str_detect(
      term,
      "^(comp|cos|seas|diff)_t[0-3]$"
    )
  ) |>

  mutate(

    emphasis =
      case_when(

        str_detect(
          term,
          "^comp_"
        ) ~
          "Competitive",

        str_detect(
          term,
          "^cos_"
        ) ~
          "Cosmetic",

        str_detect(
          term,
          "^seas_"
        ) ~
          "Seasonal",

        str_detect(
          term,
          "^diff_"
        ) ~
          "Difficulty/Balance",

        TRUE ~
          NA_character_
      ),


    event_time =
      as.integer(
        str_match(
          term,
          "_t([0-3])$"
        )[, 2]
      ),


    emphasis =
      factor(
        emphasis,
        levels = c(
          "Competitive",
          "Cosmetic",
          "Seasonal",
          "Difficulty/Balance"
        )
      )
  ) |>

  arrange(
    emphasis,
    event_time
  )


cat(
  "\n============================================================\n"
)

cat(
  "PROGRESSION EMPHASIS - DAY 0 THROUGH DAY +3\n"
)

cat(
  "============================================================\n"
)


print(
  lag_focal |>
    select(
      emphasis,
      event_time,
      estimate,
      std_error,
      p_value,
      conf_low,
      conf_high
    ),
  n = Inf,
  width = Inf
)


# ============================================================
# 22) COMMUNICATION-OCCURRENCE ROBUSTNESS RESULTS
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "COMMUNICATION OCCURRENCE - DAY 0 THROUGH DAY +3\n"
)

cat(
  "============================================================\n"
)


lag_results |>
  filter(
    str_detect(
      term,
      "^comm_t[0-3]$"
    )
  ) |>
  select(
    term,
    estimate,
    std_error,
    p_value,
    conf_low,
    conf_high
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 23) CREATE DISTRIBUTED-LAG FIGURE
# ============================================================

p <- ggplot(

  lag_focal,

  aes(
    x = event_time,
    y = estimate
  )

) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.4
  ) +

  geom_errorbar(
    aes(
      ymin = conf_low,
      ymax = conf_high
    ),
    width = 0.10,
    linewidth = 0.4
  ) +

  geom_line(
    linewidth = 0.7
  ) +

  geom_point(
    size = 2
  ) +

  facet_wrap(
    ~ emphasis,
    ncol = 2
  ) +

  scale_x_continuous(
    breaks = 0:3,
    labels = c(
      "Day 0",
      "Day +1",
      "Day +2",
      "Day +3"
    )
  ) +

  labs(
    x =
      "Days Since Communication",

    y =
      "Association with Logged Daily Engagement",

    caption =
      paste0(
        "Estimates represent a one-SD increase in progression ",
        "emphasis. Error bars are 95% confidence intervals."
      )
  ) +

  theme_bw(
    base_size = 11
  ) +

  theme(
    panel.grid.minor =
      element_blank(),

    panel.grid.major.x =
      element_blank(),

    strip.background =
      element_rect(
        fill = "grey90"
      ),

    strip.text =
      element_text(
        face = "bold"
      ),

    plot.caption =
      element_text(
        hjust = 0
      )
  )


print(p)


# ============================================================
# 24) SAVE ESSENTIAL OUTPUTS
# ============================================================

write_csv(

  model_results,

  "output/tables/study2/study2_model_results.csv"
)


write_csv(

  all_vif,

  "output/tables/study2/study2_vif_results.csv"
)


ggsave(

  "output/figures/study2/study2_daily_lag_emphasis.png",

  p,

  width = 8,

  height = 6,

  dpi = 300
)


# ============================================================
# 25) FINAL SUMMARY
# ============================================================

cat(
  "\n============================================================\n"
)

cat(
  "DONE - STEP 06\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nMODEL 1 - PRIMARY:\n"
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
  " - N:",
  nrow(primary_df),
  "\n"
)


cat(
  "\nMODEL 2 - DAY 0 SENSITIVITY:\n"
)

cat(
  " - Pre:  Days -3 to -1\n"
)

cat(
  " - Post: Days 0 to +2\n"
)

cat(
  " - Day 0 included\n"
)

cat(
  " - Outcome: engagement_change_incl_day0\n"
)

cat(
  " - N:",
  nrow(sensitivity_df),
  "\n"
)


cat(
  "\nMODEL 3 - DAILY ROBUSTNESS:\n"
)

cat(
  " - One row per actual game-day\n"
)

cat(
  " - Communication occurrence/content from Day 0 through Day +3\n"
)

cat(
  " - Nearby communications modeled simultaneously\n"
)

cat(
  " - N:",
  nrow(lag_df),
  "\n"
)


cat(
  "\nALL MODELS:\n"
)

cat(
  " - Game fixed effects\n"
)

cat(
  " - Weekday fixed effects\n"
)

cat(
  " - Calendar month fixed effects\n"
)

cat(
  " - Calendar year fixed effects\n"
)

cat(
  " - Standard errors clustered by game-month\n"
)


cat(
  "\nSAVED:\n"
)

cat(
  " - output/tables/study2/study2_model_results.csv\n"
)

cat(
  " - output/tables/study2/study2_vif_results.csv\n"
)

cat(
  " - output/figures/study2/study2_daily_lag_emphasis.png\n"
)