# ============================================================
# 06 STUDY 2 MODEL TESTING
# ============================================================

rm(list = ls())

library(dplyr)
library(readr)
library(lubridate)
library(fixest)
library(purrr)
library(tibble)

dir.create(
  "output/tables/study2",
  showWarnings = FALSE,
  recursive = TRUE
)


# ============================================================
# 1) LOAD EVENT DATA
# ============================================================

event_data <- read_csv(
  "data/interim/study2/study2_event_data.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),

    # Nearby communications
    previous_7d = as.integer(
      coalesce(days_since_previous <= 7, FALSE)
    ),

    next_7d = as.integer(
      coalesce(days_until_next <= 7, FALSE)
    ),

    previous_14d = as.integer(
      coalesce(days_since_previous <= 14, FALSE)
    ),

    next_14d = as.integer(
      coalesce(days_until_next <= 14, FALSE)
    ),

    # Time fixed effects
    month_f = factor(month(event_date)),
    year_f = factor(year(event_date)),

    # Cluster variable
    game_month = interaction(
      game,
      format(event_date, "%Y-%m"),
      drop = TRUE
    )
  )


cat("\nEvent rows:", nrow(event_data), "\n")
cat("Games:", n_distinct(event_data$game), "\n")


# ============================================================
# 2) STANDARDIZE CONTINUOUS PREDICTORS
# ============================================================

event_data <- event_data |>
  mutate(
    z_competitive =
      as.numeric(scale(rel_competitive)),

    z_cosmetic =
      as.numeric(scale(rel_cosmetic)),

    z_seasonal =
      as.numeric(scale(rel_seasonal)),

    z_difficulty =
      as.numeric(scale(rel_difficulty)),

    z_log_char_count =
      as.numeric(scale(log_char_count)),

    z_log_game_age =
      as.numeric(scale(log_game_age_days))
  )


common_predictors <- c(
  "z_competitive",
  "z_cosmetic",
  "z_seasonal",
  "z_difficulty",
  "z_log_char_count",
  "season_title",
  "z_log_game_age"
)


# ============================================================
# 3) CREATE MODEL SAMPLES
# ============================================================

# Primary 7-day sample

primary_df <- event_data |>
  filter(
    complete_7d,
    !is.na(engagement_change_7d),
    if_all(
      all_of(
        c(
          common_predictors,
          "previous_7d",
          "next_7d",
          "game",
          "month_f",
          "year_f",
          "game_month"
        )
      ),
      ~ !is.na(.x)
    )
  )


# 14-day robustness sample

robust_14d_df <- event_data |>
  filter(
    complete_14d,
    !is.na(engagement_change_14d),
    if_all(
      all_of(
        c(
          common_predictors,
          "previous_14d",
          "next_14d",
          "game",
          "month_f",
          "year_f",
          "game_month"
        )
      ),
      ~ !is.na(.x)
    )
  )


# Non-overlapping 7-day sensitivity sample

clean_7d_df <- event_data |>
  filter(
    complete_7d,
    !overlap_7d,
    !is.na(engagement_change_7d),
    if_all(
      all_of(
        c(
          common_predictors,
          "game",
          "month_f",
          "year_f",
          "game_month"
        )
      ),
      ~ !is.na(.x)
    )
  )


# ============================================================
# 4) INSPECT MODEL SAMPLES
# ============================================================

cat("\n--- MODEL SAMPLES ---\n")

sample_summary <- tibble(
  model = c(
    "7-day primary",
    "14-day robustness",
    "7-day clean window"
  ),

  n = c(
    nrow(primary_df),
    nrow(robust_14d_df),
    nrow(clean_7d_df)
  ),

  games = c(
    n_distinct(primary_df$game),
    n_distinct(robust_14d_df$game),
    n_distinct(clean_7d_df$game)
  ),

  clusters = c(
    n_distinct(primary_df$game_month),
    n_distinct(robust_14d_df$game_month),
    n_distinct(clean_7d_df$game_month)
  ),

  mean_change = c(
    mean(primary_df$engagement_change_7d),
    mean(robust_14d_df$engagement_change_14d),
    mean(clean_7d_df$engagement_change_7d)
  )
)

print(
  sample_summary,
  n = Inf,
  width = Inf
)


cat("\n--- PRIMARY SAMPLE BY GAME ---\n")

primary_df |>
  count(
    game,
    name = "updates"
  ) |>
  print(
    n = Inf,
    width = Inf
  )


cat("\n--- CLEAN 7-DAY SAMPLE BY GAME ---\n")

clean_7d_df |>
  count(
    game,
    name = "updates"
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 5) DESCRIPTIVES AND CORRELATIONS
# ============================================================

cat("\n--- PRIMARY SAMPLE DESCRIPTIVES ---\n")

primary_df |>
  summarise(
    mean_change =
      mean(engagement_change_7d),

    sd_change =
      sd(engagement_change_7d),

    mean_competitive =
      mean(rel_competitive),

    mean_cosmetic =
      mean(rel_cosmetic),

    mean_seasonal =
      mean(rel_seasonal),

    mean_difficulty =
      mean(rel_difficulty),

    pct_previous_7d =
      mean(previous_7d),

    pct_next_7d =
      mean(next_7d)
  ) |>
  print(width = Inf)


cat("\n--- PREDICTOR CORRELATIONS ---\n")

cor_vars <- c(
  "z_competitive",
  "z_cosmetic",
  "z_seasonal",
  "z_difficulty",
  "z_log_char_count",
  "season_title",
  "z_log_game_age",
  "previous_7d",
  "next_7d"
)

cor_matrix <- primary_df |>
  select(all_of(cor_vars)) |>
  cor(
    use = "pairwise.complete.obs"
  )

print(
  round(cor_matrix, 3)
)


# ============================================================
# 6) MODEL 1: PRIMARY 7-DAY MODEL
# ============================================================

model_7d <- feols(
  engagement_change_7d ~

    z_competitive +
    z_cosmetic +
    z_seasonal +
    z_difficulty +

    z_log_char_count +
    season_title +
    z_log_game_age +

    previous_7d +
    next_7d

  |

    game +
    month_f +
    year_f,

  data = primary_df,

  vcov = ~ game_month
)


# ============================================================
# 7) MODEL 2: 14-DAY ROBUSTNESS
# ============================================================

model_14d <- feols(
  engagement_change_14d ~

    z_competitive +
    z_cosmetic +
    z_seasonal +
    z_difficulty +

    z_log_char_count +
    season_title +
    z_log_game_age +

    previous_14d +
    next_14d

  |

    game +
    month_f +
    year_f,

  data = robust_14d_df,

  vcov = ~ game_month
)


# ============================================================
# 8) MODEL 3: NON-OVERLAPPING 7-DAY WINDOW
# ============================================================

model_clean_7d <- feols(
  engagement_change_7d ~

    z_competitive +
    z_cosmetic +
    z_seasonal +
    z_difficulty +

    z_log_char_count +
    season_title +
    z_log_game_age

  |

    game +
    month_f +
    year_f,

  data = clean_7d_df,

  vcov = ~ game_month
)


# ============================================================
# 9) PRINT MODELS
# ============================================================

cat("\n--- STUDY 2 MODELS ---\n")

etable(
  model_7d,
  model_14d,
  model_clean_7d,
  digits = 3,
  fitstat = ~ n + r2 + ar2 + wr2
)


# ============================================================
# 10) VIF FUNCTION
# ============================================================

calculate_vif <- function(
    data,
    vars,
    fe_list,
    model_name) {

  X <- data |>
    select(all_of(vars)) |>
    as.matrix()

  X_within <- fixest::demean(
    X,
    f = fe_list
  ) |>
    as.data.frame()

  names(X_within) <- vars

  map_dfr(
    vars,
    function(variable) {

      other_vars <- setdiff(
        vars,
        variable
      )

      fit <- lm(
        reformulate(
          other_vars,
          response = variable
        ),
        data = X_within
      )

      r2 <- summary(fit)$r.squared

      tibble(
        model = model_name,
        variable = variable,
        vif = 1 / (1 - r2)
      )
    }
  )
}


# ============================================================
# 11) VIF CHECKS
# ============================================================

vif_7d <- calculate_vif(
  data = primary_df,

  vars = c(
    common_predictors,
    "previous_7d",
    "next_7d"
  ),

  fe_list = list(
    primary_df$game,
    primary_df$month_f,
    primary_df$year_f
  ),

  model_name = "7-day primary"
)


vif_14d <- calculate_vif(
  data = robust_14d_df,

  vars = c(
    common_predictors,
    "previous_14d",
    "next_14d"
  ),

  fe_list = list(
    robust_14d_df$game,
    robust_14d_df$month_f,
    robust_14d_df$year_f
  ),

  model_name = "14-day robustness"
)


vif_clean <- calculate_vif(
  data = clean_7d_df,

  vars = common_predictors,

  fe_list = list(
    clean_7d_df$game,
    clean_7d_df$month_f,
    clean_7d_df$year_f
  ),

  model_name = "7-day clean window"
)


all_vif <- bind_rows(
  vif_7d,
  vif_14d,
  vif_clean
)


cat("\n--- VIF RESULTS ---\n")

all_vif |>
  arrange(
    model,
    desc(vif)
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 12) EXTRACT MODEL RESULTS
# ============================================================

extract_model <- function(
    model,
    model_name) {

  model_n <- nobs(model)

  coef_table <- as.data.frame(
    coeftable(model)
  )

  ci_table <- as.data.frame(
    confint(model)
  )

  tibble(
    model = model_name,
    term = rownames(coef_table),
    estimate = coef_table[[1]],
    std_error = coef_table[[2]],
    p_value = coef_table[[4]],
    conf_low = ci_table[[1]],
    conf_high = ci_table[[2]],
    n = model_n
  )
}


model_results <- bind_rows(

  extract_model(
    model_7d,
    "7-day primary"
  ),

  extract_model(
    model_14d,
    "14-day robustness"
  ),

  extract_model(
    model_clean_7d,
    "7-day clean window"
  )
)


# ============================================================
# 13) PRINT FOCAL RESULTS
# ============================================================

cat("\n--- PROGRESSION EMPHASIS RESULTS ---\n")

model_results |>
  filter(
    term %in%
      c(
        "z_competitive",
        "z_cosmetic",
        "z_seasonal",
        "z_difficulty"
      )
  ) |>
  select(
    model,
    term,
    estimate,
    std_error,
    p_value,
    conf_low,
    conf_high,
    n
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 14) SAVE
# ============================================================

write_csv(
  model_results,
  "output/tables/study2/study2_model_results.csv"
)

write_csv(
  all_vif,
  "output/tables/study2/study2_vif_results.csv"
)

write_csv(
  sample_summary,
  "output/tables/study2/study2_model_samples.csv"
)


cat("\nDONE - STEP 06\n")