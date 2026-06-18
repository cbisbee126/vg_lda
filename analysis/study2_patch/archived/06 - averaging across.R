library(readr)
library(dplyr)

patch_levers <- read_csv("data_processed/patch_levers.csv", show_col_types = FALSE)

# Overall Study 2 designer propensity
study2_overall <- patch_levers %>%
  summarise(
    competitive = mean(rel_competitive, na.rm = TRUE),
    cosmetic    = mean(rel_cosmetic, na.rm = TRUE),
    seasonal    = mean(rel_seasonal, na.rm = TRUE),
    difficulty  = mean(rel_difficulty, na.rm = TRUE)
  )

print(study2_overall)

# Game-level Study 2 designer propensity
study2_by_game <- patch_levers %>%
  group_by(game) %>%
  summarise(
    competitive = mean(rel_competitive, na.rm = TRUE),
    cosmetic    = mean(rel_cosmetic, na.rm = TRUE),
    seasonal    = mean(rel_seasonal, na.rm = TRUE),
    difficulty  = mean(rel_difficulty, na.rm = TRUE),
    .groups = "drop"
  )

print(study2_by_game)