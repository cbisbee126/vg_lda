library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD STUDY 2 (DEVELOPER) DATA
# Use patch_levers by default
# If you want only the final merged/modeling sample later,
# you can switch this file path.
# ============================================================

patch_levers <- read_csv(
  "data_processed/patch_levers.csv",
  show_col_types = FALSE
)

# ============================================================
# 2) COMPUTE DEVELOPER EMPHASIS
# ============================================================

study2_overall <- patch_levers %>%
  summarise(
    competitive = mean(rel_competitive, na.rm = TRUE),
    cosmetic    = mean(rel_cosmetic, na.rm = TRUE),
    seasonal    = mean(rel_seasonal, na.rm = TRUE),
    difficulty  = mean(rel_difficulty, na.rm = TRUE)
  )

study2_overall_norm <- study2_overall %>%
  mutate(
    total = competitive + cosmetic + seasonal + difficulty,
    competitive = competitive / total,
    cosmetic    = cosmetic / total,
    seasonal    = seasonal / total,
    difficulty  = difficulty / total
  ) %>%
  select(-total)

# ============================================================
# 3) STUDY 1 (CONSUMER) VALUES
# Replace only if these numbers change
# ============================================================

study1_overall_norm <- tibble(
  competitive = 0.455,
  cosmetic    = 0.184,
  seasonal    = 0.181,
  difficulty  = 0.179
)

# ============================================================
# 4) BUILD COMPARISON TABLE
# ============================================================

comparison_table <- tibble(
  lever = c(
    "Competitive\nProgression",
    "Cosmetics\n& Identity",
    "Seasonal\nSystems",
    "Difficulty\n& Balance"
  ),
  consumer = c(
    study1_overall_norm$competitive,
    study1_overall_norm$cosmetic,
    study1_overall_norm$seasonal,
    study1_overall_norm$difficulty
  ),
  developer = c(
    study2_overall_norm$competitive,
    study2_overall_norm$cosmetic,
    study2_overall_norm$seasonal,
    study2_overall_norm$difficulty
  )
) %>%
  mutate(
    gap_dev_minus_consumer = developer - consumer,
    abs_gap = abs(gap_dev_minus_consumer),
    lever = factor(
      lever,
      levels = c(
        "Competitive\nProgression",
        "Cosmetics\n& Identity",
        "Seasonal\nSystems",
        "Difficulty\n& Balance"
      )
    )
  )

cat("\n--- COMPARISON TABLE ---\n")
print(comparison_table)

# ============================================================
# 5) LONG FORMAT FOR PLOT
# ============================================================

df_long <- comparison_table %>%
  select(lever, consumer, developer) %>%
  pivot_longer(
    cols = c(consumer, developer),
    names_to = "Group",
    values_to = "Weight"
  ) %>%
  mutate(
    Group = recode(
      Group,
      consumer = "Consumer Emphasis",
      developer = "Developer Emphasis"
    )
  )

# ============================================================
# 6) PLOT
# ============================================================

p <- ggplot(df_long, aes(x = lever, y = Weight, fill = Group)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = round(Weight, 2)),
    position = position_dodge(width = 0.9),
    vjust = -0.25,
    size = 3
  ) +
  labs(
    title = "Consumer vs Developer Emphasis on F2P Levers",
    x = "F2P Design Lever",
    y = "Normalized Weight",
    fill = NULL
  ) +
  theme_minimal() +
  scale_fill_manual(
    values = c(
      "Consumer Emphasis" = "steelblue",
      "Developer Emphasis" = "orange"
    )
  )

print(p)

# ============================================================
# 7) SAVE
# ============================================================

write_csv(comparison_table, "results/comparison_table.csv")

ggsave(
  filename = "results/comparison_chart.png",
  plot = p,
  width = 9,
  height = 6,
  dpi = 300
)

cat("\n✅ DONE — Comparison table and chart saved\n")