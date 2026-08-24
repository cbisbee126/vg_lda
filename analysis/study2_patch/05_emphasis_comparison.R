library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

dir.create("output/tables/study2", showWarnings = FALSE, recursive = TRUE)
dir.create("output/figures/study2", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD STUDY 2 (DEVELOPER) DATA
# Use patch_levers by default
# If you want only the final merged/modeling sample later,
# you can switch this file path.
# ============================================================

patch_levers <- read_csv(
  "data/interim/study2/patch_levers_with_controls.csv",
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
  mutate(
    lever = recode(
      as.character(lever),
      "Competitive\nProgression" = "Competitive",
      "Cosmetics\n& Identity"    = "Cosmetics",
      "Seasonal\nSystems"        = "Seasonal",
      "Difficulty\n& Balance"    = "Difficulty/Balance"
    ),
    lever = factor(
      lever,
      levels = c(
        "Competitive",
        "Cosmetics",
        "Seasonal",
        "Difficulty/Balance"
      )
    )
  ) %>%
  pivot_longer(
    cols = c(consumer, developer),
    names_to = "Group",
    values_to = "Weight"
  ) %>%
  mutate(
    Group = recode(
      Group,
      consumer  = "Player discourse",
      developer = "Developer communication"
    ),
    Group = factor(
      Group,
      levels = c(
        "Player discourse",
        "Developer communication"
      )
    )
  )

# ============================================================
# 6) PLOT
# ============================================================

p <- ggplot(
  df_long,
  aes(x = lever, y = Weight, fill = Group)
) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.68,
    color = "black",
    linewidth = 0.35
  ) +
  geom_text(
    aes(label = sprintf("%.2f", Weight)),
    position = position_dodge(width = 0.78),
    vjust = -0.35,
    size = 3.5
  ) +
  scale_x_discrete(
    labels = c(
      "Competitive" = "Competitive",
      "Cosmetics" = "Cosmetics",
      "Seasonal" = "Seasonal",
      "Difficulty/Balance" = "Difficulty &\nBalance"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 0.52),
    breaks = seq(0, 0.5, 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_fill_manual(
    values = c(
      "Player discourse" = "grey25",
      "Developer communication" = "grey80"
    )
  ) +
  labs(
    x = NULL,
    y = "Normalized emphasis",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.justification = "center",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(
      size = 11,
      margin = margin(t = 6)
    ),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(
      size = 11,
      margin = margin(r = 8)
    ),
    legend.text = element_text(size = 9),
    legend.key.width = unit(0.9, "cm"),
    plot.margin = margin(10, 15, 8, 10)
  )

print(p)
# ============================================================
# 7) SAVE
# ============================================================

write_csv(
  comparison_table,
  "output/tables/study2/comparison_table.csv"
)

ggsave(
  filename = "output/figures/study2/comparison_chart.png",
  plot = p,
  width = 8,
  height = 5,
  dpi = 300
)

cat("\nDONE — Comparison table and chart saved\n")

