# ============================================================
# 05 STUDY 1 VS STUDY 2 EMPHASIS COMPARISON
# ============================================================

rm(list = ls())

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

dir.create("output/tables/study2", showWarnings = FALSE, recursive = TRUE)
dir.create("output/figures/study2", showWarnings = FALSE, recursive = TRUE)


# ============================================================
# 1) LOAD STUDY 2 COMMUNICATION DATA
# ============================================================

communications <- read_csv(
  "data/interim/study2/update_progression_emphasis.csv",
  show_col_types = FALSE
)


# ============================================================
# 2) NORMALIZED STUDY 2 EMPHASIS
# ============================================================

study2_overall <- communications |>
  summarise(
    competitive = mean(rel_competitive, na.rm = TRUE),
    cosmetic = mean(rel_cosmetic, na.rm = TRUE),
    seasonal = mean(rel_seasonal, na.rm = TRUE),
    difficulty = mean(rel_difficulty, na.rm = TRUE)
  )

study2_overall_norm <- study2_overall |>
  mutate(
    total = competitive + cosmetic + seasonal + difficulty,
    competitive = competitive / total,
    cosmetic = cosmetic / total,
    seasonal = seasonal / total,
    difficulty = difficulty / total
  ) |>
  select(-total)


# ============================================================
# 3) STUDY 1 NORMALIZED CONSUMER VALUES
# ============================================================

study1_overall_norm <- tibble(
  competitive = 0.455,
  cosmetic = 0.184,
  seasonal = 0.181,
  difficulty = 0.179
)


# ============================================================
# 4) COMPARISON TABLE
# ============================================================

comparison_table <- tibble(
  progression_system = c(
    "Competitive progression",
    "Cosmetics and identity",
    "Seasonal systems",
    "Difficulty and balance"
  ),
  consumer_discourse = c(
    study1_overall_norm$competitive,
    study1_overall_norm$cosmetic,
    study1_overall_norm$seasonal,
    study1_overall_norm$difficulty
  ),
  developer_communication = c(
    study2_overall_norm$competitive,
    study2_overall_norm$cosmetic,
    study2_overall_norm$seasonal,
    study2_overall_norm$difficulty
  )
) |>
  mutate(
    difference =
      developer_communication - consumer_discourse
  )

cat("\n--- STUDY 1 VS STUDY 2 COMPARISON ---\n")
print(comparison_table, n = Inf, width = Inf)


# ============================================================
# 5) PLOT DATA
# ============================================================

plot_data <- comparison_table |>
  select(
    progression_system,
    consumer_discourse,
    developer_communication
  ) |>
  pivot_longer(
    cols = c(
      consumer_discourse,
      developer_communication
    ),
    names_to = "source",
    values_to = "weight"
  ) |>
  mutate(
    source = recode(
      source,
      consumer_discourse = "Consumer discourse",
      developer_communication = "Developer communication"
    ),
    source = factor(
      source,
      levels = c(
        "Consumer discourse",
        "Developer communication"
      )
    ),
    progression_system = factor(
      progression_system,
      levels = c(
        "Competitive progression",
        "Cosmetics and identity",
        "Seasonal systems",
        "Difficulty and balance"
      )
    )
  )


# ============================================================
# 6) FIGURE
# ============================================================

p <- ggplot(
  plot_data,
  aes(
    x = progression_system,
    y = weight,
    fill = source
  )
) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.68,
    color = "black",
    linewidth = 0.35
  ) +
  geom_text(
    aes(label = sprintf("%.2f", weight)),
    position = position_dodge(width = 0.78),
    vjust = -0.35,
    size = 3.5
  ) +
  scale_x_discrete(
    labels = c(
      "Competitive progression" = "Competitive",
      "Cosmetics and identity" = "Cosmetics",
      "Seasonal systems" = "Seasonal",
      "Difficulty and balance" = "Difficulty &\nBalance"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 0.52),
    breaks = seq(0, 0.5, 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_fill_manual(
    values = c(
      "Consumer discourse" = "grey25",
      "Developer communication" = "grey80"
    )
  ) +
  labs(
    x = NULL,
    y = "Share of emphasis",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.justification = "center",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 11, margin = margin(t = 6)),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 11, margin = margin(r = 8)),
    legend.text = element_text(size = 9),
    legend.key.width = grid::unit(0.9, "cm"),
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
  "output/figures/study2/comparison_chart.png",
  p,
  width = 8,
  height = 5,
  dpi = 300
)

cat("\nDONE - STEP 05\n")