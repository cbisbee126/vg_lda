library(ggplot2)
library(tidyr)
library(dplyr)

# Create dataset with formatted labels
df <- tibble(
  lever = c(
    "Competitive\nProgression",
    "Cosmetics\n& Identity",
    "Seasonal\nSystems",
    "Difficulty\n& Balance"
  ),
  `Consumer Emphasis` = c(0.449, 0.260, 0.157, 0.134),
  `Developer Emphasis` = c(0.061, 0.101, 0.077, 0.285)
)

# Convert to long format
df_long <- df %>%
  pivot_longer(
    cols = c(`Consumer Emphasis`, `Developer Emphasis`),
    names_to = "Group",
    values_to = "Weight"
  )

# Plot
ggplot(df_long, aes(x = lever, y = Weight, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(
    aes(label = round(Weight, 2)),
    position = position_dodge(width = 0.9),
    vjust = -0.25,
    size = 3
  ) +
  labs(
    title = "Consumer vs Developer Emphasis on F2P Levers",
    x = "F2P Design Lever",
    y = "Average Weight",
    fill = ""
  ) +
  theme_minimal() +
  scale_fill_manual(
    values = c("Consumer Emphasis" = "steelblue",
               "Developer Emphasis" = "orange")
  )