# ============================================================
# 01B COMBINE PATCH FILES (FINAL — STANDARDIZED)
# ============================================================

library(tidyverse)
library(stringr)
library(readr)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD DATA
# =============================================================

apex <- read_csv("data_raw/apex_patch_data.csv", show_col_types = FALSE)
marvel <- read_csv("data_raw/marvel_patch_data.csv", show_col_types = FALSE)
overwatch <- read_csv("data_raw/overwatch_patch_data.csv", show_col_types = FALSE)

# ============================================================
# 2) STANDARDIZE APEX
# ============================================================

apex_clean <- apex %>%
  transmute(
    game = "Apex Legends",
    event_date = as.Date(event_date),
    patch_title = title,
    source_url = url,
    full_text = text,
    
    # already exists
    event_id = event_id,
    
    # create word count
    word_count = str_count(full_text, "\\S+")
  )

# ============================================================
# 3) STANDARDIZE MARVEL
# ============================================================

marvel_clean <- marvel %>%
  transmute(
    game = "Marvel Rivals",
    event_date = as.Date(event_date),
    patch_title = patch_title,
    source_url = source_url,
    full_text = content_text,
    
    # create event_id (important for grouping later)
    event_id = str_c(
      "marvel_",
      str_replace_all(str_to_lower(patch_title), "[^a-z0-9]+", "_")
    ),
    
    word_count = word_count
  )

# ============================================================
# 4) STANDARDIZE OVERWATCH
# ============================================================

overwatch_clean <- overwatch %>%
  transmute(
    game = "Overwatch 2",
    event_date = as.Date(event_date),
    patch_title = patch_title,
    source_url = source_url,
    full_text = content_text,
    
    # create event_id
    event_id = str_c(
      "overwatch_",
      str_replace_all(str_to_lower(patch_title), "[^a-z0-9]+", "_")
    ),
    
    word_count = word_count
  )

# ============================================================
# 5) COMBINE ALL
# ============================================================

combined <- bind_rows(
  apex_clean,
  marvel_clean,
  overwatch_clean
)

# ============================================================
# 6) FINAL CLEAN
# ============================================================

combined <- combined %>%
  filter(
    !is.na(full_text),
    full_text != ""
  ) %>%
  mutate(
    full_text = str_squish(full_text),
    patch_title = str_squish(patch_title)
  ) %>%
  arrange(game, event_date)

# ============================================================
# 7) VALIDATION
# ============================================================

cat("\n--- DATA SUMMARY ---\n")

combined %>%
  group_by(game) %>%
  summarise(
    patches = n(),
    avg_words = mean(word_count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print(n = Inf)

cat("\n--- DATE RANGE ---\n")

combined %>%
  summarise(
    min_date = min(event_date, na.rm = TRUE),
    max_date = max(event_date, na.rm = TRUE)
  ) %>%
  print()

# ============================================================
# 8) SAVE
# ============================================================

write_csv(combined, "data_raw/combined_patch_raw.csv")

cat("\n✅ DONE: combined_patch_raw.csv created\n")
cat("🎯 Total patches:", nrow(combined), "\n")