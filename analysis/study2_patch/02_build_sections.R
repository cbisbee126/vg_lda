# ============================================================
# 02 BUILD PROGRESSION LEVERS — SENTENCE LEVEL ONLY
# Input:
#   data_processed/all_games_strict_updates.csv
#
# Outputs:
#   data_processed/patch_sentences.csv
#   data_processed/patch_levers_sentence.csv
# ============================================================

library(tidyverse)
library(stringr)
library(readr)
library(lubridate)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD DATA
# ============================================================

patches <- read_csv(
  "data_processed/all_games_strict_updates.csv",
  show_col_types = FALSE
) %>%
  mutate(
    event_date  = as.Date(event_date),
    patch_title = as.character(patch_title),
    source_url  = as.character(source_url),
    true_url    = as.character(true_url),
    full_text   = as.character(full_text)
  ) %>%
  filter(
    !is.na(full_text),
    full_text != "",
    !is.na(event_id),
    !is.na(game),
    !is.na(event_date)
  ) %>%
  mutate(
    full_text   = str_squish(full_text),
    patch_title = str_squish(patch_title),
    char_count  = nchar(full_text, type = "chars"),
    word_count  = str_count(full_text, "\\S+")
  ) %>%
  arrange(game, event_date)

cat("\n🎮 Loaded strict update events:", nrow(patches), "\n")
cat("🎮 Unique games:", n_distinct(patches$game), "\n\n")

# ============================================================
# 2) VALIDATE INPUT DATA
# ============================================================

cat("\n--- INPUT DATA SUMMARY ---\n")

patches %>%
  group_by(game) %>%
  summarise(
    updates = n(),
    avg_words = mean(word_count, na.rm = TRUE),
    avg_chars = mean(char_count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(updates)) %>%
  print(n = Inf)

cat("\n--- INPUT DATE RANGE ---\n")

patches %>%
  summarise(
    min_date = min(event_date, na.rm = TRUE),
    max_date = max(event_date, na.rm = TRUE)
  ) %>%
  print()

cat("\n--- EVENT DATE CHECK ---\n")

patches %>%
  group_by(game) %>%
  summarise(
    total_rows = n(),
    unique_event_dates = n_distinct(event_date),
    duplicate_dates = total_rows - unique_event_dates,
    .groups = "drop"
  ) %>%
  print(n = Inf)

# ============================================================
# 3) DEFINE LEVER DICTIONARIES
# ============================================================

competitive_pattern <- paste(
  c(
    "rank", "ranks", "ranked",
    "competitive",
    "mmr", "elo", "sr",
    "leaderboard", "leaderboards",
    "matchmaking",
    "placement", "placements",
    "division", "divisions",
    "tier", "tiers",
    "queue", "queues",
    "ladder",
    "promotion", "demotion",
    "rank reset", "ranked reset"
  ),
  collapse = "|"
)

cosmetic_pattern <- paste(
  c(
    "skin", "skins",
    "cosmetic", "cosmetics",
    "bundle", "bundles",
    "store", "shop", "item shop",
    "emote", "emotes",
    "spray", "sprays",
    "mythic", "legendary", "epic",
    "highlight intro", "victory pose",
    "weapon charm", "charm", "charms",
    "souvenir",
    "player icon",
    "name card",
    "voice line",
    "outfit", "outfits",
    "appearance",
    "customization", "customisation",
    "rarity",
    "heirloom",
    "camo",
    "paint", "paintjob",
    "decoration"
  ),
  collapse = "|"
)

seasonal_pattern <- paste(
  c(
    "event", "events",
    "battle pass",
    "season", "seasons", "seasonal",
    "limited-time", "limited time",
    "ltm",
    "challenge", "challenges",
    "festival",
    "operation",
    "pass",
    "reset",
    "rank reset",
    "season reset",
    "reward track",
    "milestone", "milestones",
    "season launch",
    "new season"
  ),
  collapse = "|"
)

difficulty_pattern <- paste(
  c(
    "buff", "buffs",
    "nerf", "nerfs",
    "balance", "balanced", "balancing",
    "rework", "reworks",
    "tuning",
    "cooldown", "cooldowns",
    "damage",
    "healing",
    "shield",
    "armor",
    "ultimate",
    "passive",
    "ability", "abilities",
    "scaling",
    "weapon balance",
    "gameplay",
    "difficulty",
    "boss",
    "enemy",
    "combat",
    "fairness",
    "challenge",
    "mechanics",
    "map changes",
    "hero changes",
    "adjustment", "adjustments"
  ),
  collapse = "|"
)

# ============================================================
# 4) SENTENCE-LEVEL LEVER CODING
# ============================================================

sentences <- patches %>%
  mutate(
    sentence_split = str_split(
      full_text,
      "(?<=[.!?])\\s+"
    )
  ) %>%
  unnest(sentence_split) %>%
  mutate(
    sentence_text  = str_squish(sentence_split),
    sentence_chars = nchar(sentence_text, type = "chars"),
    text_all       = str_to_lower(sentence_text)
  ) %>%
  filter(
    !is.na(sentence_text),
    sentence_text != "",
    sentence_chars > 20
  ) %>%
  mutate(
    is_competitive = str_detect(
      text_all,
      paste0("\\b(", competitive_pattern, ")\\b")
    ),

    is_cosmetic = str_detect(
      text_all,
      paste0("\\b(", cosmetic_pattern, ")\\b")
    ),

    is_seasonal = str_detect(
      text_all,
      paste0("\\b(", seasonal_pattern, ")\\b")
    ),

    is_difficulty = str_detect(
      text_all,
      paste0("\\b(", difficulty_pattern, ")\\b")
    ),

    sentence_lever_total =
      as.integer(is_competitive) +
      as.integer(is_cosmetic) +
      as.integer(is_seasonal) +
      as.integer(is_difficulty),

    comp_chars_sentence = sentence_chars * as.integer(is_competitive),
    cos_chars_sentence  = sentence_chars * as.integer(is_cosmetic),
    seas_chars_sentence = sentence_chars * as.integer(is_seasonal),
    diff_chars_sentence = sentence_chars * as.integer(is_difficulty)
  )

cat("\n🧩 Sentences retained after filtering:", nrow(sentences), "\n\n")

# ============================================================
# 5) AGGREGATE SENTENCE-LEVEL MEASURES TO EVENT LEVEL
# ============================================================

patch_levers_sentence <- sentences %>%
  group_by(
    game,
    event_id,
    event_date,
    patch_title,
    source_url
  ) %>%
  summarise(
    total_sentence_chars = sum(sentence_chars, na.rm = TRUE),
    total_sentences      = n(),

    abs_competitive = sum(comp_chars_sentence, na.rm = TRUE),
    abs_cosmetic    = sum(cos_chars_sentence, na.rm = TRUE),
    abs_seasonal    = sum(seas_chars_sentence, na.rm = TRUE),
    abs_difficulty  = sum(diff_chars_sentence, na.rm = TRUE),

    sentence_comp_hits = sum(is_competitive, na.rm = TRUE),
    sentence_cos_hits  = sum(is_cosmetic, na.rm = TRUE),
    sentence_seas_hits = sum(is_seasonal, na.rm = TRUE),
    sentence_diff_hits = sum(is_difficulty, na.rm = TRUE),

    any_lever_sentences   = sum(sentence_lever_total >= 1, na.rm = TRUE),
    multi_lever_sentences = sum(sentence_lever_total > 1, na.rm = TRUE),

    .groups = "drop"
  ) %>%
  mutate(
    rel_competitive = if_else(
      total_sentence_chars > 0,
      abs_competitive / total_sentence_chars,
      0
    ),

    rel_cosmetic = if_else(
      total_sentence_chars > 0,
      abs_cosmetic / total_sentence_chars,
      0
    ),

    rel_seasonal = if_else(
      total_sentence_chars > 0,
      abs_seasonal / total_sentence_chars,
      0
    ),

    rel_difficulty = if_else(
      total_sentence_chars > 0,
      abs_difficulty / total_sentence_chars,
      0
    ),

    total_lever_chars =
      abs_competitive +
      abs_cosmetic +
      abs_seasonal +
      abs_difficulty,

    rel_any_lever = if_else(
      total_sentence_chars > 0,
      total_lever_chars / total_sentence_chars,
      0
    )
  )

# ============================================================
# 6) ADD PATCH METADATA BACK
# ============================================================

patch_metadata <- patches %>%
  select(
    game,
    event_id,
    event_date,
    patch_title,
    source_url,
    true_url,
    full_text,
    char_count,
    word_count,
    log_char_count,
    source_type,
    any_of(c(
      "year",
      "month",
      "announcement_number",
      "days_since_last_event",
      "log_days_since_last_event",
      "broad_update",
      "strict_update",
      "announcement_category"
    ))
  ) %>%
  distinct()

patch_levers_sentence <- patch_metadata %>%
  left_join(
    patch_levers_sentence,
    by = c("game", "event_id", "event_date", "patch_title", "source_url")
  ) %>%
  mutate(
    across(
      where(is.numeric),
      ~replace_na(.x, 0)
    )
  )

# ============================================================
# 7) VALIDATION
# ============================================================

cat("\n--- SENTENCE-LEVEL MEAN SHARE CHECK ---\n")

patch_levers_sentence %>%
  summarise(
    mean_comp = mean(rel_competitive, na.rm = TRUE),
    mean_cos  = mean(rel_cosmetic, na.rm = TRUE),
    mean_seas = mean(rel_seasonal, na.rm = TRUE),
    mean_diff = mean(rel_difficulty, na.rm = TRUE),
    mean_any  = mean(rel_any_lever, na.rm = TRUE)
  ) %>%
  print()

cat("\n--- SENTENCE-LEVEL RANGE CHECK ---\n")

patch_levers_sentence %>%
  summarise(
    across(
      starts_with("rel_"),
      list(
        min = ~min(.x, na.rm = TRUE),
        max = ~max(.x, na.rm = TRUE),
        mean = ~mean(.x, na.rm = TRUE)
      )
    )
  ) %>%
  print()

cat("\n--- GAME BREAKDOWN ---\n")

game_check <- patch_levers_sentence %>%
  group_by(game) %>%
  summarise(
    updates = n(),
    mean_comp = mean(rel_competitive, na.rm = TRUE),
    mean_cos  = mean(rel_cosmetic, na.rm = TRUE),
    mean_seas = mean(rel_seasonal, na.rm = TRUE),
    mean_diff = mean(rel_difficulty, na.rm = TRUE),
    mean_any  = mean(rel_any_lever, na.rm = TRUE),
    avg_sentences = mean(total_sentences, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(updates))

print(game_check, n = Inf)

cat("\n--- SENTENCE OVERLAP CHECK ---\n")

overlap_check <- sentences %>%
  summarise(
    pct_comp_sentence = mean(is_competitive, na.rm = TRUE),
    pct_cos_sentence  = mean(is_cosmetic, na.rm = TRUE),
    pct_seas_sentence = mean(is_seasonal, na.rm = TRUE),
    pct_diff_sentence = mean(is_difficulty, na.rm = TRUE),
    pct_any_lever_sentence = mean(sentence_lever_total >= 1, na.rm = TRUE),
    pct_multi_lever_sentence = mean(sentence_lever_total > 1, na.rm = TRUE)
  )

print(overlap_check)

cat("\n--- ZERO LEVER UPDATE CHECK ---\n")

patch_levers_sentence %>%
  summarise(
    total_updates = n(),
    zero_lever_updates = sum(rel_any_lever == 0, na.rm = TRUE),
    pct_zero_lever_updates = mean(rel_any_lever == 0, na.rm = TRUE)
  ) %>%
  print()

# ============================================================
# 8) SAVE
# ============================================================

write_csv(sentences, "data_processed/patch_sentences.csv")

write_csv(
  patch_levers_sentence,
  "data_processed/patch_levers_sentence.csv"
)

write_csv(
  game_check,
  "results/step2_sentence_game_check.csv"
)

write_csv(
  overlap_check,
  "results/step2_sentence_overlap_check.csv"
)

cat("\n✅ DONE — Sentence-level progression lever dataset created\n")
cat("🎯 Strict updates analyzed:", nrow(patches), "\n")
cat("🧩 Sentences coded:", nrow(sentences), "\n")

cat("\n📁 Main output:\n")
cat("   - data_processed/patch_levers_sentence.csv\n")

cat("\n📁 Supporting output:\n")
cat("   - data_processed/patch_sentences.csv\n")

cat("\n📁 Diagnostics:\n")
cat("   - results/step2_sentence_game_check.csv\n")
cat("   - results/step2_sentence_overlap_check.csv\n")