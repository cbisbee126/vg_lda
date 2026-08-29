# ============================================================
# 02 BUILD PROGRESSION EMPHASIS MEASURES
# ============================================================
#
# PURPOSE
# -------
# Measure four progression-emphasis dimensions in qualifying
# official Steam update communications:
#   1) Competitive progression
#   2) Cosmetics / identity
#   3) Seasonal progression
#   4) Difficulty / balance
#
# UNIT OF ANALYSIS
# ----------------
# One game-day update communication.
#
# MEASURE
# -------
# Text is split into sentences. Sentences <= 20 characters are
# removed. A sentence may match more than one dimension.
#
# Relative emphasis =
#   characters in matched sentences / total retained characters
#
# OUTPUT
# ------
# data/interim/study2/patch_levers_sentence.csv
#
# The sentence-level working object is kept in memory only to
# avoid creating an unnecessary large intermediate file.
# ============================================================

rm(list = ls())

library(tidyverse)
library(stringr)
library(readr)

dir.create("data/interim/study2", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD QUALIFYING COMMUNICATION DAYS
# ============================================================

patches <- read_csv(
  "data/interim/study2/update_communication_days.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),
    event_id = as.character(update_id),
    patch_title = as.character(titles),
    full_text = as.character(combined_text),
    source_type = "official_steam_update_communication",
    full_text = str_squish(full_text),
    patch_title = str_squish(patch_title),
    char_count = nchar(full_text, type = "chars"),
    word_count = str_count(full_text, "\\S+"),
    log_char_count = log1p(char_count)
  ) |>
  filter(
    !is.na(full_text),
    full_text != "",
    !is.na(event_id),
    !is.na(game),
    !is.na(event_date)
  ) |>
  arrange(game, event_date)

cat("\nLoaded update-communication days:", nrow(patches), "\n")
cat("Games:", n_distinct(patches$game), "\n")

if (nrow(patches |> count(game, event_date) |> filter(n > 1)) > 0) {
  stop("Duplicate game-day update communications found in Step 02 input.")
}

# ============================================================
# 2) LOCKED PROGRESSION DICTIONARIES
# ============================================================

competitive_pattern <- paste(
  c(
    "rank", "ranks", "ranked", "competitive", "mmr", "elo", "sr",
    "leaderboard", "leaderboards", "matchmaking", "placement", "placements",
    "division", "divisions", "tier", "tiers", "queue", "queues", "ladder",
    "promotion", "demotion", "rank reset", "ranked reset"
  ),
  collapse = "|"
)

cosmetic_pattern <- paste(
  c(
    "skin", "skins", "cosmetic", "cosmetics", "bundle", "bundles",
    "store", "shop", "item shop", "emote", "emotes", "spray", "sprays",
    "mythic", "legendary", "epic", "highlight intro", "victory pose",
    "weapon charm", "charm", "charms", "souvenir", "player icon",
    "name card", "voice line", "outfit", "outfits", "appearance",
    "customization", "customisation", "rarity", "heirloom", "camo",
    "paint", "paintjob", "decoration"
  ),
  collapse = "|"
)

seasonal_pattern <- paste(
  c(
    "event", "events", "battle pass", "season", "seasons", "seasonal",
    "limited-time", "limited time", "ltm", "challenge", "challenges",
    "festival", "operation", "pass", "reset", "rank reset", "season reset",
    "reward track", "milestone", "milestones", "season launch", "new season"
  ),
  collapse = "|"
)

difficulty_pattern <- paste(
  c(
    "buff", "buffs", "nerf", "nerfs", "balance", "balanced", "balancing",
    "rework", "reworks", "tuning", "cooldown", "cooldowns", "damage",
    "healing", "shield", "armor", "ultimate", "passive", "ability",
    "abilities", "scaling", "weapon balance", "gameplay", "difficulty",
    "boss", "enemy", "combat", "fairness", "challenge", "mechanics",
    "map changes", "hero changes", "adjustment", "adjustments"
  ),
  collapse = "|"
)

# ============================================================
# 3) SENTENCE-LEVEL CODING
# ============================================================

sentences <- patches |>
  mutate(sentence_split = str_split(full_text, "(?<=[.!?])\\s+")) |>
  unnest(sentence_split) |>
  transmute(
    game,
    event_id,
    event_date,
    patch_title,
    sentence_text = str_squish(sentence_split),
    sentence_chars = nchar(str_squish(sentence_split), type = "chars")
  ) |>
  filter(
    !is.na(sentence_text),
    sentence_text != "",
    sentence_chars > 20
  ) |>
  mutate(
    text_all = str_to_lower(sentence_text),
    is_competitive = str_detect(text_all, paste0("\\b(", competitive_pattern, ")\\b")),
    is_cosmetic = str_detect(text_all, paste0("\\b(", cosmetic_pattern, ")\\b")),
    is_seasonal = str_detect(text_all, paste0("\\b(", seasonal_pattern, ")\\b")),
    is_difficulty = str_detect(text_all, paste0("\\b(", difficulty_pattern, ")\\b")),
    sentence_lever_total =
      as.integer(is_competitive) +
      as.integer(is_cosmetic) +
      as.integer(is_seasonal) +
      as.integer(is_difficulty),
    comp_chars_sentence = sentence_chars * as.integer(is_competitive),
    cos_chars_sentence = sentence_chars * as.integer(is_cosmetic),
    seas_chars_sentence = sentence_chars * as.integer(is_seasonal),
    diff_chars_sentence = sentence_chars * as.integer(is_difficulty)
  )

cat("Retained sentences:", nrow(sentences), "\n")

# ============================================================
# 4) AGGREGATE TO ONE COMMUNICATION DAY
# ============================================================

lever_summary <- sentences |>
  group_by(game, event_id, event_date, patch_title) |>
  summarise(
    total_sentence_chars = sum(sentence_chars, na.rm = TRUE),
    total_sentences = n(),
    abs_competitive = sum(comp_chars_sentence, na.rm = TRUE),
    abs_cosmetic = sum(cos_chars_sentence, na.rm = TRUE),
    abs_seasonal = sum(seas_chars_sentence, na.rm = TRUE),
    abs_difficulty = sum(diff_chars_sentence, na.rm = TRUE),
    sentence_comp_hits = sum(is_competitive, na.rm = TRUE),
    sentence_cos_hits = sum(is_cosmetic, na.rm = TRUE),
    sentence_seas_hits = sum(is_seasonal, na.rm = TRUE),
    sentence_diff_hits = sum(is_difficulty, na.rm = TRUE),
    any_lever_sentences = sum(sentence_lever_total >= 1, na.rm = TRUE),
    multi_lever_sentences = sum(sentence_lever_total > 1, na.rm = TRUE),
    .groups = "drop"
  ) |>
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
    avg_sentence_chars = if_else(
      total_sentences > 0,
      total_sentence_chars / total_sentences,
      NA_real_
    ),
    total_lever_chars =
      abs_competitive + abs_cosmetic + abs_seasonal + abs_difficulty,
    rel_any_lever = if_else(
      total_sentence_chars > 0,
      total_lever_chars / total_sentence_chars,
      0
    )
  )

patch_metadata <- patches |>
  select(
    game, appid, event_id, event_date, patch_title,
    source_type, full_text, char_count, word_count,
    log_char_count, n_update_posts, announcement_ids, text_chars
  ) |>
  distinct()

patch_levers_sentence <- patch_metadata |>
  left_join(
    lever_summary,
    by = c("game", "event_id", "event_date", "patch_title")
  ) |>
  mutate(
    across(
      c(
        total_sentence_chars, total_sentences,
        abs_competitive, abs_cosmetic, abs_seasonal, abs_difficulty,
        sentence_comp_hits, sentence_cos_hits, sentence_seas_hits,
        sentence_diff_hits, any_lever_sentences, multi_lever_sentences,
        rel_competitive, rel_cosmetic, rel_seasonal, rel_difficulty,
        total_lever_chars, rel_any_lever
      ),
      ~ replace_na(.x, 0)
    )
  )

# ============================================================
# 5) CONSOLE DIAGNOSTICS
# ============================================================

cat("\n--- MEAN EMPHASIS ---\n")
patch_levers_sentence |>
  summarise(
    competitive = mean(rel_competitive),
    cosmetic = mean(rel_cosmetic),
    seasonal = mean(rel_seasonal),
    difficulty = mean(rel_difficulty),
    any_lever = mean(rel_any_lever),
    zero_lever_days = sum(rel_any_lever == 0),
    pct_zero_lever_days = mean(rel_any_lever == 0)
  ) |>
  print(width = Inf)

cat("\n--- GAME BREAKDOWN ---\n")
patch_levers_sentence |>
  group_by(game) |>
  summarise(
    updates = n(),
    mean_comp = mean(rel_competitive),
    mean_cos = mean(rel_cosmetic),
    mean_seas = mean(rel_seasonal),
    mean_diff = mean(rel_difficulty),
    avg_sentence_chars = mean(avg_sentence_chars, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(desc(updates)) |>
  print(n = Inf, width = Inf)

cat("\n--- SENTENCE OVERLAP ---\n")
sentences |>
  summarise(
    pct_any_lever_sentence = mean(sentence_lever_total >= 1),
    pct_multi_lever_sentence = mean(sentence_lever_total > 1)
  ) |>
  print(width = Inf)

# ============================================================
# 6) SAVE ONE FILE
# ============================================================

output_file <- "data/interim/study2/patch_levers_sentence.csv"
write_csv(patch_levers_sentence, output_file)

cat("\nDONE - Step 02\n")
cat("Update-communication days:", nrow(patch_levers_sentence), "\n")
cat("Saved:", output_file, "\n")
