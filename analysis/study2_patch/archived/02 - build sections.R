# ============================================================
# 02 BUILD LEVERS — SENTENCE LEVEL (FINAL — ROBUST)
# ============================================================

library(tidyverse)
library(stringr)
library(readr)
library(lubridate)

dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) LOAD DATA
# ============================================================

patches <- read_csv(
  "data_raw/combined_patch_raw.csv",
  show_col_types = FALSE
)

# ============================================================
# 2) SPLIT INTO SENTENCES (KEY CHANGE)
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
    sentence_text = str_squish(sentence_split),
    sentence_chars = nchar(sentence_text),
    text_all = str_to_lower(sentence_text)
  ) %>%
  filter(
    !is.na(sentence_text),
    sentence_text != "",
    sentence_chars > 20
  )

# ============================================================
# 3) DICTIONARY SIGNALS (TIGHTENED + FINAL)
# ============================================================

sentences <- sentences %>%
  mutate(

    # -------------------------
    # COMPETITIVE
    # -------------------------
    is_competitive = str_detect(
      text_all,
      "\\b(rank|ranked|mmr|elo|leaderboard|tournament|matchmaking|placement|division|tier|queue|sr)\\b"
    ),

    # -------------------------
    # COSMETIC
    # -------------------------
    is_cosmetic = str_detect(
      text_all,
      "\\b(skin|skins|cosmetic|cosmetics|bundle|bundles|store|shop|emote|emotes|spray|sprays|mythic|legendary|epic|highlight intro|victory pose|weapon charm|charms|souvenir|player icon|name card|voice line|outfit|appearance|heirloom)\\b"
    ),

    # -------------------------
    # SEASONAL
    # -------------------------
    is_seasonal = str_detect(
      text_all,
      "\\b(event|battle pass|season|limited-time|ltm|challenge)\\b"
    ),

    # -------------------------
    # DIFFICULTY / GAMEPLAY
    # -------------------------
    is_difficulty = str_detect(
      text_all,
      "\\b(buff|nerf|balance|balanced|rework|tuning|cooldown|damage|healing|shield|armor|ultimate|passive|ability|abilities|scaling)\\b"
    )
  )

# ============================================================
# 4) CHARACTER ALLOCATION (CORE LOGIC)
# ============================================================

sentences <- sentences %>%
  mutate(
    comp_chars = sentence_chars * is_competitive,
    seas_chars = sentence_chars * is_seasonal,
    diff_chars = sentence_chars * is_difficulty,
    cos_chars  = sentence_chars * is_cosmetic
  )

# ============================================================
# 5) AGGREGATE TO PATCH LEVEL
# ============================================================

patch_levers <- sentences %>%
  group_by(game, event_id, event_date) %>%
  summarise(
    total_chars = sum(sentence_chars),

    abs_competitive = sum(comp_chars),
    abs_seasonal    = sum(seas_chars),
    abs_difficulty  = sum(diff_chars),
    abs_cosmetic    = sum(cos_chars),

    .groups = "drop"
  )

# ============================================================
# 6) NORMALIZE (YOUR VARIABLES)
# ============================================================

patch_levers <- patch_levers %>%
  mutate(
    rel_competitive = abs_competitive / total_chars,
    rel_seasonal    = abs_seasonal / total_chars,
    rel_difficulty  = abs_difficulty / total_chars,
    rel_cosmetic    = abs_cosmetic / total_chars
  )

# ============================================================
# 7) VALIDATION
# ============================================================

cat("\n--- MEAN SHARE CHECK ---\n")

patch_levers %>%
  summarise(
    mean_comp = mean(rel_competitive),
    mean_seas = mean(rel_seasonal),
    mean_diff = mean(rel_difficulty),
    mean_cos  = mean(rel_cosmetic)
  ) %>%
  print()

cat("\n--- RANGE CHECK ---\n")

patch_levers %>%
  summarise(
    across(starts_with("rel_"), list(min = min, max = max))
  ) %>%
  print()

cat("\n--- GAME BREAKDOWN ---\n")

patch_levers %>%
  group_by(game) %>%
  summarise(
    mean_comp = mean(rel_competitive),
    mean_seas = mean(rel_seasonal),
    mean_diff = mean(rel_difficulty),
    mean_cos  = mean(rel_cosmetic),
    .groups = "drop"
  ) %>%
  print(n = Inf)

# ============================================================
# 8) SAVE
# ============================================================

write_csv(sentences, "data_processed/patch_sentences.csv")
write_csv(patch_levers, "data_processed/patch_levers.csv")

cat("\n✅ DONE — Sentence-level lever dataset built correctly\n")