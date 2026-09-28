# ============================================================
# 02 BUILD PROGRESSION EMPHASIS MEASURES
# ============================================================

rm(list = ls())

library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(tibble)

dir.create(
  "data/interim/study2",
  showWarnings = FALSE,
  recursive = TRUE
)


# ============================================================
# 1. LOAD DATA
# ============================================================

updates <- read_csv(
  "data/interim/study2/update_communication_days_clean.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date)
  )


# Keep only each game's free-to-play period

f2p_dates <- tibble(
  game = c(
    "Marvel Rivals",
    "Apex Legends",
    "Overwatch 2",
    "Counter-Strike 2",
    "War Thunder",
    "THE FINALS",
    "Brawlhalla",
    "Warframe"
  ),
  f2p_start = as.Date(c(
    "2024-12-06",
    "2019-02-04",
    "2022-10-04",
    "2018-12-06",
    "2013-08-15",
    "2023-12-07",
    "2015-11-03",
    "2013-03-25"
  ))
)

updates <- updates |>
  left_join(
    f2p_dates,
    by = "game"
  ) |>
  filter(
    event_date >= f2p_start
  )


# ============================================================
# 2. COMPETITIVE PROGRESSION
# ============================================================

competitive_pattern <- regex(
  paste(
    c(
      "\\branked\\b",

      "\\branked (mode|play|queue|match|matches|season|ladder|division|rewards?)\\b",

      "\\bcompetitive (mode|play|queue|rank|season|rewards?|points?|rating)\\b",

      "\\bskill rating\\b",
      "\\bskill rank\\b",

      "\\brank tiers?\\b",
      "\\brank divisions?\\b",

      "\\bmmr\\b",
      "\\belo\\b",

      "\\bleaderboards?\\b",

      "\\bpromotion matches?\\b",
      "\\bdemotion matches?\\b",

      "\\brank reset\\b",
      "\\branked reset\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# War Thunder also uses "ranked" for vehicle tiers

war_thunder_ranked_exclude <- regex(
  paste(
    c(
      "\\b(high|higher|low|lower|top)[- ]ranked\\b",
      "\\branked (aircraft|vehicles?|ground vehicles?|airfields?|ships?|tanks?)\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# ============================================================
# 3. COSMETICS / IDENTITY
# ============================================================

# Terms with a clear cosmetic or identity meaning

cosmetic_pattern <- regex(
  paste(
    c(
      "\\bskins?\\b",
      "\\bcosmetics?\\b",

      "\\boutfits?\\b",
      "\\bcostumes?\\b",

      "\\bemotes?\\b",
      "\\bavatars?\\b",
      "\\btaunts?\\b",

      "\\bplayer icons?\\b",
      "\\bname cards?\\b",
      "\\bnameplates?\\b",
      "\\bvoice lines?\\b",

      "\\bcharms?\\b",

      "\\bcamouflage\\b",
      "\\bcamouflages\\b",
      "\\bpaintjobs?\\b",

      "\\bdecals?\\b",
      "\\bstickers?\\b",

      "\\bweapon finishes?\\b",

      "\\bhighlight intros?\\b",
      "\\bvictory poses?\\b",

      "\\bko effects?\\b",
      "\\bsidekicks?\\b",

      "\\bsouvenirs?\\b",
      "\\bheirlooms?\\b",

      "\\bsyandanas?\\b",
      "\\bephemera\\b",

      "\\bmusic kits?\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# Ambiguous terms that need cosmetic context

cosmetic_context_pattern <- regex(
  paste(
    c(
      "\\b(player|profile) banners?\\b",
      "\\bbanner frames?\\b",

      "\\b(character|appearance|cosmetic) customization\\b",
      "\\bcustomization\\b.{0,30}\\b(character|appearance|cosmetic|skin|outfit)\\b",

      "\\bhelmet skins?\\b",
      "\\bcosmetic helmets?\\b",

      "\\bholo sprays?\\b",
      "\\b(exclusive|cosmetic) sprays?\\b",
      "\\bspray wheel\\b",
      "\\bsprays?.{0,20}\\b(emotes?|emojis?|rewards?)\\b",

      "\\bcamos?\\b.{0,30}\\b(vehicle|aircraft|tank|ship|weapon|skin|cosmetic|appearance|reward|unlock|purchase)\\b",
      "\\b(vehicle|aircraft|tank|ship|weapon|skin|cosmetic|appearance|reward|unlock|purchase)\\b.{0,30}\\bcamos?\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


cosmetic_exclude_pattern <- regex(
  paste(
    c(
      "\\bspray patterns?\\b",
      "\\bspray control\\b",
      "\\bspray-and-pray\\b",

      "\\bcharms?\\s+(her way|his way|their way|foes?|enemies?|opponents?)\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# ============================================================
# 4. SEASONAL PROGRESSION
# ============================================================

# Direct seasonal progression terminology

seasonal_pattern <- regex(
  paste(
    c(
      "\\bseasons?\\b",
      "\\bseasonal\\b",

      "\\bbattle ?passes?\\b",

      "\\blimited[- ]time\\b",

      "\\breward tracks?\\b",

      "\\bfestivals?\\b",

      "\\bnightwave\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# Events and challenges count when tied to
# progression, rewards, or recurring cycles

seasonal_context_pattern <- regex(
  paste(
    c(
      "\\b(limited[- ]time|seasonal|collection|holiday|in-game) events?\\b",

      "\\banniversary (events?|celebrations?|rewards?)\\b",

      "\\bevents?.{0,30}\\b(rewards?|tracks?|passes?|challenges?|missions?|items?|store|earn|unlock|collect)\\b",

      "\\b(rewards?|tracks?|passes?|challenges?|missions?|earn|unlock|collect).{0,30}\\bevents?\\b",

      "\\b(daily|weekly|seasonal|season|event) challenges?\\b",

      "\\bchallenges?.{0,30}\\b(rewards?|season|battle ?pass|events?)\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# Clearly different use of "season"

seasonal_exclude_pattern <- regex(
  "\\b(esports?|championship|pro league) seasons?\\b",
  ignore_case = TRUE
)


# ============================================================
# 5. DIFFICULTY / BALANCE
# ============================================================

# Explicit gameplay tuning language

difficulty_direct_pattern <- regex(
  paste(
    c(
      "\\bnerfs?\\b",
      "\\bnerfed\\b",
      "\\bnerfing\\b",

      "\\bbuffed\\b",
      "\\b(buff|buffs) (to|for)\\b",
      "\\b(weapon|hero|character|legend|ability|gameplay) buffs?\\b",

      "\\bbalance changes?\\b",
      "\\bbalance updates?\\b",
      "\\bbalance adjustments?\\b",

      "\\b(gameplay|weapon|hero|character|legend|ability) balance\\b",

      "\\b(gameplay|weapon|hero|character|legend|ability|map) reworks?\\b",

      "\\b(gameplay|weapon|hero|character|legend|ability) tuning\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# Common gameplay statistics

gameplay_stats <- paste(
  c(
    "reload speed",
    "rate of fire",
    "fire rate",
    "movement speed",
    "attack speed",
    "ammo capacity",
    "recover time",
    "recovery time",

    "damage",
    "healing",
    "health",
    "shield",
    "armor",

    "ammo",

    "recoil",
    "spread",

    "reload",

    "cooldown",
    "cooldowns",

    "stun",
    "force",

    "knockback"
  ),
  collapse = "|"
)


change_terms <- paste(
  c(
    "increase", "increased", "increases", "increasing",
    "decrease", "decreased", "decreases", "decreasing",
    "reduce", "reduced", "reduces", "reducing",
    "lower", "lowered", "lowers", "lowering",
    "raise", "raised", "raises", "raising",
    "adjust", "adjusted", "adjusts", "adjusting",
    "change", "changed", "changes", "changing",
    "modify", "modified", "modifies", "modifying"
  ),
  collapse = "|"
)


# Directional gameplay-stat changes

difficulty_stat_pattern <- regex(
  paste0(
    "\\b(",
    change_terms,
    ")\\b.{0,30}\\b(",
    gameplay_stats,
    ")\\b",

    "|",

    "\\b(",
    gameplay_stats,
    ")\\b.{0,30}\\b(",
    change_terms,
    ")\\b"
  ),
  ignore_case = TRUE
)


# Explicit numeric stat changes

number_pattern <- "-?\\d+(?:\\.\\d+)?%?"

difficulty_numeric_pattern <- regex(
  paste0(
    "\\b(",
    gameplay_stats,
    ")\\b.{0,50}(",

    "from\\s+",
    number_pattern,
    ".{0,15}\\s+to\\s+",
    number_pattern,

    "|",

    number_pattern,
    "\\s*(?:->|→)\\s*",
    number_pattern,

    ")"
  ),
  ignore_case = TRUE
)


# Explicit gameplay difficulty and scaling language

difficulty_context_pattern <- regex(
  paste(
    c(
      "\\benemy difficulty\\b",
      "\\bbot difficulty\\b",

      "\\bdifficulty levels?\\b",
      "\\bdifficulty settings?\\b",

      "\\bhigher difficulty\\b",
      "\\blower difficulty\\b",

      "\\bdifficulty scaling\\b",

      "\\bdamage scaling\\b",
      "\\bhealth scaling\\b",
      "\\benemy scaling\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# Recurring non-gameplay uses of otherwise relevant terms

difficulty_exclude_pattern <- regex(
  paste(
    c(
      "\\b(game|server|system|matchmaking) health\\b",

      "\\b(stereo|audio) spread\\b",

      "\\bair force\\b",

      "\\barmor material\\b",

      "\\bammo rack (filling|fill)\\b"
    ),
    collapse = "|"
  ),
  ignore_case = TRUE
)


# ============================================================
# 6. SPLIT COMMUNICATIONS INTO TEXT UNITS
# ============================================================

units <- updates |>
  select(
    game,
    update_id,
    event_date,
    clean_text
  ) |>
  mutate(
    sentence = str_split(
      clean_text,
      "(?<=[.!?])\\s+|\\n+"
    )
  ) |>
  unnest(sentence) |>
  mutate(
    sentence = str_squish(sentence),
    sentence_chars = nchar(sentence)
  ) |>
  filter(
    sentence != "",
    !str_detect(
      sentence,
      "^[[:punct:][:space:]]+$"
    )
  ) |>
  mutate(

    competitive =
      str_detect(
        sentence,
        competitive_pattern
      ) &
      !(
        game == "War Thunder" &
          str_detect(
            sentence,
            war_thunder_ranked_exclude
          )
      ),

    cosmetic =
      (
        str_detect(
          sentence,
          cosmetic_pattern
        ) |
          str_detect(
            sentence,
            cosmetic_context_pattern
          )
      ) &
      !str_detect(
        sentence,
        cosmetic_exclude_pattern
      ),

    seasonal =
      (
        str_detect(
          sentence,
          seasonal_pattern
        ) |
          str_detect(
            sentence,
            seasonal_context_pattern
          )
      ) &
      !str_detect(
        sentence,
        seasonal_exclude_pattern
      ),

    difficulty =
      (
        str_detect(
          sentence,
          difficulty_direct_pattern
        ) |
          str_detect(
            sentence,
            difficulty_stat_pattern
          ) |
          str_detect(
            sentence,
            difficulty_numeric_pattern
          ) |
          str_detect(
            sentence,
            difficulty_context_pattern
          )
      ) &
      !str_detect(
        sentence,
        difficulty_exclude_pattern
      ),

    any_theme =
      competitive |
      cosmetic |
      seasonal |
      difficulty
  )


# ============================================================
# 7. CALCULATE UPDATE-LEVEL EMPHASIS
# ============================================================

emphasis <- units |>
  group_by(
    game,
    update_id,
    event_date
  ) |>
  summarise(
    retained_units = n(),

    retained_chars =
      sum(sentence_chars),

    rel_competitive =
      sum(sentence_chars * competitive) /
      retained_chars,

    rel_cosmetic =
      sum(sentence_chars * cosmetic) /
      retained_chars,

    rel_seasonal =
      sum(sentence_chars * seasonal) /
      retained_chars,

    rel_difficulty =
      sum(sentence_chars * difficulty) /
      retained_chars,

    rel_any_theme =
      sum(sentence_chars * any_theme) /
      retained_chars,

    .groups = "drop"
  )


# Add emphasis measures back to update-level data

update_emphasis <- updates |>
  left_join(
    emphasis,
    by = c(
      "game",
      "update_id",
      "event_date"
    )
  ) |>
  mutate(
    across(
      c(
        retained_units,
        retained_chars,
        rel_competitive,
        rel_cosmetic,
        rel_seasonal,
        rel_difficulty,
        rel_any_theme
      ),
      ~ replace_na(.x, 0)
    ),

    char_count =
      nchar(clean_text),

    word_count =
      str_count(
        clean_text,
        "\\S+"
      )
  )


# ============================================================
# 8. INSPECT
# ============================================================

cat("\n--- F2P SAMPLE ---\n")

update_emphasis |>
  group_by(game) |>
  summarise(
    updates = n(),
    first_date = min(event_date),
    last_date = max(event_date),
    .groups = "drop"
  ) |>
  print(
    n = Inf,
    width = Inf
  )


cat("\n--- MEAN EMPHASIS ---\n")

update_emphasis |>
  summarise(
    competitive =
      mean(rel_competitive),

    cosmetic =
      mean(rel_cosmetic),

    seasonal =
      mean(rel_seasonal),

    difficulty =
      mean(rel_difficulty),

    any_theme =
      mean(rel_any_theme),

    zero_theme_updates =
      sum(rel_any_theme == 0)
  ) |>
  print(width = Inf)


cat("\n--- NONZERO UPDATES ---\n")

update_emphasis |>
  summarise(
    competitive =
      sum(rel_competitive > 0),

    cosmetic =
      sum(rel_cosmetic > 0),

    seasonal =
      sum(rel_seasonal > 0),

    difficulty =
      sum(rel_difficulty > 0)
  ) |>
  print(width = Inf)


cat("\n--- EMPHASIS BY GAME ---\n")

update_emphasis |>
  group_by(game) |>
  summarise(
    updates = n(),

    competitive =
      mean(rel_competitive),

    cosmetic =
      mean(rel_cosmetic),

    seasonal =
      mean(rel_seasonal),

    difficulty =
      mean(rel_difficulty),

    .groups = "drop"
  ) |>
  print(
    n = Inf,
    width = Inf
  )


# ============================================================
# 9. BASIC CHECKS
# ============================================================

stopifnot(
  nrow(update_emphasis) ==
    nrow(updates),

  !anyDuplicated(
    update_emphasis$update_id
  ),

  !anyDuplicated(
    update_emphasis[
      c(
        "game",
        "event_date"
      )
    ]
  ),

  !any(
    is.na(
      update_emphasis$clean_text
    )
  ),

  all(
    update_emphasis$rel_competitive >= 0 &
      update_emphasis$rel_competitive <= 1
  ),

  all(
    update_emphasis$rel_cosmetic >= 0 &
      update_emphasis$rel_cosmetic <= 1
  ),

  all(
    update_emphasis$rel_seasonal >= 0 &
      update_emphasis$rel_seasonal <= 1
  ),

  all(
    update_emphasis$rel_difficulty >= 0 &
      update_emphasis$rel_difficulty <= 1
  ),

  all(
    update_emphasis$rel_any_theme >= 0 &
      update_emphasis$rel_any_theme <= 1
  )
)


# ============================================================
# 10. SAVE
# ============================================================

write_csv(
  update_emphasis,
  "data/interim/study2/update_progression_emphasis.csv"
)