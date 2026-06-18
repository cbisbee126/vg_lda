# ============================================================
# 01 BUILD STEAM ANNOUNCEMENT DATASET
# Goal:
#   1) Scrape all Steam developer announcements
#   2) Keep full annotated master dataset
#   3) Separate all news, broad updates, and strict updates
#   4) Do NOT code progression levers in this step
# ============================================================

rm(list = ls())

# ============================================================
# 0) PACKAGES
# ============================================================

required_packages <- c(
  "httr2",
  "jsonlite",
  "dplyr",
  "stringr",
  "purrr",
  "readr",
  "lubridate",
  "rvest",
  "tibble"
)

installed <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
}

invisible(lapply(required_packages, library, character.only = TRUE))

cat("✅ Packages ready\n")

# ============================================================
# 1) PARAMETERS
# ============================================================

dir.create("data_raw", showWarnings = FALSE, recursive = TRUE)
dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE, recursive = TRUE)

games <- tibble::tibble(
  game = c(
    "Marvel Rivals",
    "Apex Legends",
    "Overwatch 2",
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS",
    "War Thunder",
    "THE FINALS",
    "Brawlhalla"
  ),
  appid = c(
    2767030,
    1172470,
    2357570,
    730,
    578080,
    236390,
    2073850,
    291550
  )
)

NEWS_COUNT <- 5000
MIN_CHARS  <- 50

# ============================================================
# 2) HELPERS
# ============================================================

extract_announcement_url <- function(url, appid) {
  id <- stringr::str_extract(url, "\\d+")

  if (is.na(id)) return(url)

  paste0("https://store.steampowered.com/news/app/", appid, "/view/", id)
}

safe_read_html <- function(url) {
  tryCatch({
    Sys.sleep(0.3)

    httr2::request(url) |>
      httr2::req_user_agent("Mozilla/5.0") |>
      httr2::req_perform() |>
      httr2::resp_body_html()

  }, error = function(e) {
    message("❌ Failed: ", url)
    return(NULL)
  })
}

extract_full_text <- function(url) {
  pg <- safe_read_html(url)

  if (is.null(pg)) return(NA_character_)

  txt <- pg |>
    rvest::html_elements(".body") |>
    rvest::html_text2()

  if (length(txt) == 0) return(NA_character_)

  txt <- paste(txt, collapse = "\n") |>
    stringr::str_squish()

  if (nchar(txt) < MIN_CHARS) return(NA_character_)

  txt
}

needs_fallback <- function(txt) {
  is.na(txt) ||
    txt == "" ||
    stringr::str_detect(txt, "\\.\\.\\.$")
}

make_event_id <- function(game, raw_id, title) {
  if (!is.na(raw_id) && raw_id != "") {
    return(paste0("steam_", raw_id))
  }

  safe_game <- stringr::str_replace_all(
    stringr::str_to_lower(game),
    "[^a-z0-9]+",
    "_"
  )

  safe_title <- stringr::str_replace_all(
    stringr::str_to_lower(title),
    "[^a-z0-9]+",
    "_"
  )

  paste0(safe_game, "_", safe_title)
}

add_time_vars <- function(df) {
  df |>
    dplyr::arrange(game, event_date) |>
    dplyr::group_by(game) |>
    dplyr::mutate(
      year                      = lubridate::year(event_date),
      month                     = lubridate::month(event_date),
      announcement_number       = dplyr::row_number(),
      days_since_last_event     = as.numeric(event_date - dplyr::lag(event_date)),
      log_days_since_last_event = log1p(days_since_last_event)
    ) |>
    dplyr::ungroup()
}

collapse_one_post_per_game_day <- function(df) {
  df |>
    dplyr::group_by(game, event_date) |>
    dplyr::arrange(
      dplyr::desc(strict_update),
      dplyr::desc(broad_update),
      dplyr::desc(char_count),
      .by_group = TRUE
    ) |>
    dplyr::slice(1) |>
    dplyr::ungroup()
}

# ============================================================
# 3) MAIN SCRAPE LOOP
# ============================================================

all_data <- vector("list", nrow(games))

for (i in seq_len(nrow(games))) {

  game_name <- games$game[i]
  app_id    <- games$appid[i]

  cat("\n🚀 Scraping:", game_name, "\n")

  api_url <- paste0(
    "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/",
    "?appid=", app_id,
    "&count=", NEWS_COUNT,
    "&maxlength=0",
    "&format=json"
  )

  resp <- tryCatch({
    httr2::request(api_url) |>
      httr2::req_user_agent("Mozilla/5.0") |>
      httr2::req_perform()
  }, error = function(e) {
    message("❌ API request failed for ", game_name)
    return(NULL)
  })

  if (is.null(resp)) {
    all_data[[i]] <- tibble::tibble()
    next
  }

  json <- jsonlite::fromJSON(
    httr2::resp_body_string(resp),
    simplifyDataFrame = TRUE
  )

  if (length(json$appnews$newsitems) == 0) {
    all_data[[i]] <- tibble::tibble()
    next
  }

  df <- tibble::as_tibble(json$appnews$newsitems) |>
    dplyr::transmute(
      game         = game_name,
      raw_event_id = as.character(gid),
      event_date   = as.Date(lubridate::as_datetime(date)),
      patch_title  = stringr::str_squish(title),
      source_url   = url,
      api_contents = stringr::str_squish(contents)
    )

  df <- df |>
    dplyr::mutate(
      true_url = purrr::map_chr(
        source_url,
        ~ extract_announcement_url(.x, app_id)
      )
    )

  df <- df |>
    dplyr::mutate(
      full_text = purrr::pmap_chr(
        list(true_url, api_contents),
        function(url, txt) {
          if (needs_fallback(txt)) {
            scraped <- extract_full_text(url)
            if (!is.na(scraped)) return(scraped)
          }

          txt
        }
      )
    ) |>
    dplyr::mutate(
      full_text   = stringr::str_squish(full_text),
      patch_title = stringr::str_squish(patch_title)
    ) |>
    dplyr::filter(
      !is.na(full_text),
      full_text != ""
    ) |>
    dplyr::mutate(
      char_count     = nchar(full_text, type = "chars"),
      word_count     = stringr::str_count(full_text, "\\S+"),
      log_char_count = log1p(char_count),
      source_type    = "steam"
    ) |>
    dplyr::filter(char_count >= MIN_CHARS) |>
    dplyr::mutate(
      event_id = purrr::pmap_chr(
        list(game, raw_event_id, patch_title),
        make_event_id
      )
    ) |>
    dplyr::select(
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
      source_type
    )

  all_data[[i]] <- df
}

# ============================================================
# 4) COMBINE + CLEAN MASTER ANNOUNCEMENT DATASET
# ============================================================

news_raw <- dplyr::bind_rows(all_data) |>
  dplyr::filter(
    !is.na(event_date),
    !is.na(patch_title),
    patch_title != ""
  ) |>
  dplyr::mutate(
    patch_title = stringr::str_squish(patch_title),
    full_text   = stringr::str_squish(full_text),
    title_lower = stringr::str_to_lower(patch_title),
    text_all    = stringr::str_to_lower(paste(patch_title, full_text))
  ) |>
  dplyr::arrange(game, event_date) |>
  dplyr::distinct(game, event_id, .keep_all = TRUE)

cat("\n✅ Combined master announcement rows:", nrow(news_raw), "\n")

# ============================================================
# 5) ANNOUNCEMENT-TYPE PATTERNS ONLY
# No progression-lever coding in this script
# ============================================================

# Broad update language.
# This identifies general developer update communications.
broad_update_pattern <- paste(
  c(
    "update",
    "updates",
    "patch",
    "patches",
    "patch notes",
    "patch note",
    "hotfix",
    "hot fix",
    "notes",
    "release notes",
    "balance update",
    "balance patch"
  ),
  collapse = "|"
)

# Strict version indicators.
# This identifies posts that look like formal/versioned updates.
version_pattern <- paste(
  c(
    # v2.47, v.2.47, v 2.47
    "\\bv\\.?\\s*\\d+(\\.\\d+)+\\b",

    # version 2.47
    "\\bversion\\s*\\d+(\\.\\d+)+\\b",

    # patch 2.47, patch notes 2.47
    "\\bpatch\\s*(notes?)?\\s*\\d+(\\.\\d+)+\\b",

    # update 2.47
    "\\bupdate\\s*\\d+(\\.\\d+)+\\b",

    # hotfix 2.47, hotfix #12
    "\\bhot\\s*fix\\s*#?\\s*\\d+(\\.\\d+)*\\b",
    "\\bhotfix\\s*#?\\s*\\d+(\\.\\d+)*\\b",

    # build 12345
    "\\bbuild\\s*\\d+\\b",

    # 2.47 Patch Notes
    "\\b\\d+(\\.\\d+)+\\s*patch\\s*notes?\\b",

    # 2.47 Update
    "\\b\\d+(\\.\\d+)+\\s*update\\b",

    # Update 1, Update 2, etc.
    "\\bupdate\\s*#?\\s*\\d+\\b",

    # Patch 1, Patch 2, etc.
    "\\bpatch\\s*#?\\s*\\d+\\b"
  ),
  collapse = "|"
)

# Strict title language.
# This catches formal patch-note posts even without version numbers.
strict_update_title_pattern <- paste(
  c(
    "patch notes",
    "patch note",
    "hotfix",
    "hot fix",
    "release notes",
    "balance patch",
    "balance update"
  ),
  collapse = "|"
)

# General news/promo exclusions.
# These are not removed from the master file.
# They are only prevented from being classified as broad/strict update datasets.
exclude_pattern <- paste(
  c(
    "giveaway",
    "sweepstakes",
    "trailer",
    "teaser",
    "soundtrack",
    "merch",
    "merchandise",
    "fan art",
    "art contest",
    "creator spotlight",
    "community spotlight",
    "tournament",
    "championship",
    "esports",
    "e-sports",
    "sale",
    "discount",
    "free weekend",
    "wishlist",
    "dev diary",
    "developer diary",
    "behind the scenes"
  ),
  collapse = "|"
)

soft_news_pattern <- paste(
  c(
    "announcement",
    "community",
    "spotlight",
    "event",
    "trailer",
    "teaser",
    "sale",
    "esports",
    "tournament",
    "championship",
    "contest",
    "giveaway",
    "developer diary",
    "dev diary"
  ),
  collapse = "|"
)

# ============================================================
# 6) FLAGS — ANNOUNCEMENT TYPE ONLY
# ============================================================

news_annotated <- news_raw |>
  dplyr::mutate(
    exclude_hit =
      stringr::str_detect(title_lower, exclude_pattern),

    soft_news_hit =
      stringr::str_detect(title_lower, soft_news_pattern),

    broad_update_hit =
      stringr::str_detect(title_lower, broad_update_pattern) |
      stringr::str_detect(text_all, broad_update_pattern),

    # Version search is title-based to avoid false positives from body numbers.
    version_hit =
      stringr::str_detect(title_lower, version_pattern),

    # Strict update title search is title-based.
    strict_update_title_hit =
      stringr::str_detect(title_lower, strict_update_title_pattern),

    # Category 1: all news
    all_news =
      TRUE,

    # Category 2: broad updates
    broad_update =
      broad_update_hit & !exclude_hit,

    # Category 3: strict updates
    strict_update =
      !exclude_hit &
      (
        version_hit |
        strict_update_title_hit
      ),

    announcement_category = dplyr::case_when(
      strict_update ~ "strict_update",
      broad_update ~ "broad_update",
      exclude_hit ~ "excluded_news_or_promo",
      soft_news_hit ~ "soft_news",
      TRUE ~ "general_announcement"
    )
  )

# ============================================================
# 7) CREATE OUTPUT DATASETS
# ============================================================

# 7A. Full master file
all_news <- news_annotated

# 7B. Broad update dataset
broad_updates <- news_annotated |>
  dplyr::filter(broad_update) |>
  collapse_one_post_per_game_day()

# 7C. Strict update dataset
strict_updates <- news_annotated |>
  dplyr::filter(strict_update) |>
  collapse_one_post_per_game_day()

# ============================================================
# 8) ADD TIME VARIABLES
# ============================================================

all_news       <- add_time_vars(all_news)
broad_updates  <- add_time_vars(broad_updates)
strict_updates <- add_time_vars(strict_updates)

# ============================================================
# 9) VALIDATION
# ============================================================

cat("\n============================================================\n")
cat("MASTER ANNOUNCEMENT VALIDATION\n")
cat("============================================================\n")

cat("\n📊 Master rows per game:\n")
print(all_news |> dplyr::count(game, sort = TRUE))

cat("\n📊 Master date range:\n")
print(
  all_news |>
    dplyr::summarise(
      min_date = min(event_date, na.rm = TRUE),
      max_date = max(event_date, na.rm = TRUE)
    )
)

cat("\n📊 Step 1 flag summary by game:\n")
flag_summary <- all_news |>
  dplyr::group_by(game) |>
  dplyr::summarise(
    total_news_rows             = dplyr::n(),
    broad_update_hit_rows       = sum(broad_update_hit, na.rm = TRUE),
    broad_update_rows           = sum(broad_update, na.rm = TRUE),
    version_rows                = sum(version_hit, na.rm = TRUE),
    strict_update_title_rows    = sum(strict_update_title_hit, na.rm = TRUE),
    strict_update_rows          = sum(strict_update, na.rm = TRUE),
    excluded_rows               = sum(exclude_hit, na.rm = TRUE),
    soft_news_rows              = sum(soft_news_hit, na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    pct_broad_update = broad_update_rows / total_news_rows,
    pct_strict_update = strict_update_rows / total_news_rows
  ) |>
  dplyr::arrange(dplyr::desc(total_news_rows))

print(flag_summary, n = Inf)

cat("\n📊 Announcement category summary:\n")
category_summary <- all_news |>
  dplyr::count(game, announcement_category, sort = TRUE)

print(category_summary, n = Inf)

cat("\n============================================================\n")
cat("CONVENIENCE DATASET VALIDATION\n")
cat("============================================================\n")

cat("\n📊 Broad updates per game:\n")
print(broad_updates |> dplyr::count(game, sort = TRUE))

cat("\n📊 Strict updates per game:\n")
print(strict_updates |> dplyr::count(game, sort = TRUE))

cat("\n📊 Same-day multiple posts after collapse — broad updates:\n")
print(
  broad_updates |>
    dplyr::count(game, event_date, sort = TRUE) |>
    dplyr::filter(n > 1)
)

cat("\n📊 Same-day multiple posts after collapse — strict updates:\n")
print(
  strict_updates |>
    dplyr::count(game, event_date, sort = TRUE) |>
    dplyr::filter(n > 1)
)

# ============================================================
# 10) SAMPLE TITLES FOR MANUAL CHECKING
# ============================================================

cat("\n============================================================\n")
cat("MANUAL CHECK: SAMPLE TITLES\n")
cat("============================================================\n")

cat("\n🔎 Sample strict update titles:\n")
strict_sample <- strict_updates |>
  dplyr::select(game, event_date, patch_title, announcement_category, char_count) |>
  dplyr::arrange(game, dplyr::desc(event_date)) |>
  dplyr::group_by(game) |>
  dplyr::slice_head(n = 8) |>
  dplyr::ungroup()

print(strict_sample, n = 80)

cat("\n🔎 Sample broad update titles that are NOT strict updates:\n")
broad_not_strict_sample <- all_news |>
  dplyr::filter(
    broad_update,
    !strict_update
  ) |>
  dplyr::select(game, event_date, patch_title, announcement_category, char_count) |>
  dplyr::arrange(game, dplyr::desc(event_date)) |>
  dplyr::group_by(game) |>
  dplyr::slice_head(n = 8) |>
  dplyr::ungroup()

print(broad_not_strict_sample, n = 80)

cat("\n🔎 Sample excluded titles:\n")
excluded_sample <- all_news |>
  dplyr::filter(exclude_hit) |>
  dplyr::select(game, event_date, patch_title, announcement_category, char_count) |>
  dplyr::arrange(game, dplyr::desc(event_date)) |>
  dplyr::group_by(game) |>
  dplyr::slice_head(n = 5) |>
  dplyr::ungroup()

print(excluded_sample, n = 80)

# ============================================================
# 11) SAVE FILES
# ============================================================

# Full master annotated announcement file
readr::write_csv(
  all_news,
  "data_raw/all_games_news_annotated.csv"
)

readr::write_csv(
  all_news,
  "data_processed/all_games_news_annotated.csv"
)

# Broad update file
readr::write_csv(
  broad_updates,
  "data_raw/all_games_broad_updates.csv"
)

readr::write_csv(
  broad_updates,
  "data_processed/all_games_broad_updates.csv"
)

# Strict update file
readr::write_csv(
  strict_updates,
  "data_raw/all_games_strict_updates.csv"
)

readr::write_csv(
  strict_updates,
  "data_processed/all_games_strict_updates.csv"
)

# Diagnostics
readr::write_csv(
  flag_summary,
  "results/step1_flag_summary_by_game.csv"
)

readr::write_csv(
  category_summary,
  "results/step1_announcement_category_summary.csv"
)

readr::write_csv(
  strict_sample,
  "results/step1_sample_strict_update_titles.csv"
)

readr::write_csv(
  broad_not_strict_sample,
  "results/step1_sample_broad_not_strict_titles.csv"
)

readr::write_csv(
  excluded_sample,
  "results/step1_sample_excluded_titles.csv"
)

# ============================================================
# 12) FINAL MESSAGE
# ============================================================

cat("\n============================================================\n")
cat("✅ DONE — Step 1 Steam announcement datasets created\n")
cat("============================================================\n")

cat("\n📁 All news files:\n")
cat("   - data_raw/all_games_news_annotated.csv\n")
cat("   - data_processed/all_games_news_annotated.csv\n")

cat("\n📁 Broad update files:\n")
cat("   - data_raw/all_games_broad_updates.csv\n")
cat("   - data_processed/all_games_broad_updates.csv\n")

cat("\n📁 Strict update files:\n")
cat("   - data_raw/all_games_strict_updates.csv\n")
cat("   - data_processed/all_games_strict_updates.csv\n")

cat("\n📁 Diagnostic files:\n")
cat("   - results/step1_flag_summary_by_game.csv\n")
cat("   - results/step1_announcement_category_summary.csv\n")
cat("   - results/step1_sample_strict_update_titles.csv\n")
cat("   - results/step1_sample_broad_not_strict_titles.csv\n")
cat("   - results/step1_sample_excluded_titles.csv\n")

cat("\n🎯 All news rows:", nrow(all_news), "\n")
cat("🎯 Broad update rows:", nrow(broad_updates), "\n")
cat("🎯 Strict update rows:", nrow(strict_updates), "\n")

