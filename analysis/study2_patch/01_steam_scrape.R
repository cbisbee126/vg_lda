# ============================================================
# 01 STEAM UPDATE COMMUNICATION SCRAPE
# ============================================================
#
# PURPOSE
# -------
# Retrieve official Steam communications for the eight Study 2
# games and identify current/live update-related communications.
#
# SOURCE RULE
# -----------
# Keep only official Steam feeds:
#   steam_community_announcements
#   steam_updates
#
# SAMPLE RULE
# -----------
# Include title-based signals of current/live game changes such
# as updates, patches, seasons, store changes, rotations, events,
# releases, and launches. Exclude clear previews/future notices,
# maintenance/server notices, and unrelated esports/community/
# promotional posts unless a direct update signal is also present.
#
# Multiple qualifying posts for the same game-date are combined.
# Progression content is NOT used to determine sample eligibility.
#
# OUTPUT
# ------
# data/interim/study2/update_communication_days.csv
# ============================================================

rm(list = ls())

library(httr2)
library(jsonlite)
library(dplyr)
library(stringr)
library(readr)
library(lubridate)
library(tibble)

dir.create("data/interim/study2", showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1) GAMES + API SETTINGS
# ============================================================

games <- tibble(
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

BATCH_SIZE <- 1000
MAX_PAGES <- 100
OFFICIAL_FEEDS <- c(
  "steam_community_announcements",
  "steam_updates"
)
OFFICIAL_FEEDS_PARAMETER <- paste(OFFICIAL_FEEDS, collapse = ",")

# ============================================================
# 2) HELPERS
# ============================================================

ensure_api_columns <- function(df) {
  char_vars <- c(
    "gid", "title", "url", "contents",
    "author", "feedlabel", "feedname"
  )

  for (x in char_vars) {
    if (!(x %in% names(df))) df[[x]] <- NA_character_
  }

  if (!("date" %in% names(df))) df$date <- NA_real_
  df
}

fetch_news_page <- function(app_id, enddate = NULL) {
  req <- request(
    "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
  ) |>
    req_url_query(
      appid = app_id,
      count = BATCH_SIZE,
      maxlength = 0,
      feeds = OFFICIAL_FEEDS_PARAMETER,
      format = "json"
    ) |>
    req_user_agent("Mozilla/5.0") |>
    req_retry(max_tries = 3)

  if (!is.null(enddate)) {
    req <- req |> req_url_query(enddate = enddate)
  }

  response <- tryCatch(
    req |> req_perform(),
    error = function(e) {
      message("API request failed for appid ", app_id, ": ", conditionMessage(e))
      NULL
    }
  )

  if (is.null(response)) return(tibble())

  json <- fromJSON(
    resp_body_string(response),
    simplifyDataFrame = TRUE
  )

  newsitems <- json$appnews$newsitems
  if (is.null(newsitems) || length(newsitems) == 0) return(tibble())

  ensure_api_columns(as_tibble(newsitems))
}

clean_news_page <- function(df, game_name, app_id) {
  if (nrow(df) == 0) return(tibble())

  df |>
    transmute(
      game = game_name,
      appid = app_id,
      announcement_id = as.character(gid),
      unix_date = as.numeric(date),
      event_date = as.Date(as_datetime(date)),
      title = str_squish(coalesce(as.character(title), "")),
      full_text = str_squish(coalesce(as.character(contents), "")),
      feedname = str_to_lower(str_squish(coalesce(as.character(feedname), ""))),
      feedlabel = str_to_lower(str_squish(coalesce(as.character(feedlabel), ""))),
      author = str_squish(coalesce(as.character(author), "")),
      source_url = coalesce(as.character(url), "")
    ) |>
    filter(
      !is.na(event_date),
      announcement_id != ""
    )
}

# ============================================================
# 3) SCRAPE COMPLETE OFFICIAL HISTORY
# ============================================================

all_game_posts <- vector("list", nrow(games))
pagination_audit <- vector("list", nrow(games))

for (i in seq_len(nrow(games))) {
  game_name <- games$game[i]
  app_id <- games$appid[i]

  cat("\n============================================================\n")
  cat("SCRAPING:", game_name, "\n")
  cat("============================================================\n")

  pages <- list()
  page_number <- 1
  current_enddate <- NULL
  seen_ids <- character()
  stop_reason <- NA_character_

  while (page_number <= MAX_PAGES) {
    raw_page <- fetch_news_page(app_id, current_enddate)

    if (nrow(raw_page) == 0) {
      stop_reason <- "no_more_posts"
      break
    }

    page <- clean_news_page(raw_page, game_name, app_id)
    page_new <- page |> filter(!(announcement_id %in% seen_ids))

    cat("Page", page_number, "- new rows:", nrow(page_new), "\n")

    if (nrow(page_new) == 0) {
      stop_reason <- "no_new_posts"
      break
    }

    pages[[page_number]] <- page_new
    seen_ids <- c(seen_ids, page_new$announcement_id)

    if (nrow(raw_page) < BATCH_SIZE) {
      stop_reason <- "complete"
      break
    }

    oldest_timestamp <- min(page$unix_date, na.rm = TRUE)
    new_enddate <- oldest_timestamp - 1

    if (!is.null(current_enddate) && new_enddate >= current_enddate) {
      stop_reason <- "pagination_failed"
      break
    }

    current_enddate <- new_enddate
    page_number <- page_number + 1
    Sys.sleep(0.2)
  }

  game_posts <- bind_rows(pages) |>
    distinct(announcement_id, .keep_all = TRUE)

  all_game_posts[[i]] <- game_posts
  pagination_audit[[i]] <- tibble(
    game = game_name,
    official_posts = nrow(game_posts),
    first_date = if (nrow(game_posts) > 0) min(game_posts$event_date) else as.Date(NA),
    last_date = if (nrow(game_posts) > 0) max(game_posts$event_date) else as.Date(NA),
    stop_reason = stop_reason
  )
}

official_posts <- bind_rows(all_game_posts) |>
  distinct(game, announcement_id, .keep_all = TRUE) |>
  arrange(game, event_date)

pagination_check <- bind_rows(pagination_audit)

cat("\n--- PAGINATION CHECK ---\n")
print(pagination_check, n = Inf, width = Inf)

if (any(pagination_check$stop_reason == "pagination_failed")) {
  stop("Pagination failed for at least one game.")
}

unexpected <- official_posts |>
  filter(!(feedname %in% OFFICIAL_FEEDS))

if (nrow(unexpected) > 0) {
  stop("Unexpected non-official feed entered the sample.")
}

cat("\n--- OFFICIAL SOURCE CHECK ---\n")
official_posts |>
  count(game, feedname, sort = TRUE) |>
  print(n = Inf, width = Inf)

# ============================================================
# 4) TITLE-BASED UPDATE-COMMUNICATION RULE
# ============================================================

update_pattern <- paste0(
  "\\b(",
  "update|updates|patch|patches|patch notes|hotfix|hot fix|",
  "season|midseason|mid-season|",
  "store update|shop update|store rotation|shop rotation|rotation|",
  "event|release notes|",
  "now live|is live|goes live|live now|available now|out now|",
  "released|launch|launches|launched|begins|starts|arrives|is here",
  ")\\b"
)

future_pattern <- paste0(
  "\\b(",
  "preview|roadmap|teaser|trailer|coming soon|coming next|",
  "coming tomorrow|coming next week|upcoming|first look|",
  "reveal date|reveals? on|dev diary|developer diary|devblog|",
  "in development|livestream|live stream",
  ")\\b"
)

maintenance_pattern <- paste0(
  "\\b(",
  "maintenance|server status|scheduled downtime|downtime|outage|",
  "server maintenance|server update",
  ")\\b"
)

esports_pattern <- paste0(
  "\\b(",
  "esports?|tournament|championship|qualifier|qualifiers|",
  "pro league|global series|world championship|grand finals|",
  "match schedule|tournament recap",
  ")\\b"
)

other_exclude_pattern <- paste0(
  "\\b(",
  "twitch drops?|drop campaign|giveaway|pre-order|preorder|",
  "merchandise|screenshot competition|community spotlight|",
  "weekly bans? notice|ban notice",
  ")\\b"
)

direct_update_pattern <- paste0(
  "\\b(",
  "update|updates|patch|patches|hotfix|hot fix|",
  "season|midseason|mid-season|store update|shop update|rotation|",
  "now live|is live|goes live|released|launches|launched",
  ")\\b"
)

classified_posts <- official_posts |>
  mutate(
    title_lower = str_to_lower(title),
    update_signal = str_detect(title_lower, update_pattern),
    direct_update_signal = str_detect(title_lower, direct_update_pattern),
    future_signal = str_detect(title_lower, future_pattern),
    maintenance_signal = str_detect(title_lower, maintenance_pattern),
    esports_signal = str_detect(title_lower, esports_pattern),
    other_exclude_signal = str_detect(title_lower, other_exclude_pattern),
    update_communication =
      update_signal &
      !future_signal &
      !maintenance_signal &
      !(esports_signal & !direct_update_signal) &
      !(other_exclude_signal & !direct_update_signal)
  )

# ============================================================
# 5) AUDIT + COLLAPSE TO ONE GAME-DAY
# ============================================================

audit <- classified_posts |>
  summarise(
    official_posts = n(),
    update_related = sum(update_signal),
    future_excluded = sum(update_signal & future_signal),
    maintenance_excluded = sum(update_signal & maintenance_signal),
    final_update_posts = sum(update_communication)
  )

cat("\n--- UPDATE COMMUNICATION AUDIT ---\n")
print(audit, width = Inf)

cat("\n--- QUALIFYING POSTS BY GAME ---\n")
classified_posts |>
  group_by(game) |>
  summarise(
    official_posts = n(),
    qualifying_posts = sum(update_communication),
    pct_qualifying = mean(update_communication),
    .groups = "drop"
  ) |>
  arrange(desc(qualifying_posts)) |>
  print(n = Inf, width = Inf)

update_posts <- classified_posts |>
  filter(update_communication)

update_days <- update_posts |>
  group_by(game, appid, event_date) |>
  summarise(
    n_update_posts = n(),
    announcement_ids = paste(announcement_id, collapse = " | "),
    titles = paste(unique(title), collapse = " || "),
    combined_text = paste(
      str_squish(paste(title, full_text)),
      collapse = "\n\n"
    ),
    .groups = "drop"
  ) |>
  mutate(
    update_id = paste0(
      str_replace_all(str_to_lower(game), "[^a-z0-9]+", "_"),
      "__",
      format(event_date, "%Y%m%d")
    ),
    text_chars = nchar(combined_text)
  ) |>
  select(
    game, appid, update_id, event_date,
    n_update_posts, announcement_ids, titles,
    combined_text, text_chars
  ) |>
  arrange(game, event_date)

if (nrow(update_days |> count(game, event_date) |> filter(n > 1)) > 0) {
  stop("Duplicate game-date events remain after same-day collapse.")
}

if (any(is.na(update_days$combined_text) | update_days$combined_text == "")) {
  warning("At least one update day has no usable text.")
}

cat("\n--- FINAL UPDATE-COMMUNICATION SAMPLE ---\n")
update_days |>
  group_by(game) |>
  summarise(
    update_days = n(),
    first_update = min(event_date),
    last_update = max(event_date),
    mean_posts_per_day = mean(n_update_posts),
    median_text_chars = median(text_chars),
    .groups = "drop"
  ) |>
  arrange(desc(update_days)) |>
  print(n = Inf, width = Inf)

# ============================================================
# 6) SAVE ONE FILE
# ============================================================

output_file <- "data/interim/study2/update_communication_days.csv"
write_csv(update_days, output_file)

cat("\nDONE - Step 01\n")
cat("Official posts examined:", nrow(official_posts), "\n")
cat("Qualifying update posts:", nrow(update_posts), "\n")
cat("Final update-communication days:", nrow(update_days), "\n")
cat("Saved:", output_file, "\n")
