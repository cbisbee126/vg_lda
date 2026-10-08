rm(list = ls())

library(httr2)
library(dplyr)
library(stringr)
library(readr)
library(tibble)

dir.create("data/interim/study2", showWarnings = FALSE, recursive = TRUE)

study_end <- as.Date("2026-09-25")


# Games
games <- tibble(
  game = c(
    "Marvel Rivals",
    "Apex Legends",
    "Overwatch 2",
    "Counter-Strike 2",
    "War Thunder",
    "THE FINALS",
    "Brawlhalla",
    "Warframe",
    "Dota 2",
    "PUBG: BATTLEGROUNDS",
    "The First Descendant",
    "Delta Force",
    "NARAKA: BLADEPOINT",
    "Halo Infinite",
    "Once Human"
  ),
  appid = c(
    2767030, 1172470, 2357570, 730,
    236390, 2073850, 291550, 230410,
    570, 578080, 2074920, 2507950,
    1203220, 1240440, 2139460
  )
)


feeds <- c(
  "steam_community_announcements",
  "steam_updates"
)


# Scrape one game's official Steam posts

scrape_game <- function(game, appid) {

  pages <- list()
  seen_ids <- character()

  enddate <- as.numeric(
    as.POSIXct(
      "2026-09-25 23:59:59",
      tz = "UTC"
    )
  )

  for (page_num in 1:100) {

    request <- request(
      "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
    ) |>
      req_url_query(
        appid = appid,
        count = 1000,
        maxlength = 0,
        feeds = paste(feeds, collapse = ","),
        enddate = enddate
      ) |>
      req_retry(max_tries = 3)

    news <- request |>
      req_perform() |>
      resp_body_json(simplifyVector = TRUE)

    news <- news$appnews$newsitems

    if (is.null(news) || NROW(news) == 0) {
      break
    }

    raw_n <- NROW(news)

    page <- as_tibble(news) |>
      transmute(
        game = game,
        appid = appid,
        announcement_id = as.character(gid),
        unix_date = as.numeric(date),
        event_date = as.Date(
          as.POSIXct(
            date,
            origin = "1970-01-01",
            tz = "UTC"
          )
        ),
        title = coalesce(as.character(title), ""),
        body_raw = coalesce(as.character(contents), ""),
        feedname = str_to_lower(
          coalesce(as.character(feedname), "")
        )
      ) |>
      filter(
        !is.na(event_date),
        event_date <= study_end,
        announcement_id != "",
        !announcement_id %in% seen_ids
      )

    if (nrow(page) == 0) {
      break
    }

    pages[[page_num]] <- page
    seen_ids <- c(seen_ids, page$announcement_id)

    if (raw_n < 1000) {
      break
    }

    enddate <- min(page$unix_date, na.rm = TRUE) - 1

    Sys.sleep(0.2)
  }

  bind_rows(pages) |>
    distinct(announcement_id, .keep_all = TRUE)
}


# Scrape all games

posts <- vector("list", nrow(games))

for (i in 1:nrow(games)) {

  cat("Scraping:", games$game[i], "\n")

  posts[[i]] <- scrape_game(
    games$game[i],
    games$appid[i]
  )
}

official_posts <- bind_rows(posts) |>
  distinct(
    game,
    event_date,
    title,
    body_raw,
    .keep_all = TRUE
  ) |>
  arrange(game, event_date)


# Identify update communications

update_pattern <- regex(
  "\\b(update|updates|patch|patches|patch notes|hotfix|hot fix|season|midseason|mid-season|event|store update|shop update|store rotation|shop rotation|rotation|release notes)\\b",
  ignore_case = TRUE
)

live_pattern <- regex(
  "\\b(now live|is live|live now|starts now|starts today|begins today|arrives|is here|available now)\\b",
  ignore_case = TRUE
)

exclude_pattern <- regex(
  "\\b(preview|pre-release|prerelease|roadmap|teaser|trailer|coming|upcoming|tomorrow|next week|next month|near future|first look|reveal|reveals|revealed|get ready|launch event|update announcement|reveal date|launches on|arrives on|starts on|begins on|ending soon|dev diary|developer diary|dev ?blog|dev server|in development|maintenance|server status|downtime|outage|server maintenance|server update|website)\\b|\\[development\\]",
  ignore_case = TRUE
)

promo_pattern <- regex(
  "\\b(esports?|tournament|championship|qualifier|qualifiers|pro league|global series|world championship|grand finals|invitational|group stage|participating teams?|fantasy league|community showdown|match schedule|on-site event|onsite event|games week|best players|tennocon|twitch drops?|giveaway|free weekend|pre-order|preorder|merch|merchandise|sale)\\b",
  ignore_case = TRUE
)

strong_update_pattern <- regex(
  "\\b(patch|patches|patch notes|hotfix|hot fix|store update|shop update|store rotation|shop rotation|rotation|release notes)\\b",
  ignore_case = TRUE
)


# Classify posts

classified_posts <- official_posts |>
  mutate(
    update_signal = str_detect(title, update_pattern),
    live_signal = str_detect(title, live_pattern),
    excluded = str_detect(title, exclude_pattern),
    promotional = str_detect(title, promo_pattern),
    strong_update = str_detect(title, strong_update_pattern),

    update_communication =
      (update_signal | live_signal) &
      !excluded &
      !(promotional & !strong_update)
  )

update_posts <- classified_posts |>
  filter(update_communication)


# Combine qualifying posts from the same game and day

update_days_raw <- update_posts |>
  group_by(game, appid, event_date) |>
  summarise(
    n_update_posts = n(),
    announcement_ids = paste(
      announcement_id,
      collapse = " | "
    ),
    titles = paste(
      unique(title),
      collapse = " || "
    ),
    raw_text = paste(
      paste(title, body_raw, sep = "\n"),
      collapse = "\n\n"
    ),
    .groups = "drop"
  ) |>
  mutate(
    update_id = paste0(
      str_replace_all(
        str_to_lower(game),
        "[^a-z0-9]+",
        "_"
      ),
      "__",
      format(event_date, "%Y%m%d")
    )
  ) |>
  select(
    game,
    appid,
    update_id,
    event_date,
    n_update_posts,
    announcement_ids,
    titles,
    raw_text
  ) |>
  arrange(game, event_date)


# Inspect

cat("\n--- POSTS BY GAME ---\n")

official_posts |>
  count(game, feedname) |>
  print(n = Inf)

cat("\n--- CLASSIFICATION ---\n")

classified_posts |>
  group_by(game) |>
  summarise(
    official_posts = n(),
    update_signal = sum(update_signal),
    live_signal = sum(live_signal),
    included = sum(update_communication),
    .groups = "drop"
  ) |>
  print(n = Inf)

cat("\n--- UPDATE DAYS ---\n")

update_days_raw |>
  count(game, name = "update_days") |>
  print(n = Inf)

cat("\n--- SAMPLE INCLUDED TITLES ---\n")

classified_posts |>
  filter(update_communication) |>
  select(game, event_date, title) |>
  slice_head(n = 20) |>
  print(n = 20)

cat("\n--- FINAL DATA ---\n")

glimpse(update_days_raw)


# Basic checks

stopifnot(
  all(official_posts$feedname %in% feeds),
  all(official_posts$event_date <= study_end),
  all(update_days_raw$event_date <= study_end),
  !anyDuplicated(
    official_posts[c("game", "announcement_id")]
  ),
  !anyDuplicated(
    update_days_raw[c("game", "event_date")]
  ),
  !anyDuplicated(update_days_raw$update_id),
  !any(is.na(update_days_raw$raw_text)),
  all(update_days_raw$raw_text != "")
)


# Save

write_csv(
  update_days_raw,
  "data/interim/study2/update_communication_days_raw.csv"
)