# ============================================================
# SCRAPE — OVERWATCH
# ============================================================

library(tidyverse)
library(rvest)
library(httr)
library(lubridate)
library(stringr)
library(purrr)
library(readr)

BASE_URL <- "https://overwatch.blizzard.com/en-us/news/patch-notes/live"

safe_read <- function(url) {
  Sys.sleep(0.5)
  res <- GET(url)
  if (status_code(res) != 200) return(NULL)
  read_html(content(res, as = "text", encoding = "UTF-8"))
}

months <- c(
  "2026-03","2026-02","2026-01",
  "2025-12","2025-11","2025-10","2025-09","2025-08","2025-07",
  "2025-06","2025-05","2025-04","2025-03","2025-02","2025-01",
  "2024-12","2024-11","2024-10","2024-09","2024-08","2024-07",
  "2024-06","2024-05","2024-04","2024-03","2024-02","2024-01",
  "2023-12","2023-11","2023-10","2023-09","2023-08","2023-07",
  "2023-06","2023-05","2023-04","2023-03","2023-02","2023-01"
)

parse_month <- function(month) {

  parts <- str_split(month, "-", simplify = TRUE)
  url <- paste0(BASE_URL, "/", parts[1], "/", parts[2])

  pg <- safe_read(url)
  if (is.null(pg)) return(NULL)

  patches <- pg %>% html_elements(".PatchNotes-patch")
  if (length(patches) == 0) return(NULL)

  map_dfr(patches, function(p) {

    date <- p %>% html_element(".PatchNotes-date") %>% html_text2()
    title <- p %>% html_element(".PatchNotes-patchTitle") %>% html_text2()

    full_text <- p %>%
      html_elements("p, li") %>%
      html_text2() %>%
      paste(collapse = "\n")

    tibble(
      game = "Overwatch 2",
      event_date = mdy(date),
      patch_title = title,
      source_url = NA_character_,
      content_text = full_text,
      word_count = str_count(full_text, "\\w+")
    )
  })
}

overwatch_data <- map_dfr(months, parse_month) %>%
  filter(!is.na(event_date)) %>%
  distinct(event_date, patch_title, .keep_all = TRUE)

write_csv(overwatch_data, "data_raw/overwatch_patch_data.csv")

cat("✅ Overwatch done\n")