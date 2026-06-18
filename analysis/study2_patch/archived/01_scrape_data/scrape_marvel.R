# ============================================================
# SCRAPE — MARVEL RIVALS
# ============================================================

library(rvest)
library(httr)
library(dplyr)
library(stringr)
library(lubridate)
library(purrr)
library(readr)

UA_STRING <- "Mozilla/5.0 (compatible; PatchStudyBot/1.0)"
SLEEP_SEC <- 0.5

# ------------------------------------------------------------
# SAFE FETCH
# ------------------------------------------------------------
safe_read_html <- function(url) {
  tryCatch({
    Sys.sleep(SLEEP_SEC)
    res <- GET(url, user_agent(UA_STRING))
    if (status_code(res) != 200) return(NULL)
    read_html(res)
  }, error = function(e) {
    message("❌ Failed: ", url)
    return(NULL)
  })
}

# ------------------------------------------------------------
# EXTRACT PATCH
# ------------------------------------------------------------
extract_patch <- function(url) {
  
  pg <- safe_read_html(url)
  if (is.null(pg)) return(NULL)
  
  title <- pg %>%
    html_element("h1.artTitle") %>%
    html_text2() %>%
    str_trim()
  
  date_raw <- pg %>%
    html_element("p.date") %>%
    html_text2() %>%
    str_trim()
  
  event_date <- suppressWarnings(ymd(date_raw))
  
  content_html <- pg %>%
    html_element(".artText") %>%
    as.character()
  
  content_text <- pg %>%
    html_element(".artText") %>%
    html_text2() %>%
    str_squish()
  
  tibble(
    game = "Marvel Rivals",
    event_date = event_date,
    patch_title = title,
    source_url = url,
    content_text = content_text,
    content_html = content_html,
    word_count = str_count(content_text, "\\w+")
  )
}

# ------------------------------------------------------------
# GET LINKS
# ------------------------------------------------------------
base_url <- "https://www.marvelrivals.com/gameupdate/"

pg <- read_html(base_url)

total_pages <- pg %>%
  html_element(".hd_totalPages") %>%
  html_attr("value") %>%
  as.numeric()

page_urls <- c(
  base_url,
  paste0(base_url, "index_", 2:total_pages, ".html")
)

get_links_from_page <- function(url) {
  pg <- read_html(url)
  pg %>%
    html_elements("a.list-item") %>%
    html_attr("href") %>%
    na.omit() %>%
    unique()
}

all_links <- map_dfr(page_urls, ~ tibble(url = get_links_from_page(.x))) %>%
  mutate(
    url = if_else(
      str_detect(url, "^http"),
      url,
      paste0("https://www.marvelrivals.com", url)
    )
  ) %>%
  distinct()

# ------------------------------------------------------------
# SCRAPE
# ------------------------------------------------------------
patch_data <- map_dfr(all_links$url, extract_patch)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
write_csv(patch_data, "data_raw/marvel_patch_data.csv")

cat("✅ Marvel done\n")