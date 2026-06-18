# ============================================================
# APEX LEGENDS PATCH SCRAPER (FINAL — ROBUST)
# ============================================================

rm(list = ls())

library(tidyverse)
library(rvest)
library(httr)
library(lubridate)
library(stringr)
library(purrr)
library(readr)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

BASE_URL <- "https://www.ea.com/games/apex-legends/news"

UA_STRING <- "Mozilla/5.0 (compatible; PatchStudyBot/1.0)"
SLEEP_SEC <- 1

dir.create("data_raw/cache", showWarnings = FALSE, recursive = TRUE)
dir.create("data_processed", showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# SAFE FETCH
# ------------------------------------------------------------

safe_read_html <- function(url) {
  Sys.sleep(SLEEP_SEC)
  
  res <- GET(url, user_agent(UA_STRING))
  
  if (status_code(res) != 200) {
    message("❌ Failed: ", url)
    return(NULL)
  }
  
  content(res, as = "parsed")
}

# ------------------------------------------------------------
# STEP 1: GET LINKS (FIXED — MULTIPLE PAGES)
# ------------------------------------------------------------

message("🔎 Scraping archive pages...")

pages <- paste0(BASE_URL, "?page=", 0:3)

links <- c()

for (p in pages) {
  
  message("🔎 Scraping: ", p)
  
  archive_page <- safe_read_html(p)
  
  if (is.null(archive_page)) next
  
  page_links <- archive_page %>%
    html_elements("a") %>%
    html_attr("href") %>%
    unique()
  
  # Keep only news links
  page_links <- page_links[str_detect(page_links, "/news/")]
  
  # Make absolute
  page_links <- ifelse(str_detect(page_links, "^http"),
                       page_links,
                       paste0("https://www.ea.com", page_links))
  
  # Remove junk
  page_links <- page_links[!page_links %in% c("https://www.ea.com")]
  
  # Keep only relevant pages
  page_links <- page_links[str_detect(page_links, "event|patch|update|season")]
  
  links <- c(links, page_links)
}

links <- unique(links)

message("✅ Total candidate links: ", length(links))

# ------------------------------------------------------------
# STEP 2: EXTRACT PATCH DATA
# ------------------------------------------------------------

extract_patch <- function(url) {
  
  message("📄 Processing: ", url)
  
  page <- safe_read_html(url)
  
  if (is.null(page)) return(NULL)
  
  # -------------------------
  # TITLE
  # -------------------------
  title <- page %>%
    html_element("h1") %>%
    html_text(trim = TRUE)
  
  # -------------------------
  # DATE (ROBUST — TEXT MATCH)
  # -------------------------
  all_text <- page %>%
    html_elements("p") %>%
    html_text2()
  
  date_raw <- all_text[str_detect(
    all_text,
    "January|February|March|April|May|June|July|August|September|October|November|December"
  )][1]
  
  event_date <- suppressWarnings(mdy(date_raw))
  
  # -------------------------
  # BODY TEXT
  # -------------------------
  body <- all_text %>%
    paste(collapse = " ")
  
  # -------------------------
  # VALIDATION
  # -------------------------
  if (is.na(event_date)) {
    message("⚠️ Skipping (no date): ", url)
    return(NULL)
  }
  
  if (is.na(title) || title == "") {
    message("⚠️ Skipping (no title): ", url)
    return(NULL)
  }
  
  # -------------------------
  # OUTPUT
  # -------------------------
  tibble(
    game = "Apex Legends",
    url = url,
    event_id = str_extract(url, "[^/]+$"),
    event_date = event_date,
    title = title,
    text = body
  )
}

# ------------------------------------------------------------
# STEP 3: RUN PIPELINE
# ------------------------------------------------------------

patch_data <- map_dfr(links, function(url) {
  tryCatch(
    extract_patch(url),
    error = function(e) {
      message("❌ ERROR: ", url)
      return(NULL)
    }
  )
})

# ------------------------------------------------------------
# STEP 4: CLEAN + SORT
# ------------------------------------------------------------

patch_data <- patch_data %>%
  arrange(event_date) %>%
  distinct(url, .keep_all = TRUE)

# ------------------------------------------------------------
# STEP 5: SAVE
# ------------------------------------------------------------

write_csv(patch_data, "data_raw/apex_patch_data.csv")

message("\n🎯 DONE — Apex patches collected: ", nrow(patch_data))