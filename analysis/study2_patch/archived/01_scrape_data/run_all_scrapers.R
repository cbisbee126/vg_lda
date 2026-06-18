# ============================================================
# RUN ALL SCRAPERS
# ============================================================

cat("Starting scraping pipeline...\n")

# Create folder in root
dir.create("data_raw", showWarnings = FALSE, recursive = TRUE)

# Run scripts FROM subfolder
source("01_scrape_data/scrape_marvel.R")
source("01_scrape_data/scrape_apex.R")
source("01_scrape_data/scrape_overwatch.R")

cat("\nAll scraping complete ✅\n")