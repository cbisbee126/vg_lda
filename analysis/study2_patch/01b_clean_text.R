rm(list = ls())

library(dplyr)
library(stringr)
library(readr)


# ============================================================
# 1. LOAD RAW UPDATE TEXT
# ============================================================

updates <- read_csv(
  "data/interim/study2/update_communication_days_raw.csv",
  show_col_types = FALSE
) |>
  mutate(
    event_date = as.Date(event_date),

    incomplete_warframe =
      game == "Warframe" &
      event_date < as.Date("2016-01-01") &
      str_length(raw_text) < 500 &
      str_detect(
        raw_text,
        regex(
          "full (patch )?notes|full notes here|full notes:|steamcommunity\\.com",
          ignore_case = TRUE
        )
      )
  )


# ============================================================
# 2. REMOVE INCOMPLETE WARFRAME REDIRECT POSTS
# ============================================================

cat("\n--- INCOMPLETE WARFRAME POSTS REMOVED ---\n")

updates |>
  filter(incomplete_warframe) |>
  select(
    event_date,
    titles,
    raw_text
  ) |>
  print(
    n = Inf,
    width = Inf
  )


cat(
  "\nRemoved:",
  sum(updates$incomplete_warframe),
  "incomplete Warframe posts\n"
)


updates <- updates |>
  filter(!incomplete_warframe) |>
  select(-incomplete_warframe)


# ============================================================
# 3. CLEAN TEXT
# ============================================================

updates_clean <- updates |>
  mutate(

    # --------------------------------------------------------
    # Clean titles
    # --------------------------------------------------------

    titles = titles |>

      # Standardize line breaks
      str_replace_all("\r\n?", " ") |>

      # Fix common HTML entities
      str_replace_all("&amp;", "&") |>
      str_replace_all("&quot;", "\"") |>
      str_replace_all("&#39;|&apos;", "'") |>
      str_replace_all("&nbsp;|&#160;", " ") |>

      # Normalize quotes and dashes
      str_replace_all("[\u2018\u2019]", "'") |>
      str_replace_all("[\u201C\u201D]", "\"") |>
      str_replace_all("[\u2013\u2014]", " - ") |>
      str_replace_all("\u2026", "...") |>

      # Normalize nonbreaking spaces
      str_replace_all(
        "[\u00A0\u202F]",
        " "
      ) |>

      # Remove invisible characters
      str_replace_all(
        "[\u200B-\u200D\u2060\uFEFF\u3164\uFE0F]",
        ""
      ) |>

      # Final spacing
      str_squish(),


    # --------------------------------------------------------
    # Clean communication body
    # --------------------------------------------------------

    clean_text = raw_text |>

      # Standardize line breaks
      str_replace_all(
        "\r\n?",
        "\n"
      ) |>

      # Remove Steam images
      str_replace_all(
        regex(
          "\\[img[^\\]]*\\][\\s\\S]*?\\[/img\\]",
          ignore_case = TRUE
        ),
        " "
      ) |>

      # Preserve Steam paragraph, heading, and list breaks
      str_replace_all(
        regex(
          "\\[/?(p|h1|h2|h3|hr|tr|td|th|list|olist)[^\\]]*\\]|\\[/?\\*\\]",
          ignore_case = TRUE
        ),
        "\n"
      ) |>

      # Remove remaining BBCode
      str_replace_all(
        "\\[[^\\]]*\\]",
        " "
      ) |>

      # Preserve HTML line breaks
      str_replace_all(
        regex(
          "<br\\s*/?>",
          ignore_case = TRUE
        ),
        "\n"
      ) |>

      # Remove URLs
      str_replace_all(
        "https?://\\S+|www\\.\\S+",
        " "
      ) |>

      # Remove remaining HTML
      str_replace_all(
        "<[^>]+>",
        " "
      ) |>

      # Fix common HTML entities
      str_replace_all("&amp;", "&") |>
      str_replace_all("&quot;", "\"") |>
      str_replace_all("&#39;|&apos;", "'") |>
      str_replace_all("&nbsp;|&#160;", " ") |>
      str_replace_all(
        "&#145;|&#146;|&#8216;|&#8217;",
        "'"
      ) |>
      str_replace_all(
        "&#147;|&#148;|&#8220;|&#8221;",
        "\""
      ) |>
      str_replace_all(
        "&#150;|&#151;|&#8211;|&#8212;",
        " - "
      ) |>

      # Remove any remaining HTML entities
      str_replace_all(
        "&[#A-Za-z0-9]+;",
        " "
      ) |>

      # Normalize quotes and dashes
      str_replace_all(
        "[\u2018\u2019]",
        "'"
      ) |>
      str_replace_all(
        "[\u201C\u201D]",
        "\""
      ) |>
      str_replace_all(
        "[\u2013\u2014]",
        " - "
      ) |>
      str_replace_all(
        "\u2026",
        "..."
      ) |>

      # Normalize nonbreaking spaces
      str_replace_all(
        "[\u00A0\u202F]",
        " "
      ) |>

      # Remove decorative symbols
      str_replace_all(
        "\\p{So}",
        " "
      ) |>
      str_replace_all(
        "[\u2190-\u21FF]",
        " "
      ) |>
      str_replace_all(
        "[\u2022\u203B\u2020\u2981\u25A2]",
        " "
      ) |>
      str_replace_all(
        "[\u25B7\U0001F3FB-\U0001F3FF]",
        " "
      ) |>

      # Remove zero-width and invisible characters
      str_replace_all(
        "[\u200B-\u200D\u2060\uFEFF\u3164\uFE0F]",
        ""
      ) |>

      # Remove malformed leftover BBCode brackets
      str_replace_all(
        "/[A-Za-z0-9_*]+\\]",
        " "
      ) |>
      str_replace_all(
        "[\\[\\]]",
        " "
      ) |>

      # Remove lines containing only angle brackets
      str_replace_all(
        regex(
          "^\\s*[<>]+\\s*$",
          multiline = TRUE
        ),
        ""
      ) |>

      # Remove backslashes used only as line formatting
      # Keeps backslashes that occur inside meaningful text
      str_replace_all(
        regex(
          "^\\s*\\\\+\\s*|\\s*\\\\+\\s*$",
          multiline = TRUE
        ),
        ""
      ) |>

      # Remove any remaining punctuation-only lines
      str_replace_all(
        regex(
          "^\\s*[[:punct:]]+\\s*$",
          multiline = TRUE
        ),
        ""
      ) |>

      # Clean spaces while preserving line structure
      str_replace_all(
        "[ \t]+",
        " "
      ) |>
      str_replace_all(
        " *\n+ *",
        "\n"
      ) |>

      # Remove leading and trailing whitespace
      str_trim()
  )


# ============================================================
# 4. INSPECT RAW VS. CLEANED TEXT
# ============================================================

cat("\n--- RAW VS CLEANED EXAMPLES ---\n")

updates_clean |>
  select(
    game,
    update_id,
    raw_text,
    clean_text
  ) |>
  slice_head(n = 10) |>
  print(
    n = 10,
    width = Inf
  )


# ============================================================
# 5. CHECK LINE STRUCTURE
# ============================================================

cat("\n--- LINE STRUCTURE ---\n")

updates_clean |>
  summarise(
    updates = n(),

    updates_with_lines =
      sum(
        str_detect(
          clean_text,
          "\n"
        )
      ),

    mean_lines =
      mean(
        str_count(
          clean_text,
          "\n"
        ) + 1
      )
  ) |>
  print()


# ============================================================
# 6. CHECK FOR LEFTOVER FORMATTING
# ============================================================

cat("\n--- LEFTOVER FORMATTING ---\n")

format_checks <- updates_clean |>
  summarise(

    urls =
      sum(
        str_detect(
          clean_text,
          "https?://|www\\."
        )
      ),

    brackets =
      sum(
        str_detect(
          clean_text,
          "\\[|\\]"
        )
      ),

    html =
      sum(
        str_detect(
          clean_text,
          "<[^>]+>"
        )
      ),

    html_entities =
      sum(
        str_detect(
          clean_text,
          "&[#A-Za-z0-9]+;"
        )
      ),

    punctuation_only_lines =
      sum(
        str_detect(
          clean_text,
          regex(
            "^\\s*[[:punct:]]+\\s*$",
            multiline = TRUE
          )
        )
      ),

    line_boundary_backslashes =
      sum(
        str_detect(
          clean_text,
          regex(
            "^\\s*\\\\+|\\\\+\\s*$",
            multiline = TRUE
          )
        )
      ),

    invisible_characters =
      sum(
        str_detect(
          clean_text,
          "[\u200B-\u200D\u2060\uFEFF\u3164\uFE0F]"
        )
      ),

    nonbreaking_spaces =
      sum(
        str_detect(
          clean_text,
          "[\u00A0\u202F]"
        )
      ),

    replacement_characters =
      sum(
        str_detect(
          clean_text,
          "\uFFFD"
        )
      )
  )

print(
  format_checks,
  width = Inf
)


# ============================================================
# 7. CLEAN SAMPLE
# ============================================================

cat("\n--- CLEAN SAMPLE ---\n")

updates_clean |>
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


# ============================================================
# 8. BASIC CHECKS
# ============================================================

stopifnot(

  !anyDuplicated(
    updates_clean$update_id
  ),

  !anyDuplicated(
    updates_clean[
      c(
        "game",
        "event_date"
      )
    ]
  ),

  !any(
    is.na(
      updates_clean$clean_text
    )
  ),

  all(
    updates_clean$clean_text != ""
  ),

  format_checks$urls == 0,

  format_checks$brackets == 0,

  format_checks$html == 0,

  format_checks$html_entities == 0,

  format_checks$punctuation_only_lines == 0,

  format_checks$line_boundary_backslashes == 0,

  format_checks$invisible_characters == 0,

  format_checks$nonbreaking_spaces == 0,

  format_checks$replacement_characters == 0
)


# ============================================================
# 9. SAVE
# ============================================================

updates_clean |>
  select(-raw_text) |>
  write_csv(
    "data/interim/study2/update_communication_days_clean.csv"
  )


cat("\nDONE - CLEAN UPDATE COMMUNICATION DATA\n")