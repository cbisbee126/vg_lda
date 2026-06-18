# Playing for Progress - Videogame F2P Engagement Study

Code and analysis for a two-study paper on free-to-play (F2P) videogame engagement:

- **Study 1 (Python, LDA):** topic modeling of YouTube comments to surface the progression
  systems players discuss - competitive, cosmetics/identity, seasonal, and difficulty/balance.
- **Study 2 (R, HLM):** scraping Steam patch notes, measuring developer "emphasis" on those same
  systems, and testing how it relates to player-engagement lift and retention.

## Structure

- `analysis/study1_lda/` - Python LDA pipeline (`01_clean` -> `02_ksweep` -> `03_topic_table` -> `04_topic_distributions`)
- `analysis/study2_patch/` - R pipeline (`01_steam_scrape` -> ... -> `06_model_testing`)
- `src/` - reusable Python modules for Study 1
- `data/` - inputs and intermediates (not tracked; shared via Box)
- `output/` - generated tables and figures (not tracked)
- `docs/` - methodology notes and reference tables

## Running

Run scripts from the repo root.

Study 1 (Python):
```
uv venv && uv pip install -r requirements.txt
python analysis/study1_lda/01_clean_fp.py
```

Study 2 (R) - deps: tidyverse, lme4, lmerTest, broom.mixed, performance, httr2, jsonlite, rvest:
```
Rscript analysis/study2_patch/01_steam_scrape.R
```

Raw data is shared via Box (not in the repo). See each pipeline's README for details.

## Contact

cbisbe1@lsu.edu
