# README --- Patch Engagement Analysis

## Overview

This project looks at how different types of patch content affect player
engagement in free-to-play games.

We treat each patch as an **event** and measure how player activity
behaves: - on the patch day - immediately after (vs. the day before) -
over the following week

We then relate that to what the patch actually contains.

------------------------------------------------------------------------

## Data

### 1. Patch Data

Patch notes are taken directly from Steam where developers make announcements about
changes to their game. See Step 01 below!

------------------------------------------------------------------------

### 2. Steam Data

`data_raw/*_steam_data.csv`

One file per game:
    - Apex Legends
    - Overwatch 2
    - Brawlhalla
    - War Thunder
    - The Finals
    - Marvel Rovals
    - PUBG
    - Counter-Strike 2

Raw structure: 
- `DateTime` (Collapsed to daily)
- `Players` (# peak pleayers) 
- `Average Players` (# average players)

We chose average players as it is a better metric for true engagement.

------------------------------------------------------------------------

## Pipeline

### Step 1: Collect Steam Game Announcement Data (01_stream_scrape.R)

- Using the Steam API we are able to use game ID to collect announcements made by the developer
- For robustness, annotate the events into four categories:
    - 1: All news = Anything pulled from Steam
    - 2: Broad updates = Any announcement that contains terms such as "patch", "update", "hotfix".
    This captures the broader  developer communication stream around game changes.
    - 3: Strict updates = Any announcement with a version number, indicative of patch note we are 
    most interested in. These are major changes to the game, including new seasons. ex v.2.0.0.1

------------------------------------------------------------------------

## Step 02: Build Developer Emphasis Measures (02_build_sections.R)
- This step takes the patch note text and breaks it into **sentences**.
- Sentences are natural units for conveying ideas in patch notes, so they are used to create the predictor variables for **developer emphasis**.
- Developer emphasis captures how much attention a patch note gives to different progression levers, such as:
  - competitive updates
  - new battle passes or seasonal content
  - difficulty and balance changes
  - cosmetics or identity-based rewards
- Each sentence is scanned for progression-lever keywords. If the sentence contains keywords linked to a lever, it is flagged for that lever.
- A sentence can be flagged for more than one lever. For example, a sentence about “ranked season rewards” may be flagged as both competitive and seasonal.
- For each patch note, the raw character count of sentences flagged for each lever is summed and divided by the total character count of the patch note text.
- The final output is a set of relative emphasis measures showing the share of each patch note devoted to each progression lever.
------------------------------------------------------------------------

### Step 3: Add Relevant Controls (03_add_controls.R)

- The data is loaded and relevant controls are added
- These are:
    - release_date = The official release date for each game.
    - days_since_release = Number of days between the game’s release date and the update date.
    - log_days_since_release = Logged version of days_since_release, used to reduce skew.
    - total_chars = Total character count used as the patch/update length measure.
    - log_total_chars = Logged version of total_chars, used as a length control.
    - new_season_patch = Indicator variable equal to 1 if the update happened within ±2 days of a known season/reset date, otherwise 0. +/- 2 days is used as some seasonal release patch notes came out the night prior to it was used in the game.

------------------------------------------------------------------------

### Step 4: Merge Steam Data (04_merge_steam_engagement_data.R)
The goal of this step is to merge engagement data with our patch data to be used in modeling.

We use lift to define the "hype" of the patch note itself. For robustness and to understand potential timing,
we test day of lift (day=0), immediately after (day=+1), and a bit more delayed (day=+2).

Retention stays the same and is defined as average of pre compared to post. 

Creates outcome variables, which are:
engagement_lift_0d = day 0 average log players - pre-window average log players
engagement_retention_0d = post days +1 to +7 average log players - day 0 average log players

engagement_lift_1d = days 0 to +1 average log players - pre-window average log players
engagement_retention_1d = post days +2 to +7 average log players - days 0 to +1 average log players

engagement_lift_2d = days 0 to +2 average log players - pre-window average log players
engagement_retention_2d = post days +3 to +7 average log players - days 0 to +2 average log players

------------------------------------------------------------------------

### Step 5: Comparison of Player vs Developer Emphasis (05_emphasis_comparison.R)

This code takes the prevalence of each of the four levers and compares frequency of mention.

This highlights any emphasis difference between what players actually talk about and developers
focus on in patch note text.

Makes a barchart to compare the differences between the two groups. 

------------------------------------------------------------------------

### Step 6: Models (06_model_testing.R)
 Due to the sturctured nature of the data, an HLM specifcation made the most sense. We have patch notes
 nested in 8 games over time. Therefore there is expected to be within and between variation.

 Another interesting thought is the cylical nature of updates, especially seasons. This leads
 to the thought that the levers may not be strictly linear in terms of engagement. This holds especially
 true when considering our engagement outcomes are about hype and retention, processes that 
 can be best captured in a u-shape.

DVs = lift and retention

IVs = competitive, cosmetic, seasonal, and difficulty levers

Controls = days since release, new_patch, lagged engagement
***Important to note that new_patch is actually the time around the new_season_patch. New season dates
were given manually for accuracy, however some games released the patch note before the new season started. 
This was found to be within +\- 2 days. Therefore new_patch is better understood as new_season_timing.***

Model 1: Basic HLM strucutre with the games as between subjects and time within. Uses predictors,
controls and outcomes.

Model 2: Tests the potential quadratic nature of the data by adding non-linear terms for all the
predictors.

Add in potnetially: Genre

------------------------------------------------------------------------

## Key Design Choices

-   Unit of analysis = patch day
-   Baseline = previous day
-   Retention = average of next 7 days
-   Logs used for stability and % interpretation

------------------------------------------------------------------------

## Final Output

The final output contains:
- A correlation table to see if any of the predictors are alarmingly correlated
- A table of model results for each model tested as well as its fit

Main file to look at = "/results/study2_hlm_models_clean_results_wide.csv"

Keep in mind that:
0d = lift is only on patch note release, and retention is 1+-7+ days after patch
1d = lift is average of 0-1+ days after patch note release, retentsion is 2+-7+
2d = lift is average of 0-2+ days after patch note release, retentions is 3+-7+

------------------------------------------------------------------------

## Summary

We model how patch content affects player engagement by looking at what
happens before, during, and after each patch release.
