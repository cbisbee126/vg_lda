# README — Study 2 Steam Communication and Engagement Analysis

## Overview

Study 2 examines how official gaming-firm communications about game updates and consumer-facing activities are associated with short-term Steam engagement in free-to-play games.

We identify qualifying official update-related communications distributed through Steam, measure the relative emphasis placed on four progression dimensions, and relate these communication characteristics to changes in daily Steam engagement.

The four progression dimensions are:

- Competitive progression
- Cosmetics and identity
- Seasonal progression
- Difficulty and balance

The final engagement analysis uses three complementary specifications:

1. A 3-day pre/post model excluding the communication day
2. A 3-day pre/post sensitivity model including the communication day
3. A daily distributed-lag robustness model covering Day 0 through Day +3

The Study 2 sample includes eight free-to-play games:

- Apex Legends
- Overwatch 2
- Brawlhalla
- War Thunder
- THE FINALS
- Marvel Rivals
- PUBG: BATTLEGROUNDS
- Counter-Strike 2


------------------------------------------------------------------------

## Data

### 1. Official Steam Communication Data

Official game communications are retrieved using the Steam News API.

The initial communication pool is restricted to two official Steam feeds:

- `steam_community_announcements`
- `steam_updates`

Step 01 then identifies qualifying current/live update-related communications using title-based inclusion and exclusion rules.

Multiple qualifying communications occurring for the same game on the same calendar day are combined into a single game-day communication event because Steam engagement is measured daily.


### 2. Steam Engagement Data

Steam engagement files are stored in:

`data/raw/study2/`

There is one Steam file per game.

The raw files contain:

- `DateTime`
- `Players`
- `Average Players`

Steam observations are collapsed to one observation per game-day.

Daily average Steam players is used as the engagement measure and is log transformed for analysis.


------------------------------------------------------------------------

## Pipeline

### Step 01: Collect Official Steam Communications (`01_steam_scrape.R`)

This step retrieves the available history of official Steam communications for the eight games using the Steam News API.

The script:

- restricts the source pool to the `steam_community_announcements` and `steam_updates` feeds;
- uses pagination to retrieve the available communication history rather than relying on a single API page;
- identifies qualifying current/live update-related communications using title-based signals;
- combines multiple qualifying communications occurring for the same game on the same calendar day; and
- does not use progression-system content to determine whether a communication enters the sample.

Qualifying communication signals include terms related to:

- updates
- patches and patch notes
- hotfixes
- seasons and midseasons
- store or shop updates
- rotations
- events
- releases
- launches
- currently available or live content

The script excludes communications clearly focused on future rather than current activity, including:

- previews
- roadmaps
- teasers
- trailers
- upcoming announcements
- developer diaries
- maintenance
- server-status notices
- scheduled downtime

Unrelated esports, promotional, and community communications are also excluded unless the title contains a direct game-update signal.

Output:

`data/interim/study2/update_communication_days.csv`


------------------------------------------------------------------------

### Step 02: Build Gaming-Firm Emphasis Measures (`02_build_sections.R`)

This step measures the relative emphasis placed on the four progression dimensions within each qualifying communication.

The communication text is divided into sentences. Sentences containing 20 or fewer characters are removed.

Each retained sentence is evaluated using the four progression dictionaries:

- Competitive progression
- Cosmetics and identity
- Seasonal progression
- Difficulty and balance

A sentence can be classified into more than one progression dimension. For example, a sentence describing ranked season rewards may be classified as both competitive and seasonal.

For each communication, the script calculates the character count of sentences classified into each dimension.

Relative emphasis is calculated as:

`characters in matched sentences / total retained sentence characters`

The resulting measures are:

- `rel_competitive`
- `rel_cosmetic`
- `rel_seasonal`
- `rel_difficulty`

Because sentences may match multiple dimensions, these measures are not required to sum to 1 within a communication.

The script also calculates diagnostic measures including sentence counts, matched character counts, and sentence overlap across progression dimensions.

Output:

`data/interim/study2/patch_levers_sentence.csv`


------------------------------------------------------------------------

### Step 03: Add Communication Controls (`03_add_controls.R`)

This step adds the communication-level controls used in the engagement models.

The controls are:

- `log_total_chars`  
  Logged total character count of the communication.

- `log_avg_sentence_chars`  
  Logged average sentence length.

- `season_related_communication`  
  Indicator equal to 1 when the communication title explicitly references a season or midseason.

The season-related communication indicator should not be interpreted as a verified season-launch date. It captures whether the communication itself explicitly references a season or midseason.

Calendar variables are also created for later use in the fixed-effects models.

Observed game age is not created in this step. It is created in Step 04 using the beginning of each game's available Steam engagement series.

Output:

`data/interim/study2/patch_levers_with_controls.csv`


------------------------------------------------------------------------

### Step 03b: Check Communication Spacing (`03b_check_communication_spacing.R`)

This diagnostic examines how frequently qualifying official update-related communications occur.

Two types of spacing are calculated.

#### Consecutive Communication Gaps

For each game, the script calculates the number of days between consecutive qualifying communications.

The diagnostic reports:

- mean days between consecutive communications;
- median days between consecutive communications;
- minimum and maximum gaps; and
- the percentage of consecutive gaps within 3, 5, and 7 days.

#### Nearest Communication Spacing

For each focal communication, the script identifies the closest qualifying communication occurring either before or after it.

The diagnostic reports:

- mean nearest-communication gap;
- median nearest-communication gap; and
- the percentage of communications with another qualifying communication within 3, 5, and 7 days.

These diagnostics are used to assess how frequently communication windows overlap and to motivate the shorter 3-day engagement window.

This script does not modify the analysis sample or save an additional data file.


------------------------------------------------------------------------

### Step 04: Merge Steam Engagement Data (`04_merge_steam_engagement_data.R`)

This step merges the communication features with daily Steam engagement and creates the datasets used in the final models.

Steam observations are first collapsed to one observation per game-day.

A complete calendar is then created for each game so that a one-day lag always represents one actual calendar day rather than simply the previous observed Steam row.

Communications are restricted to dates falling within the available Steam engagement history for each game.

Observed game age is calculated relative to the beginning of each game's available Steam series.

Communication predictors are standardized before model estimation.


#### Primary Pre/Post Outcome: Day 0 Excluded

The primary specification compares average logged Steam engagement before and after each qualifying communication.

Pre-period:

`Days -3, -2, and -1`

Post-period:

`Days +1, +2, and +3`

The outcome is:

`engagement_change_excl_day0 = post-period mean - pre-period mean`

Day 0 is excluded because a Steam communication can be posted at different times within the calendar day.


#### Day 0 Sensitivity Outcome

A second outcome tests whether the results depend on excluding the communication day.

Pre-period:

`Days -3, -2, and -1`

Post-period:

`Day 0, Day +1, and Day +2`

The outcome is:

`engagement_change_incl_day0 = post-period mean - pre-period mean`

Both the pre- and post-periods therefore contain three days.


#### Daily Panel

Step 04 also creates a unique daily game-date panel for the distributed-lag robustness model.

Communication variables are observed on the date a qualifying communication occurs and are coded as zero on dates without a qualifying communication.

Step 06 then creates the Day 0 through Day +3 lagged communication exposures.

Output:

`data/interim/study2/study2_analysis_data.rds`

The saved object contains:

- `event_data`
- `daily_panel`
- `steam_coverage`


------------------------------------------------------------------------

### Step 05: Compare Consumer and Gaming-Firm Emphasis (`05_emphasis_comparison.R`)

This step addresses the descriptive comparison between Study 1 consumer discourse and Study 2 gaming-firm communication.

For Study 2, the script calculates the mean relative emphasis placed on:

- Competitive progression
- Cosmetics and identity
- Seasonal progression
- Difficulty and balance

These four means are then normalized so that their combined emphasis sums to 1.

The normalized Study 2 values are compared with the normalized Study 1 consumer-discourse values.

This provides a descriptive comparison of which progression systems receive relatively greater attention in consumer discourse versus gaming-firm communication.

Outputs:

`output/tables/study2/comparison_table.csv`

`output/figures/study2/comparison_chart.png`


------------------------------------------------------------------------

### Step 06: Estimate Engagement Models (`06_model_testing.R`)

Step 06 estimates the final Study 2 engagement models.

The focal predictors are:

- Competitive progression emphasis
- Cosmetic and identity emphasis
- Seasonal progression emphasis
- Difficulty and balance emphasis

Communication-level controls include:

- logged communication length;
- logged average sentence length; and
- season-related communication.

Observed game age is also included as a control.


#### Model 1: Primary Pre/Post Model

The primary model uses:

Pre-period:

`Days -3 to -1`

Post-period:

`Days +1 to +3`

Day 0 is excluded.

The dependent variable is:

`engagement_change_excl_day0`

The script first estimates an intercept-only model testing the average pre/post engagement change across qualifying communications.

It then estimates the content model relating the four progression-emphasis measures and controls to the pre/post engagement change.


#### Model 2: Day 0 Sensitivity Model

The sensitivity specification uses:

Pre-period:

`Days -3 to -1`

Post-period:

`Days 0 to +2`

The dependent variable is:

`engagement_change_incl_day0`

The script first estimates an intercept-only model testing the average pre/post engagement change under this alternative timing specification.

It then estimates a content model using the same predictors, controls, fixed effects, and clustering strategy as the primary content model.

The purpose of this specification is to determine whether conclusions depend on excluding Day 0.


#### Model 3: Daily Distributed-Lag Robustness Model

The robustness model uses one observation per actual game-day.

For engagement observed on calendar day t:

- `t0` = communication occurring today
- `t1` = communication occurring one day earlier
- `t2` = communication occurring two days earlier
- `t3` = communication occurring three days earlier

Communication occurrence, progression emphasis, communication length, average sentence length, and season-related communication are entered at all four horizons simultaneously.

Because all recent communications enter the same model, nearby communications do not need to be treated as isolated events.


#### Fixed Effects

The two multivariable pre/post models and the daily distributed-lag model include:

- game fixed effects;
- weekday fixed effects;
- calendar month fixed effects; and
- calendar year fixed effects.

The intercept-only pre/post tests estimate the average engagement change and use game-month clustered standard errors without additional fixed effects.


#### Standard Errors

Standard errors are clustered by game-month across all models.


#### Multicollinearity Diagnostics

Variance inflation factors are calculated after removing the same fixed effects used in the corresponding multivariable model.

VIF diagnostics are produced for:

- the primary pre/post content model;
- the Day 0 sensitivity content model; and
- the daily distributed-lag model.


Outputs:

`output/tables/study2/study2_model_results.csv`

`output/tables/study2/study2_vif_results.csv`

`output/figures/study2/study2_daily_lag_emphasis.png`


------------------------------------------------------------------------

## Key Design Choices

- Initial communication source = official Steam announcement and update feeds
- Communication unit = qualifying official game-day communication
- Multiple qualifying posts on the same game-day are combined
- Progression content does not determine communication eligibility
- Progression emphasis is measured using sentence-level character shares
- Sentences may contribute to more than one progression dimension
- Primary pre-period = Days -3 to -1
- Primary post-period = Days +1 to +3
- Primary specification excludes Day 0
- Sensitivity post-period = Days 0 to +2
- Daily robustness unit = unique game-date
- Daily robustness horizon = Day 0 through Day +3
- Engagement measure = logged daily average Steam players
- Communication predictors are standardized for model estimation
- Game age is based on observed Steam-series coverage
- Multivariable model fixed effects = game, weekday, calendar month, and calendar year
- Standard errors are clustered by game-month


------------------------------------------------------------------------

## Main Outputs

The main Study 2 outputs are:

### Study 1 vs. Study 2 Comparison

`output/tables/study2/comparison_table.csv`

`output/figures/study2/comparison_chart.png`


### Engagement Models

`output/tables/study2/study2_model_results.csv`

This file contains results for:

- the overall pre/post change excluding Day 0;
- the primary content model excluding Day 0;
- the overall pre/post change including Day 0;
- the content sensitivity model including Day 0; and
- the daily distributed-lag robustness model.


### Multicollinearity Diagnostics

`output/tables/study2/study2_vif_results.csv`


### Daily Distributed-Lag Figure

`output/figures/study2/study2_daily_lag_emphasis.png`


------------------------------------------------------------------------

## Recommended Run Order

Run the Study 2 scripts in the following order:

1. `01_steam_scrape.R`
2. `02_build_sections.R`
3. `03_add_controls.R`
4. `03b_check_communication_spacing.R`
5. `04_merge_steam_engagement_data.R`
6. `05_emphasis_comparison.R`
7. `06_model_testing.R`

Step 01 retrieves live Steam communication data. Therefore, rerunning the full pipeline at a later date may add newly available communications even when the communication-selection logic has not changed.


------------------------------------------------------------------------

## Summary

Study 2 examines whether official gaming-firm update-related communications are associated with short-term Steam engagement and whether those associations vary with the progression content emphasized in the communication.

The analysis begins with a reproducible sample of official Steam communications, measures gaming-firm emphasis across four progression dimensions, and combines those communication measures with daily Steam engagement.

The final engagement analysis uses two short pre/post specifications and a daily distributed-lag model. Together, these specifications evaluate the robustness of the results to the treatment of the communication day and to the frequent occurrence of nearby communications.