# Topic Table Generation Scripts

These scripts generate topic tables from best LDA models.

## Scripts

1. **`generate_topic_table_all_games.py`** - All Games Combined (K=10)
2. **`generate_topic_table_fp.py`** - FP Competitive Games (K=5)
3. **`generate_topic_table_sd.py`** - SD Story-Driven Games (K=2)

## Usage

```bash
# From project root
python scripts/analysis/generate_topic_table_all_games.py
python scripts/analysis/generate_topic_table_fp.py
python scripts/analysis/generate_topic_table_sd.py
```

## Configuration Parameters

Each script has a **CONFIGURATION** section at the top for editing parameters:

### Key Parameters to Customize

```python
# Model input
MODEL_PATH = "data/final/best_lda_model_AllGames_k10.model"  # Path to best model

# Table parameters
NUM_KEYWORDS = 10           # How many keywords to display per topic (try 10, 15, 20)
DECIMAL_PLACES = 4          # Decimal places for probabilities (0.0076 vs 0.008)
AUTO_LABELS = False         # True: auto-generate labels, False: use MANUAL_LABELS
NUM_LABEL_WORDS = 3         # If AUTO_LABELS=True, words per label

# Manual labels - edit these!
MANUAL_LABELS = [
    "Combat & progression mechanics",
    "Story characters / RDR2 narrative",
    # ... one label per topic
]

# Output files
OUTPUT_MARKDOWN = "data/final/topic_table_AllGames_k10.md"  # Markdown table
OUTPUT_CSV = "data/final/topic_table_AllGames_k10.csv"       # CSV table
OUTPUT_LATEX = None  # Set to path to generate LaTeX, or None to skip

# Display options
PRINT_TO_CONSOLE = True     # Print table to console
VERBOSE = True              # Show progress messages
```

## Output Formats

### 1. Markdown Table
```markdown
| Topic # | Top 10 Keywords (with weights) | Preliminary Label |
|---------|--------------------------------|-------------------|
| 0 | attack (0.0076), enemy (0.0060), ... | Combat & progression |
```

### 2. CSV Table
```csv
topic_id,top_10_keywords,preliminary_label
0,"attack (0.0076), enemy (0.0060), ...",Combat & progression
```

### 3. LaTeX Table (optional)
```latex
\begin{table}[h]
\centering
\begin{tabular}{|l|p{8cm}|p{3cm}|}
\hline
Topic \# & Top 10 Keywords (with weights) & Preliminary Label \\
...
```

## Common Customizations

### Change Number of Keywords
```python
NUM_KEYWORDS = 20  # Show top 20 instead of top 10
```

### Change Output Location
```python
OUTPUT_MARKDOWN = "results/topic_tables/all_games_k10.md"
OUTPUT_CSV = "results/topic_tables/all_games_k10.csv"
```

### Auto-Generate Labels
```python
AUTO_LABELS = True          # Use top keywords as label
NUM_LABEL_WORDS = 4         # e.g., "attack / enemy / fight / hard"
```

### Export LaTeX for Papers
```python
OUTPUT_LATEX = "data/final/topic_table_AllGames_k10.tex"
```

### Remove Labels Column
```python
# In the script, set labels to None:
labels = None  # Or we can commentß out the MANUAL_LABELS assignment
```

### Adjust Probability Precision
```python
DECIMAL_PLACES = 3  # 0.008 instead of 0.0076
DECIMAL_PLACES = 5  # 0.00756 for more precision
```

## Example Workflow

After running a new K-sweep and selecting the best K:

1. **Update MODEL_PATH** to point to new best model
2. **Edit MANUAL_LABELS** to provide interpretive labels for each topic
3. **Adjust NUM_KEYWORDS** for more/fewer keywords
4. **Run the script**:
   ```bash
   python scripts/analysis/generate_topic_table_all_games.py
   ```
5. **Check outputs** in `data/final/`

## Dependencies

Requires:
- `gensim` (for loading LDA models)
- `src/topic_formatting.py` (utility functions)

All scripts use the same reusable functions from `src/topic_formatting.py`, so any improvements to formatting benefit all three analyses.
