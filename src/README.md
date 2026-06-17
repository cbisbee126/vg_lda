# Source Library (src/)

Reusable Python modules for video game LDA analysis. Extracted from duplicate code in cleaning and analysis scripts.

## Modules

### stopwords.py
**Status**: Complete

Centralized stopword definitions for preprocessing YouTube gaming comments.

**Contents**:
- 429 total stopwords (444 with franchise tokens)
- Organized into categories:
  - NLTK English stopwords (198 words)
  - Generic chat/filler terms (162 words)
  - Platform metadata (10 words)
  - Creator names (13 words)
  - Gaming metadata (6 words)
  - Extended common words (70 words)
  - Franchise tokens (15 words, optional)

**Usage**:
```python
from src.stopwords import get_stopwords

# Exclude franchise tokens (default, for cross-game analysis)
stopwords = get_stopwords(include_franchise=False)

# Include franchise tokens (for non-game-specific topics)
stopwords = get_stopwords(include_franchise=True)

# Print statistics
from src.stopwords import print_stopword_stats
print_stopword_stats()
```

**Used by**:
- `scripts/cleaning/run_all_games_cleaning.py`
- `scripts/cleaning/run_fp_cleaning.py`
- `scripts/cleaning/run_sd_cleaning.py`

### preprocessing.py
**Status**: Placeholder (future refactoring)

Will contain common text preprocessing functions:
- `normalize_text()` - URL/HTML removal, punctuation cleaning
- `lemmatize_tokens()` - POS-aware lemmatization
- `build_phrase_models()` - Bigram/trigram detection
- `filter_stopwords()` - Token filtering
- `build_dictionary()` - Gensim dictionary creation

### modeling.py
**Status**: Placeholder (future refactoring)

Will contain LDA modeling utilities:
- `train_lda_model()` - Model training wrapper
- `evaluate_model_coherence()` - Coherence calculation
- `evaluate_model_perplexity()` - Perplexity calculation
- `select_best_k()` - K-selection logic
- `save_model_artifacts()` - Model/topic saving

### visualization.py
**Status**: Placeholder (future refactoring)

Will contain plotting functions:
- `plot_k_sweep()` - K-selection curves
- `plot_topic_distributions()` - Topic prevalence plots
- `plot_game_topic_heatmap()` - Cross-game topic heatmaps

### utils.py
**Status**: Placeholder (future refactoring)

Will contain general utilities:
- `load_corpus()` - Corpus loading helper
- `load_dictionary()` - Dictionary loading helper
- `save_results()` - Results saving
- `format_topics()` - Topic formatting


## Design Philosophy

**DRY (Don't Repeat Yourself)**: Extract common code patterns into reusable functions.

**Backward Compatibility**: Refactoring should not break existing scripts. Keep parameter names and behavior consistent.

**Documentation**: All functions should have docstrings with parameters, returns, and examples.

**Testing**: Consider adding unit tests in `tests/` directory as modules grow.

## Refactoring Checklist

When extracting functions to src/:
1. Identify duplicate code across scripts
2. Extract to appropriate module (preprocessing, modeling, viz, utils)
3. Add comprehensive docstring
4. Update all scripts to import from src/
5. Test that scripts still work
6. Update documentation (README, PREPROCESSING_SUMMARY.md)

## Future Enhancements

- Add type hints for better IDE support
- Create `tests/` directory with unit tests
- Add configuration management (`src/config.py`)
- Consider adding logging utilities (`src/logging_utils.py`)