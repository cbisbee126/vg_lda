# Scripts Directory

Organized Python scripts for LDA analysis pipeline.

## Directory Structure

### cleaning/
Data preprocessing scripts that create cleaned corpora, dictionaries, and train/test splits.

- `run_all_games_cleaning.py` - All Games Combined
- `run_fp_cleaning.py` - First-Person/Competitive games
- `run_sd_cleaning.py` - Story-Driven games

We need to run these first to prepare data for K-selection.

### k_selection/
K-sweep scripts that train models across multiple K values and identify optimal topic count.
Increments might be edited and changed if need be.

- `run_all_games_k_sweep.py` - All Games (K=5-100)
- `run_fp_k_sweep.py` - FP games (K=2-10)
- `run_sd_k_sweep.py` - SD games (K=2-10)

Records coherence and log-perplexity metrics, saves best model.

### analysis/
Post-modeling analysis and visualization scripts.

- `generate_topic_distributions.py` - Create per-document and per-game topic distributions

### legacy/
Old script versions from Phase 1 (preserved for reference).

- `run_all_games_k_sweep.py` - Original K=2-10 sweep
- `run_fp_full.py` - Monolithic FP script (combines cleaning + K-sweep)

## Usage

```bash
# Activate virtual environment
source ../venv/bin/activate

# Typical workflow for FP analysis:
python cleaning/run_fp_cleaning.py
python k_selection/run_fp_k_sweep.py
python analysis/generate_topic_distributions.py
```

