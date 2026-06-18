# analysis/study1_lda/ - Study 1 LDA pipeline (Python)

Run in order from the repo root (paths are relative to root). Each step reuses the
importable modules in `src/`. The three corpora (All Games, FP/competitive, SD/story-driven)
share the same stages.

## Pipeline

1. `01_clean_{all_games,fp,sd}.py` - normalize, POS-lemmatize, phrase-detect; build the
   gensim dictionary and stratified 90/10 split. Reads raw parquet from `data/raw/`; writes
   the cleaned corpus, dictionary, and split JSON to `data/derived/`.
2. `02_ksweep_{all_games,fp,sd}.py` - K-sweep (K=2-10), coherence (c_v) + held-out
   log-perplexity; saves the best model to `data/derived/`, metrics CSV to `output/tables/`,
   and the sweep plot to `output/figures/`.
3. `03_topic_table_{all_games,fp,sd}.py` - top-keyword topic table from a saved model;
   writes CSV/Markdown to `output/tables/`.
4. `04_topic_distributions.py` - per-document and per-game topic distributions and figures
   (All Games); CSVs to `output/tables/`, plots to `output/figures/`.

## Run

```bash
# from the repo root, with the uv .venv active
python analysis/study1_lda/01_clean_fp.py
python analysis/study1_lda/02_ksweep_fp.py
python analysis/study1_lda/03_topic_table_fp.py
```

## Notes

- Raw inputs live in Box (`data/raw/` is read-only); `data/`, `output/`, `results/` are gitignored.
- Current best K: All Games 7, FP 8, SD 3 (coherence-selected, post 2025-11-07 re-clean).
- KNOWN STALE: the `03_topic_table_*` scripts hardcode older model k-values (AllGames k5,
  FP k4) and the `02_ksweep_*` metrics/plot filenames keep an old "5_50"/"5_100" label even
  though the sweep is K=2-10. Reconcile with the model of record during the manuscript review
  (see `_logs/2026-06-17_standard-adoption.md`).
- Exploration goes in `_scratch_NN_*.py` (greppable, not pipeline).
