#!/usr/bin/env python3
"""
K-selection sweep for SD (Story-Driven) games
Tests K=5-50 in increments of 5
Sequential training with full multicore parallelization per model.
"""
import os, math, json, time
import pandas as pd
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel


def train_eval_k(k, train_corpus, test_corpus, train_texts, dictionary, config):
    """Train and evaluate a single LDA model for given K"""
    print(f"\n[TRAINING] Training K={k}...")
    start = time.time()

    model = LdaMulticore(
        corpus=train_corpus,
        id2word=dictionary,
        num_topics=k,
        passes=config['PASSES'],
        iterations=config['ITERS'],
        random_state=config['RANDOM_STATE'],
        workers=config['WORKERS'],
        chunksize=config['CHUNKSIZE'],
        eval_every=None,
        alpha='asymmetric',
        eta='auto',
    )

    c_v = CoherenceModel(
        model=model,
        texts=train_texts,
        dictionary=dictionary,
        coherence="c_v"
    ).get_coherence()

    log_perp = model.log_perplexity(test_corpus)
    elapsed = time.time() - start

    print(f"[OK] K={k} complete | c_v={c_v:.4f} | log_perp={log_perp:.4f} | {elapsed/60:.1f}min")

    return {
        'k': k,
        'c_v': c_v,
        'log_perplexity': log_perp,
        'time_seconds': elapsed,
        'model': model
    }


def main():
    # Config
    INPUT_FILE = os.path.join("data", "derived", "Filtered_Combined_SD_Cleaned.parquet")
    OUT_DIR = os.path.join("data", "derived")
    TABLES_DIR = os.path.join("output", "tables")
    FIGURES_DIR = os.path.join("output", "figures")
    DICT_PATH = os.path.join(OUT_DIR, "lda_dictionary_SD.dict")
    SPLIT_JSON = os.path.join(OUT_DIR, "lda_split_SD_stratified.json")

    # K_GRID = list(range(5, 51, 5))  # [5, 10, 15, ..., 50]
    K_GRID = list(range(2, 11, 1))  # [2-10]
    RANDOM_STATE = 42
    PASSES, ITERS = 5, 400
    CHUNKSIZE = 2000
    WORKERS = os.cpu_count() or 12

    print("=" * 80)
    print("SD ANALYSIS: K-SELECTION SWEEP (K=5-50)")
    print("=" * 80)
    print(f"\n[CONFIG] Configuration:")
    print(f"   Total cores: {WORKERS}")
    print(f"   Workers per model: {WORKERS} (sequential K training)")
    print(f"   K range: {K_GRID[0]} to {K_GRID[-1]} (step {K_GRID[1]-K_GRID[0]}, n={len(K_GRID)})")

    RESULTS_CSV = os.path.join(TABLES_DIR, "lda_k_selection_SD_metrics_5_50.csv")
    PLOT_COMBINED = os.path.join(FIGURES_DIR, "lda_k_sweep_SD_5_50.png")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    print("\n[LOADING] Loading data/dictionary...")
    df = pd.read_parquet(INPUT_FILE)
    texts = df["tokens"].tolist()

    dictionary = Dictionary.load(DICT_PATH)
    corpus = [dictionary.doc2bow(t) for t in texts]
    print(f"[OK] docs={len(corpus)}  vocab={len(dictionary)}")

    # Load split
    print(f"[LOADING] Loading split: {SPLIT_JSON}")
    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    train_corpus = [corpus[i] for i in train_idx]
    test_corpus = [corpus[i] for i in test_idx]
    train_texts = [texts[i] for i in train_idx]
    print(f"[TEST] Train: {len(train_corpus)}  Test: {len(test_corpus)}")

    # Config dict
    config = {
        'PASSES': PASSES,
        'ITERS': ITERS,
        'RANDOM_STATE': RANDOM_STATE,
        'WORKERS': WORKERS,
        'CHUNKSIZE': CHUNKSIZE
    }

    # Train models sequentially
    print(f"\n[TRAINING] Training {len(K_GRID)} models sequentially (full parallelization per model)...")
    start_total = time.time()

    results = []
    for k in K_GRID:
        result = train_eval_k(k, train_corpus, test_corpus, train_texts, dictionary, config)
        results.append(result)

    elapsed_total = time.time() - start_total
    print(f"\n[OK] All {len(K_GRID)} models trained in {elapsed_total/60:.1f} minutes")
    print(f"   Average: {elapsed_total/len(K_GRID):.1f}s per model")

    # Process results
    rows = [{'k': r['k'], 'c_v': r['c_v'], 'log_perplexity': r['log_perplexity'],
             'time_seconds': r['time_seconds']} for r in results]
    dfm = pd.DataFrame(rows).sort_values("k")
    dfm.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVED] Saved metrics -> {RESULTS_CSV}")

    # Find best model (coherence primary, log-perplexity tie-breaker)
    best = max(results, key=lambda r: (r['c_v'], r['log_perplexity']))
    best_k = best['k']
    best_model = best['model']

    print(f"\n[BEST] Best K={best_k}")
    print(f"   Coherence (c_v): {best['c_v']:.4f}")
    print(f"   Log-perplexity: {best['log_perplexity']:.4f}")

    # Save best model
    best_path = os.path.join(OUT_DIR, f"best_lda_model_SD_k{best_k}.model")
    best_model.save(best_path)
    print(f"[SAVED] Saved best model -> {best_path}")

    # Export topic terms
    def dump_topics(model, topn=20, path=None):
        rows = []
        for t in range(model.num_topics):
            for rank, (w, p) in enumerate(model.show_topic(t, topn=topn), start=1):
                rows.append({"topic": t, "rank": rank, "word": w, "prob": p})
        dt = pd.DataFrame(rows)
        if path:
            dt.to_csv(path, index=False)
        return dt

    topics_csv = os.path.join(TABLES_DIR, f"best_topics_SD_k{best_k}.csv")
    dump_topics(best_model, topn=20, path=topics_csv)
    print(f"[SAVED] Topic top-terms saved -> {topics_csv}")

    # Plot
    def plot_combined(df, best_k, out_path):
        df = df.sort_values("k")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(df["k"], df["c_v"], marker="o", label="c_v", color='blue')
        ax1.set_xlabel("K (number of topics)")
        ax1.set_ylabel("Coherence (c_v)", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df["k"], df["log_perplexity"], marker="s", linestyle="--",
                label="log_perplexity", color='red')
        ax2.set_ylabel("log_perplexity (higher is better)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax1.axvline(best_k, linestyle=":", linewidth=1.5, color='green',
                   label=f'Best K={best_k}')
        ax1.set_title(f"LDA K Sweep (SD Games, K=5-50) — Best K={best_k}")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[SAVED] Saved plot -> {out_path}")

    plot_combined(dfm, best_k, PLOT_COMBINED)
    print("\n[OK] K-selection sweep complete!")


if __name__ == '__main__':
    main()
