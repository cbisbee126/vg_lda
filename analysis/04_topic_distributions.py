#!/usr/bin/env python3
# Generate topic distributions and additional visualizations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

def main():
    # Paths
    OUT_DIR = os.path.join("data", "derived")
    TABLES_DIR = os.path.join("output", "tables")
    FIGURES_DIR = os.path.join("output", "figures")

    INPUT_FILE = os.path.join(OUT_DIR, "Filtered_Combined_AllGames_Cleaned.parquet")
    DICT_PATH = os.path.join(OUT_DIR, "lda_dictionary_AllGames.dict")
    MODEL_PATH = os.path.join(OUT_DIR, "best_lda_model_AllGames_k7.model")

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    print("[LOADING] Loading data, dictionary, and model...")
    df = pd.read_parquet(INPUT_FILE)
    dictionary = Dictionary.load(DICT_PATH)
    model = LdaMulticore.load(MODEL_PATH)

    texts = df["tokens"].tolist()
    games = df["game"].tolist()
    corpus = [dictionary.doc2bow(t) for t in texts]

    print(f"[OK] Loaded {len(corpus)} documents, {len(dictionary)} vocab terms, K={model.num_topics} model")

    # ========================================================================
    # 1. Generate per-document topic distributions
    # ========================================================================
    print("\n[INFO] Computing per-document topic distributions...")
    doc_topics = []
    for i, bow in enumerate(corpus):
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
        topic_probs = [prob for _, prob in sorted(topic_dist)]

        # Get dominant topic
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else -1
        dominant_prob = max(topic_dist, key=lambda x: x[1])[1] if topic_dist else 0.0

        doc_topics.append({
            'doc_id': i,
            'game': games[i],
            'dominant_topic': dominant_topic,
            'dominant_prob': dominant_prob,
            **{f'topic_{t}': topic_probs[t] for t in range(model.num_topics)}
        })

    doc_topics_df = pd.DataFrame(doc_topics)
    doc_topics_path = os.path.join(TABLES_DIR, "document_topic_distributions.csv")
    doc_topics_df.to_csv(doc_topics_path, index=False)
    print(f"[SAVED] Saved per-document distributions -> {doc_topics_path}")

    # ========================================================================
    # 2. Generate per-game topic prevalence
    # ========================================================================
    print("\n[INFO] Computing per-game topic prevalence...")

    # Method 1: Average topic probabilities per game
    game_topic_avg = doc_topics_df.groupby('game')[[f'topic_{t}' for t in range(model.num_topics)]].mean()
    game_topic_avg_path = os.path.join(TABLES_DIR, "game_topic_prevalence_avg.csv")
    game_topic_avg.to_csv(game_topic_avg_path)
    print(f"[SAVED] Saved per-game topic averages -> {game_topic_avg_path}")

    # Method 2: Dominant topic counts per game
    topic_game_counts = defaultdict(lambda: Counter())
    for _, row in doc_topics_df.iterrows():
        topic_game_counts[row['dominant_topic']][row['game']] += 1

    game_dominant_rows = []
    for topic in range(model.num_topics):
        total = sum(topic_game_counts[topic].values())
        for game, count in topic_game_counts[topic].items():
            game_dominant_rows.append({
                'topic': topic,
                'game': game,
                'doc_count': count,
                'share_in_topic': count / total if total > 0 else 0.0
            })

    game_dominant_df = pd.DataFrame(game_dominant_rows)
    game_dominant_path = os.path.join(TABLES_DIR, "game_topic_prevalence_dominant.csv")
    game_dominant_df.to_csv(game_dominant_path, index=False)
    print(f"[SAVED] Saved per-game dominant topic counts -> {game_dominant_path}")

    # ========================================================================
    # 3. Create formatted topic summary table
    # ========================================================================
    print("\n[INFO] Creating formatted topic summary...")

    topic_summaries = []
    for t in range(model.num_topics):
        top_words = [w for w, _ in model.show_topic(t, topn=10)]
        doc_count = (doc_topics_df['dominant_topic'] == t).sum()
        doc_share = doc_count / len(doc_topics_df)

        topic_summaries.append({
            'topic': t,
            'top_10_words': ', '.join(top_words),
            'doc_count': doc_count,
            'doc_share': f"{doc_share:.2%}"
        })

    topic_summary_df = pd.DataFrame(topic_summaries)
    summary_path = os.path.join(TABLES_DIR, "topic_summary.csv")
    topic_summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Saved topic summary -> {summary_path}")

    # ========================================================================
    # 4. Visualizations
    # ========================================================================
    print("\n[INFO] Generating visualizations...")

    # Heatmap: Per-game topic prevalence (average probabilities)
    plt.figure(figsize=(12, 8))
    sns.heatmap(game_topic_avg.T, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=game_topic_avg.index,
                yticklabels=[f'Topic {t}' for t in range(model.num_topics)])
    plt.title('Topic Prevalence by Game (Average Probability)')
    plt.xlabel('Game')
    plt.ylabel('Topic')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, "game_topic_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[SAVED] Saved heatmap -> {heatmap_path}")

    # Bar chart: Document counts per topic
    plt.figure(figsize=(10, 6))
    topic_counts = doc_topics_df['dominant_topic'].value_counts().sort_index()
    plt.bar(range(model.num_topics), [topic_counts.get(t, 0) for t in range(model.num_topics)],
            color='steelblue', edgecolor='black')
    plt.xlabel('Topic')
    plt.ylabel('Document Count')
    plt.title('Document Distribution Across Topics (Dominant Topic Assignment)')
    plt.xticks(range(model.num_topics))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "topic_distribution_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[SAVED] Saved bar chart -> {bar_path}")

    # Stacked bar chart: Per-game topic distribution
    game_topic_pivot = doc_topics_df.groupby('game')['dominant_topic'].value_counts().unstack(fill_value=0)
    game_topic_pivot_pct = game_topic_pivot.div(game_topic_pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 8))
    game_topic_pivot_pct.plot(kind='bar', stacked=True, ax=ax,
                              colormap='tab10', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Game')
    ax.set_ylabel('Proportion of Documents')
    ax.set_title('Topic Distribution by Game (Stacked Proportions)')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    stacked_path = os.path.join(FIGURES_DIR, "game_topic_stacked.png")
    plt.savefig(stacked_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Saved stacked bar chart -> {stacked_path}")

    print("\n[OK] All topic distributions and visualizations generated!")

if __name__ == '__main__':
    main()
