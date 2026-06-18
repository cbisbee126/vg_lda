#!/usr/bin/env python3
"""
Data cleaning pipeline for FP (First-Person/Competitive) games
Processes 5 games: Fortnite (3 creators), Apex Legends, Rocket League, DOTA 2, Valorant
"""
import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os, re, glob, json, random
from collections import Counter
import pandas as pd
import nltk

# Ensure NLTK resources
for pkg in ("stopwords", "punkt", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(pkg)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    try:
        nltk.download("averaged_perceptron_tagger_eng")
    except:
        nltk.download("averaged_perceptron_tagger")

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag_sents
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from src.stopwords import get_stopwords


def main():
    print("=" * 80)
    print("FP ANALYSIS: DATA CLEANING PIPELINE")
    print("=" * 80)

    # Config
    INPUT_ROOTS = [os.path.join("data", "raw")]
    OUTPUT_DIR = os.path.join("data", "derived")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GLOB_PATTERNS = {
        "Fortnite_Ninja":    "Fortnite*Ninja*Comments*Analysis.parquet",
        "Fortnite_SypherPK": "Fortnite*Sypher*Comments*Analysis.parquet",
        "Fortnite_NickEh30": "Fortnite*Nick*Eh*30*Comments*Analysis.parquet",
        "Apex Legends":      "Apex*Legends*Comments*Analysis.parquet",
        "Rocket League":     "Rocket*League*Comments*Analysis.parquet",
        "DOTA 2":            "DOTA*2*Comments*Analysis.parquet",
        "Valorant":          "Valorant*Comments*Analysis.parquet",
    }

    BIGRAM_MIN_COUNT = 10
    PHRASE_THRESHOLD = 8.0
    MIN_TOKENS_ROW = 5
    NO_BELOW = 5
    NO_ABOVE = 0.50
    KEEP_N = 100_000
    KEEP_FRANCHISE_TOKENS = False
    RANDOM_STATE = 42

    # Stopwords - Load from centralized src/stopwords.py module
    STOP_WORDS = get_stopwords(include_franchise=not KEEP_FRANCHISE_TOKENS)

    # Helpers
    URL_RE = re.compile(r"(?:\@|http?\://|https?\://|www)\S+")
    HTML_RE = re.compile(r"<.*?>")
    PUNC_RE = re.compile(r"[^\w\s]")
    DIGIT_RE = re.compile(r"\d+")
    WS_RE = re.compile(r"\s+")
    LEMM = WordNetLemmatizer()
    BAD_PHRASE = re.compile(r'^[a-z]_[a-z]$')

    def _wn_pos(tag):
        if not tag: return wn.NOUN
        t = tag[0]
        return wn.ADJ if t == 'J' else wn.VERB if t == 'V' else wn.NOUN if t == 'N' else wn.ADV if t == 'R' else wn.NOUN

    def normalize(text):
        text = URL_RE.sub("", text)
        text = HTML_RE.sub("", text)
        text = text.lower()
        text = PUNC_RE.sub(" ", text)
        text = DIGIT_RE.sub("", text)
        text = WS_RE.sub(" ", text).strip()
        return text

    # Load data
    print("\n[LOADING] Loading raw data from parquet files...")
    all_dfs = []
    for label, pattern in GLOB_PATTERNS.items():
        found = []
        for root in INPUT_ROOTS:
            found += glob.glob(os.path.join(root, pattern))
        if not found:
            print(f"[WARNING] No files found for {label} (pattern: {pattern})")
            continue
        for fpath in found:
            df = pd.read_parquet(fpath)
            df["game"] = label
            all_dfs.append(df)
            print(f"  [OK] {label}: {len(df):,} comments from {fpath}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[OK] Total raw comments: {len(combined):,}")

    # Normalize
    print("\n[CLEANING] Normalizing text...")
    combined["text_clean"] = combined["text"].apply(normalize)
    combined = combined[combined["text_clean"].str.len() > 0].reset_index(drop=True)
    print(f"[OK] After normalization: {len(combined):,}")

    # Tokenize
    print("\n[TOKENIZING] Tokenizing...")
    combined["tokens_raw"] = combined["text_clean"].str.split()

    # POS-aware lemmatization
    print("\n[TAGGING] POS-tagging & lemmatization...")
    batch_size = 10_000
    num_batches = (len(combined) + batch_size - 1) // batch_size
    all_lemmas = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(combined))
        batch_tokens = combined["tokens_raw"].iloc[start:end].tolist()

        tagged = pos_tag_sents(batch_tokens)
        batch_lemmas = []
        for sent_tags in tagged:
            lemmas = [LEMM.lemmatize(w, _wn_pos(tag)) for w, tag in sent_tags]
            batch_lemmas.append(lemmas)

        all_lemmas.extend(batch_lemmas)
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Processed batch {i+1}/{num_batches}")

    combined["tokens_lemma"] = all_lemmas

    # Bigram/trigram detection
    print("\n[BUILDING] Building phrase models (bigrams/trigrams)...")
    bigram_model = Phrases(
        combined["tokens_lemma"],
        min_count=BIGRAM_MIN_COUNT,
        threshold=PHRASE_THRESHOLD,
        delimiter="_"
    )
    bigram = Phraser(bigram_model)

    trigram_model = Phrases(
        bigram[combined["tokens_lemma"]],
        min_count=BIGRAM_MIN_COUNT,
        threshold=PHRASE_THRESHOLD,
        delimiter="_"
    )
    trigram = Phraser(trigram_model)

    combined["tokens_phrases"] = combined["tokens_lemma"].apply(lambda t: trigram[bigram[t]])

    # Filter stopwords & bad phrases
    print("\n[FILTERING] Filtering stopwords...")
    def filter_tokens(tokens):
        return [
            w for w in tokens
            if w not in STOP_WORDS and len(w) > 2 and not BAD_PHRASE.match(w)
        ]

    combined["tokens"] = combined["tokens_phrases"].apply(filter_tokens)

    # Filter short documents
    print("\n[FILTERING] Filtering short documents...")
    fp = combined[combined["tokens"].str.len() >= MIN_TOKENS_ROW].copy().reset_index(drop=True)
    print(f"[OK] After filtering: {len(fp):,} documents ({100*len(fp)/len(combined):.1f}% retained)")

    # Build dictionary
    print("\n[INFO] Building gensim dictionary...")
    dictionary = Dictionary(fp["tokens"])
    print(f"  Original vocab size: {len(dictionary)}")

    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    print(f"  After pruning: {len(dictionary)}")

    # Convert to corpus
    print("\n[CONVERTING] Converting to bag-of-words corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in fp["tokens"]]

    # Train/test split (stratified by game)
    print("\n[SPLITTING] Creating stratified train/test split (90/10)...")
    random.seed(RANDOM_STATE)

    game_indices = {}
    for game in fp["game"].unique():
        game_indices[game] = fp[fp["game"] == game].index.tolist()

    test_idx = []
    for game, indices in game_indices.items():
        n_test = max(1, int(0.1 * len(indices)))
        test_idx.extend(random.sample(indices, n_test))

    train_idx = [i for i in range(len(fp)) if i not in test_idx]

    print(f"  Train: {len(train_idx):,}")
    print(f"  Test:  {len(test_idx):,}")

    # Save outputs
    print("\n[SAVING] Saving outputs...")

    fp_out = os.path.join(OUTPUT_DIR, "Filtered_Combined_FP_Cleaned.parquet")
    fp.to_parquet(fp_out, index=False)
    print(f"  [OK] Cleaned data -> {fp_out}")

    dict_out = os.path.join(OUTPUT_DIR, "lda_dictionary_FP.dict")
    dictionary.save(dict_out)
    print(f"  [OK] Dictionary -> {dict_out}")

    split_out = os.path.join(OUTPUT_DIR, "lda_split_FP_stratified.json")
    with open(split_out, "w") as f:
        json.dump({"train_idx": train_idx, "test_idx": test_idx}, f)
    print(f"  [OK] Train/test split -> {split_out}")

    # Summary stats
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(fp):,}")
    print(f"Vocabulary size: {len(dictionary):,}")
    print(f"Train/test split: {len(train_idx):,} / {len(test_idx):,}")
    print("\nDocuments per game:")
    for game in sorted(fp["game"].unique()):
        count = (fp["game"] == game).sum()
        print(f"  {game}: {count:,}")


if __name__ == '__main__':
    main()
