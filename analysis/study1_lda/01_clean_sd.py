#!/usr/bin/env python3
"""
Data cleaning pipeline for SD (Story-Driven) games
Processes 5 games: Zelda BotW, Baldur's Gate 3, Elden Ring, Hollow Knight, RDR2
Includes LEMMA_FIX to correct NLTK quirks (e.g., 'bos' -> 'boss')
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
    print("SD ANALYSIS: DATA CLEANING PIPELINE")
    print("=" * 80)

    # Config
    INPUT_ROOTS = [os.path.join("data", "raw")]
    OUTPUT_DIR = os.path.join("data", "derived")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GLOB_PATTERNS = {
        "Legend of Zelda":        "Legend*Wild*Comments*Analysis.parquet",
        "Baldur's Gate 3":        "Baldur*Gate*3*Comments*Analysis.parquet",
        "Elden Ring":             "Elden*Ring*Comments*Analysis.parquet",
        "Hollow Knight":          "Hollow*Knight*Comments*Analysis.parquet",
        "Red Dead Redemption 2":  "Red*Dead*Redemption*2*Comments*Analysis.parquet",
    }

    BIGRAM_MIN_COUNT = 5
    PHRASE_THRESHOLD = 8.0
    MIN_TOKENS_ROW = 5
    NO_BELOW = 5
    NO_ABOVE = 0.5
    KEEP_N = 100_000
    RANDOM_STATE = 11

    # Stopwords - Load from centralized src/stopwords.py module
    # SD-specific additions for story-driven games
    STOP_WORDS = get_stopwords(include_franchise=False)
    SD_SPECIFIC_STOP = {
        'episode', 'playthrough', 'walkthrough', 'though', 'stuff', 'everything',
        'area', 'found', 'merg', 'wouldnt', 'wouldve', 'youve', 'youll', 'wasnt',
        'aint', 'couldnt', 'seems', 'happens', 'happened', 'taking', 'honestly',
        'definitely', 'either', 'looking', 'looked', 'open', 'add', 'full',
        'mine', 'kept', 'tried', 'gave', 'damn', 'using', 'done', 'jack',
        'theradbrad', 'gamegrumps'
    }
    STOP_WORDS.update(SD_SPECIFIC_STOP)

    # Lemma fixes for NLTK quirks
    LEMMA_FIX = {
        "bos": "boss",
        # add more if spotted later
    }

    # Helpers
    URL_RE = re.compile(r"(?:\@|http?\://|https?\://|www)\S+")
    HTML_RE = re.compile(r"<.*?>")
    PUNC_RE = re.compile(r"[^\w\s]")
    DIGIT_RE = re.compile(r"\d+")
    WS_RE = re.compile(r"\s+")
    LEMM = WordNetLemmatizer()

    def _wn_pos(tag):
        if not tag: return wn.NOUN
        t = tag[0]
        return wn.ADJ if t == 'J' else wn.VERB if t == 'V' else wn.NOUN if t == 'N' else wn.ADV if t == 'R' else wn.NOUN

    def normalize(text):
        text = text.lower()
        text = URL_RE.sub(" ", text)
        text = HTML_RE.sub(" ", text)
        text = PUNC_RE.sub(" ", text)
        text = DIGIT_RE.sub(" ", text)
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
            df = df.dropna(subset=['author','text']).copy()
            df["game"] = label
            all_dfs.append(df)
            print(f"  [OK] {label}: {len(df):,} comments from {os.path.basename(fpath)}")

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
    combined["tokens_raw"] = combined["tokens_raw"].apply(lambda t: [w for w in t if len(w) > 2])

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

    # Apply lemma fixes and filter stopwords
    print("\n[APPLYING] Applying lemma fixes and filtering stopwords...")
    def filter_and_fix_tokens(tokens):
        # Apply lemma fixes first
        tokens = [LEMMA_FIX.get(w, w) for w in tokens]
        # Filter stopwords and short tokens
        return [w for w in tokens if w not in STOP_WORDS and len(w) > 2]

    combined["tokens"] = combined["tokens_phrases"].apply(filter_and_fix_tokens)

    # Filter short documents
    print("\n[FILTERING] Filtering short documents...")
    sd = combined[combined["tokens"].str.len() >= MIN_TOKENS_ROW].copy().reset_index(drop=True)
    print(f"[OK] After filtering: {len(sd):,} documents ({100*len(sd)/len(combined):.1f}% retained)")

    # Build dictionary
    print("\n[INFO] Building gensim dictionary...")
    dictionary = Dictionary(sd["tokens"])
    print(f"  Original vocab size: {len(dictionary)}")

    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    print(f"  After pruning: {len(dictionary)}")

    # Convert to corpus
    print("\n[CONVERTING] Converting to bag-of-words corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in sd["tokens"]]

    # Train/test split (stratified by game)
    print("\n[SPLITTING] Creating stratified train/test split (90/10)...")
    random.seed(RANDOM_STATE)

    game_indices = {}
    for game in sd["game"].unique():
        game_indices[game] = sd[sd["game"] == game].index.tolist()

    test_idx = []
    for game, indices in game_indices.items():
        n_test = max(1, int(0.1 * len(indices)))
        test_idx.extend(random.sample(indices, n_test))

    train_idx = [i for i in range(len(sd)) if i not in test_idx]

    print(f"  Train: {len(train_idx):,}")
    print(f"  Test:  {len(test_idx):,}")

    # Top tokens peek
    print("\n[INFO] Top 30 tokens after cleaning:")
    all_tokens = [w for toks in sd["tokens"] for w in toks]
    top_tokens = Counter(all_tokens).most_common(30)
    for word, count in top_tokens:
        print(f"  {word}: {count:,}")

    # Save outputs
    print("\n[SAVING] Saving outputs...")

    sd_out = os.path.join(OUTPUT_DIR, "Filtered_Combined_SD_Cleaned.parquet")
    sd.to_parquet(sd_out, index=False)
    print(f"  [OK] Cleaned data -> {sd_out}")

    dict_out = os.path.join(OUTPUT_DIR, "lda_dictionary_SD.dict")
    dictionary.save(dict_out)
    print(f"  [OK] Dictionary -> {dict_out}")

    split_out = os.path.join(OUTPUT_DIR, "lda_split_SD_stratified.json")
    with open(split_out, "w") as f:
        json.dump({"random_state": RANDOM_STATE, "train_idx": train_idx, "test_idx": test_idx}, f)
    print(f"  [OK] Train/test split -> {split_out}")

    # Summary stats
    print("\n" + "=" * 80)
    print("CLEANING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total documents: {len(sd):,}")
    print(f"Vocabulary size: {len(dictionary):,}")
    print(f"Train/test split: {len(train_idx):,} / {len(test_idx):,}")
    print("\nDocuments per game:")
    for game in sorted(sd["game"].unique()):
        count = (sd["game"] == game).sum()
        print(f"  {game}: {count:,}")


if __name__ == '__main__':
    main()
