#!/usr/bin/env python3
# Cell 1: POS-aware Cleaning Pipeline (All Games)
import sys
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import os, re, glob, json, pickle, random
from collections import Counter, defaultdict
import pandas as pd
import nltk

# ----- Ensure NLTK resources -----
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
from src.stopwords import get_stopwords

# ----- Config -----
INPUT_ROOTS = [
    os.path.join("data", "raw")
]
OUT_DIR = os.path.join("data", "derived")
os.makedirs(OUT_DIR, exist_ok=True)

GLOB_PATTERNS = {
    "Legend of Zelda":        "Legend*Wild*Comments*Analysis.parquet",
    "Fortnite_Ninja":         "Fortnite*Ninja*Comments*Analysis.parquet",
    "Fortnite_SypherPK":      "Fortnite*Sypher*Comments*Analysis.parquet",
    "Fortnite_NickEh30":      "Fortnite*Nick*Eh*30*Comments*Analysis.parquet",
    "Apex Legends":           "Apex*Legends*Comments*Analysis.parquet",
    "Baldur's Gate 3":        "Baldur*Gate*3*Comments*Analysis.parquet",
    "Rocket League":          "Rocket*League*Comments*Analysis.parquet",
    "Elden Ring":             "Elden*Ring*Comments*Analysis.parquet",
    "Hollow Knight":          "Hollow*Knight*Comments*Analysis.parquet",
    "Red Dead Redemption 2":  "Red*Dead*Redemption*2*Comments*Analysis.parquet",
    "DOTA 2":                 "DOTA*2*Comments*Analysis.parquet",
    "Valorant":               "Valorant*Comments*Analysis.parquet",
}

BIGRAM_MIN_COUNT = 5
PHRASE_THRESHOLD = 8.0
MIN_TOKENS_ROW = 5
NO_BELOW = 5
NO_ABOVE = 0.50
KEEP_N   = 100_000
KEEP_FRANCHISE_TOKENS = False

# ----- Stopwords -----
# Load from centralized src/stopwords.py module
STOP_WORDS = get_stopwords(include_franchise=not KEEP_FRANCHISE_TOKENS)

# ----- Regex/helpers -----
URL_RE   = re.compile(r"(?:\@|http?\://|https?\://|www)\S+")
HTML_RE  = re.compile(r"<.*?>")
PUNC_RE  = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d+")
WS_RE    = re.compile(r"\s+")
LEMM     = WordNetLemmatizer()

LEMMA_FIX = {}
BAD_PHRASE = re.compile(r'(^[a-z]_t$|^t_[a-z]$|^[a-z]_[a-z]$)')

def _wn_pos(tag: str):
    if not tag: return wn.NOUN
    t = tag[0]
    return wn.ADJ if t == 'J' else wn.VERB if t == 'V' else wn.NOUN if t == 'N' else wn.ADV if t == 'R' else wn.NOUN

def normalize(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = PUNC_RE.sub(" ", text)
    text = DIGIT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

def tokenize_simple(text: str):
    return text.split()

def pos_lemmatize(tokens):
    if not tokens:
        return []
    tagged = list(pos_tag_sents([tokens]))[0]
    return [LEMM.lemmatize(w, _wn_pos(tag)) for (w, tag) in tagged]

def resolve_path(pattern, roots):
    for root in roots:
        matches = glob.glob(os.path.join(root, pattern))
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    return None

# Load data
raw_dfs, missing = [], []
for label, pat in GLOB_PATTERNS.items():
    fpath = resolve_path(pat, INPUT_ROOTS)
    if not fpath:
        print(f"[WARNING] No match for {label} with pattern {pat}")
        missing.append(label)
        continue

    df = pd.read_parquet(fpath)
    if not {'author','text'}.issubset(df.columns):
        print(f"[WARNING] Required columns missing in {os.path.basename(fpath)}")
        continue

    df = df.dropna(subset=['author','text']).copy()
    df['__norm'] = df['text'].map(lambda t: normalize(t) if isinstance(t,str) else "")
    df['__raw_tokens'] = df['__norm'].map(tokenize_simple)
    df['raw_tokens'] = df['__raw_tokens'].map(pos_lemmatize)
    df['game'] = label
    raw_dfs.append(df[['author','text','raw_tokens','game']])
    print(f"[OK] {label}: {len(df)} rows")

if not raw_dfs:
    raise SystemExit("No valid inputs loaded.")

combined = pd.concat(raw_dfs, ignore_index=True)
print(f"[INFO] Total: {len(combined)} comments across {len(GLOB_PATTERNS)} sources")

# Train phrases
from gensim.models import Phrases
from gensim.models.phrases import Phraser

bigram  = Phrases(combined['raw_tokens'], min_count=BIGRAM_MIN_COUNT, threshold=PHRASE_THRESHOLD)
trigram = Phrases(bigram[combined['raw_tokens']], threshold=PHRASE_THRESHOLD)
bigram_phraser  = Phraser(bigram)
trigram_phraser = Phraser(trigram)

def apply_phrases_then_filter(toks):
    phr = trigram_phraser[bigram_phraser[toks]]
    phr = [w for w in phr if not BAD_PHRASE.match(w)]
    phr = [LEMMA_FIX.get(w, w) for w in phr]
    return [w for w in phr if w not in STOP_WORDS and len(w) > 2]

combined['tokens'] = combined['raw_tokens'].apply(apply_phrases_then_filter)

# Filter short docs
initial = len(combined)
combined = combined[combined['tokens'].str.len() >= MIN_TOKENS_ROW]
print(f"[OK] Removed {initial - len(combined)} short comments (<{MIN_TOKENS_ROW} tokens)")

# Token peek
all_tokens = [w for toks in combined['tokens'] for w in toks]
print("[INFO] Top 30 tokens:", Counter(all_tokens).most_common(30))

# Save cleaned
clean_path = os.path.join(OUT_DIR, "Filtered_Combined_AllGames_Cleaned.parquet")
combined.to_parquet(clean_path, index=False)
print(f"[SAVED] Saved cleaned data -> {clean_path}")

# Dictionary & corpus
from gensim.corpora import Dictionary
dictionary = Dictionary(combined['tokens'])
dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
corpus = [dictionary.doc2bow(t) for t in combined['tokens']]
print(f"[INFO] Dictionary: {len(dictionary)} tokens | Corpus: {len(corpus)} docs")

dict_path = os.path.join(OUT_DIR, "lda_dictionary_AllGames.dict")
dictionary.save(dict_path)
print(f"[SAVED] Saved dictionary -> {dict_path}")

# Stratified split
rng_state = 11
by_game = defaultdict(list)
for i, g in enumerate(combined['game']):
    by_game[g].append(i)

hold_idx = set()
for g, idxs in by_game.items():
    r = random.Random(rng_state)
    r.shuffle(idxs)
    k = max(1, int(0.10 * len(idxs)))
    hold_idx.update(idxs[:k])

train_idx = [i for i in range(len(combined)) if i not in hold_idx]
test_idx  = [i for i in range(len(combined)) if i in hold_idx]

split_json = os.path.join(OUT_DIR, "lda_split_AllGames_stratified.json")
with open(split_json, "w") as f:
    json.dump({"random_state": rng_state, "train_idx": train_idx, "test_idx": test_idx}, f)

print(f"[TEST] Split: Train={len(train_idx)}, Test={len(test_idx)}")
print("[OK] Cell 1 complete")
