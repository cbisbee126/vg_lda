#!/usr/bin/env python3
"""
Generate Formatted Topic Table - All Games Analysis

Creates publication-ready topic tables from the best LDA model.
Outputs markdown, CSV, and optionally LaTeX formats.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gensim.models import LdaMulticore
from src.topic_formatting import (
    extract_topic_keywords,
    generate_markdown_table,
    generate_csv_table,
    generate_latex_table,
    auto_generate_labels
)

# ============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

# Model input
MODEL_PATH = "data/derived/best_lda_model_AllGames_k5.model"

# Table parameters
NUM_KEYWORDS = 10           # Number of top keywords to display per topic
DECIMAL_PLACES = 4          # Number of decimal places for probabilities (e.g., 0.0076)
AUTO_LABELS = False         # Set to True to auto-generate labels from top words
NUM_LABEL_WORDS = 3         # If AUTO_LABELS=True, how many words to use for label

# Manual labels (used if AUTO_LABELS=False)
# Edit these to provide interpretive labels for each topic
MANUAL_LABELS = [
    "Combat & progression mechanics",
    "Story characters / RDR2 narrative",
    "Action combat / movement",
    "Audio / music / media",
    "Creator engagement / community",
    "Meta / cosmetics / platforms",
    "Competitive mechanics / teamplay",
    "Ranked competitive / toxicity",
    "Appreciation / viewer positivity",
    "Shooter tactics / combat"
]

# Output files
OUTPUT_MARKDOWN = "output/tables/topic_table_AllGames_k5.md"
OUTPUT_CSV = "output/tables/topic_table_AllGames_k5.csv"
OUTPUT_LATEX = None  # Set to path string to generate LaTeX, or None to skip

# Display options
PRINT_TO_CONSOLE = True     # Print markdown table to console
VERBOSE = True              # Print progress messages

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    if VERBOSE:
        print(f"Loading model from: {MODEL_PATH}")

    # Load model
    model = LdaMulticore.load(MODEL_PATH)

    if VERBOSE:
        print(f"Model loaded: {model.num_topics} topics")

    # Extract keywords
    topics = extract_topic_keywords(model, num_keywords=NUM_KEYWORDS)

    # Generate or use labels
    if AUTO_LABELS:
        labels = auto_generate_labels(topics, num_label_words=NUM_LABEL_WORDS)
        if VERBOSE:
            print(f"Auto-generated labels from top {NUM_LABEL_WORDS} keywords")
    else:
        labels = MANUAL_LABELS
        if VERBOSE:
            print(f"Using {len(labels)} manual labels")

    # Generate markdown table
    markdown_table = generate_markdown_table(topics, labels=labels, num_keywords=NUM_KEYWORDS)

    if PRINT_TO_CONSOLE:
        print("\n" + "="*80)
        print("TOPIC TABLE - ALL GAMES (K=10)")
        print("="*80 + "\n")
        print(markdown_table)
        print("\n")

    # Save markdown
    if OUTPUT_MARKDOWN:
        with open(OUTPUT_MARKDOWN, 'w', encoding='utf-8') as f:
            f.write(f"# Topic Table - All Games Combined (K={model.num_topics})\n\n")
            f.write(markdown_table)
            f.write(f"\n\n**Model**: `{MODEL_PATH}`\n")
            f.write(f"**Keywords per topic**: {NUM_KEYWORDS}\n")
        if VERBOSE:
            print(f"[OK] Saved markdown table to: {OUTPUT_MARKDOWN}")

    # Save CSV
    if OUTPUT_CSV:
        generate_csv_table(topics, OUTPUT_CSV, labels=labels, num_keywords=NUM_KEYWORDS)
        if VERBOSE:
            print(f"[OK] Saved CSV table to: {OUTPUT_CSV}")

    # Save LaTeX (optional)
    if OUTPUT_LATEX:
        latex_table = generate_latex_table(topics, labels=labels, num_keywords=NUM_KEYWORDS)
        with open(OUTPUT_LATEX, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        if VERBOSE:
            print(f"[OK] Saved LaTeX table to: {OUTPUT_LATEX}")

    if VERBOSE:
        print("\n[OK] Topic table generation complete!")
