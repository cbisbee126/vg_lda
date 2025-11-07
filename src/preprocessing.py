"""
Text Preprocessing Utilities

Functions for cleaning, normalizing, and preparing text data for LDA modeling.

Completed:
- Stopwords extracted to src/stopwords.py (centralized)

Future refactoring targets:
- normalize_text() - URL/HTML removal, punctuation cleaning
- lemmatize_tokens() - POS-aware lemmatization wrapper
- build_phrase_models() - Bigram/trigram detection
- filter_stopwords() - Token filtering with stopword removal
- build_dictionary() - Gensim dictionary creation with pruning
"""

# TODO: Extract common preprocessing functions from cleaning scripts
# Currently, each cleaning script has duplicate code for:
# - Text normalization (lines ~95-118 in each script)
# - POS tagging and lemmatization (lines ~123-128)
# - Phrase model training (lines ~169-178)
# - Stopword filtering (lines ~174-178)
# - Dictionary building (lines ~190-198)
