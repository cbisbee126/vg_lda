"""
Topic Table Formatting Utilities

Functions for extracting and formatting LDA topic keywords into publication-ready tables.
"""

from gensim.models import LdaMulticore
import csv


def extract_topic_keywords(model, num_keywords=10):
    """
    Extract top keywords with probabilities for each topic from LDA model.

    Args:
        model: Trained gensim LdaMulticore model
        num_keywords: Number of top keywords to extract per topic

    Returns:
        List of dicts, one per topic:
        [
            {'topic_id': 0, 'keywords': [('word1', 0.05), ('word2', 0.03), ...]},
            ...
        ]
    """
    topics = []
    for topic_id in range(model.num_topics):
        # Get top N words with probabilities
        top_words = model.show_topic(topic_id, topn=num_keywords)
        topics.append({
            'topic_id': topic_id,
            'keywords': top_words  # List of (word, prob) tuples
        })
    return topics


def format_keywords_inline(keywords, decimal_places=4):
    """
    Format keyword-probability pairs as inline string.

    Args:
        keywords: List of (word, prob) tuples
        decimal_places: Number of decimal places for probabilities

    Returns:
        String like "word1 (0.0565), word2 (0.0530), word3 (0.0431)"
    """
    formatted = []
    for word, prob in keywords:
        formatted.append(f"{word} ({prob:.{decimal_places}f})")
    return ", ".join(formatted)


def generate_markdown_table(topics, labels=None, num_keywords=10):
    """
    Generate markdown table with topics and inline keywords.

    Args:
        topics: List of topic dicts from extract_topic_keywords()
        labels: Optional list of interpretive labels (one per topic)
        num_keywords: Number of keywords to display

    Returns:
        String containing markdown table
    """
    lines = []

    # Header
    if labels:
        lines.append("| Topic # | Top {} Keywords (with weights) | Preliminary Label |".format(num_keywords))
        lines.append("|---------|--------------------------------|-------------------|")
    else:
        lines.append("| Topic # | Top {} Keywords (with weights) |".format(num_keywords))
        lines.append("|---------|--------------------------------|")

    # Rows
    for i, topic in enumerate(topics):
        keywords_str = format_keywords_inline(topic['keywords'])
        if labels:
            label = labels[i] if i < len(labels) else "TBD"
            lines.append(f"| {topic['topic_id']} | {keywords_str} | {label} |")
        else:
            lines.append(f"| {topic['topic_id']} | {keywords_str} |")

    return "\n".join(lines)


def generate_csv_table(topics, output_path, labels=None, num_keywords=10):
    """
    Generate CSV file with topics and keywords.

    Args:
        topics: List of topic dicts from extract_topic_keywords()
        output_path: Path to save CSV file
        labels: Optional list of interpretive labels
        num_keywords: Number of keywords to display
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if labels:
            fieldnames = ['topic_id', f'top_{num_keywords}_keywords', 'preliminary_label']
        else:
            fieldnames = ['topic_id', f'top_{num_keywords}_keywords']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, topic in enumerate(topics):
            keywords_str = format_keywords_inline(topic['keywords'])
            row = {
                'topic_id': topic['topic_id'],
                f'top_{num_keywords}_keywords': keywords_str
            }
            if labels:
                row['preliminary_label'] = labels[i] if i < len(labels) else "TBD"
            writer.writerow(row)


def generate_latex_table(topics, labels=None, num_keywords=10):
    """
    Generate LaTeX table with topics and inline keywords.

    Args:
        topics: List of topic dicts from extract_topic_keywords()
        labels: Optional list of interpretive labels
        num_keywords: Number of keywords to display

    Returns:
        String containing LaTeX table
    """
    lines = []

    # Table header
    if labels:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{|l|p{8cm}|p{3cm}|}")
        lines.append(r"\hline")
        lines.append(f"Topic \\# & Top {num_keywords} Keywords (with weights) & Preliminary Label \\\\")
    else:
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{|l|p{10cm}|}")
        lines.append(r"\hline")
        lines.append(f"Topic \\# & Top {num_keywords} Keywords (with weights) \\\\")

    lines.append(r"\hline")

    # Rows
    for i, topic in enumerate(topics):
        keywords_str = format_keywords_inline(topic['keywords'])
        # Escape special LaTeX characters
        keywords_str = keywords_str.replace('_', r'\_').replace('&', r'\&')

        if labels:
            label = labels[i] if i < len(labels) else "TBD"
            label = label.replace('_', r'\_').replace('&', r'\&')
            lines.append(f"{topic['topic_id']} & {keywords_str} & {label} \\\\")
        else:
            lines.append(f"{topic['topic_id']} & {keywords_str} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Topic keywords and interpretations}")
    lines.append(r"\label{tab:topics}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def auto_generate_labels(topics, num_label_words=3):
    """
    Auto-generate simple labels from top keywords.

    Args:
        topics: List of topic dicts from extract_topic_keywords()
        num_label_words: Number of top words to use for label

    Returns:
        List of auto-generated labels
    """
    labels = []
    for topic in topics:
        # Take first N words
        top_words = [word for word, prob in topic['keywords'][:num_label_words]]
        label = " / ".join(top_words)
        labels.append(label)
    return labels
