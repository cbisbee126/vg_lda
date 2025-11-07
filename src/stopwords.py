"""
Stopwords for Video Game LDA Analysis

Centralized stopword definitions for preprocessing YouTube gaming comments.
Extracted from original notebooks (commit 77af16e, October 15, 2025).
"""

from nltk.corpus import stopwords


# Generic chat, filler, and platform-specific terms
GENERIC_CHAT = {
    'video', 'game', 'online', 'youtube', 'series', 'pls', 'lol', 'omg', 'xd',
    'people', 'thing', 'play', 'playing', 'make', 'time', 'love', 'look', 'want',
    'think', 'watch', 'know', 'got', 'use', 'cant', 'going', 'never', 'ever',
    'part', 'help', 'played', 'getting', 'doesnt', 'bad', 'pretty', 'show',
    'fuck', 'shit', 'talk', 'went', 'comment', 'cool', 'amazing', 'seen', 'best',
    'like', 'get', 'one', 'dont', 'would', 'first', 'really', 'see', 'also',
    'way', 'guy', 'good', 'say', 'back', 'much', 'still', 'even', 'man', 'thats',
    'need', 'bro', 'new', 'kid', 'every', 'always', 'could', 'said', 'please',
    'youre', 'actually', 'didnt', 'feel', 'ive', 'dude', 'name', 'keep', 'gon',
    'watching', 'everyone', 'hey', 'someone', 'made', 'come', 'great', 'give',
    'well', 'fun', 'nice', 'let', 'right', 'day', 'friend', 'thought', 'work',
    'mean', 'take', 'vid', 'lmao', 'lot', 'god', 'something', 'hope', 'put',
    'cause', 'literally', 'since', 'next', 'hate', 'used', 'saying', 'funny',
    'many', 'vids', 'tbh', 'wtf', 'ngl', 'hell', 'thank', 'thanks', 'maybe',
    'already', 'oh', 'real', 'whole', 'two', 'old', 'hour', 'minute', 'top',
    'last', 'final', 'big', 'small', 'long', 'short', 'fast', 'slow', 'soon',
    'later', 'yeah', 'yall', 'wanna', 'wont', 'idk', 'guess', 'sometimes',
    'isnt', 'easy', 'point', 'almost', 'behind', 'beginning', 'true', 'sure',
    'place', 'reason', 'whats', 'talking', 'hello', 'hi'
}

# YouTube/streaming platform metadata
PLATFORM_TERMS = {
    'view', 'stream', 'watched', 'bruh', 'tho', 'thumbnail', 'sub', 'channel',
    'content', 'clip', 'stream'
}

# Gaming content creator names
CREATOR_NAMES = {
    'ninja', 'sypher', 'sypherpk', 'nick', 'nickeh', 'nickeh30', 'nick_eh',
    'shroud', 'jonas', 'zylbrad', 'brad', 'arin', 'dan', 'delirious', 'sunless',
    'sunlesskhan', 'drake', 'papa_moon', 'papamoon'
}

# In-game character names (optional - use include_characters parameter)
CHARACTER_NAMES = {
    # Red Dead Redemption 2
    'arthur', 'john', 'dutch', 'micah', 'hosea', 'sadie', 'abigail', 'sean',
    'lenny', 'javier', 'bill', 'charles', 'pearson', 'strauss', 'trelawny',
    # Apex Legends
    'wraith', 'pathfinder', 'bloodhound', 'gibraltar', 'lifeline', 'caustic',
    'mirage', 'octane', 'wattson', 'crypto', 'revenant', 'loba', 'rampart',
    'horizon', 'fuse', 'valkyrie', 'seer', 'ash', 'mad_maggie', 'newcastle',
    # Valorant
    'jett', 'phoenix', 'sage', 'sova', 'viper', 'cypher', 'reyna', 'killjoy',
    'breach', 'omen', 'raze', 'skye', 'yoru', 'astra', 'kayo', 'chamber',
    'neon', 'fade', 'harbor', 'gekko',
    # Hollow Knight
    'hornet', 'quirrel', 'elderbug', 'cornifer', 'iselda', 'myla',
    # Baldur's Gate 3
    'shadowheart', 'gale', 'astarion', 'wyll', 'karlach', 'laezel',
    # Zelda BOTW
    'zelda', 'ganon', 'mipha', 'daruk', 'urbosa', 'revali',
    # Other common names that appear
    'todd', 'max', 'jack'
}

# Gaming metadata and ranking terms
GAMING_METADATA = {
    'ranked', 'rank', 'season', 'matchmaking', 'mmr', 'elo'
}

# Additional common words from extended stopword list
EXTENDED_COMMON = {
    'can_t', 'so_much', 'feel_like', 'oh_yeah_oh_yeah', 'sea_of_thief_sea',
    'of_thief', 'wiggle_wiggle_wiggle_wiggle', 'episode', 'gonna', 'anyone',
    'second', 'little', 'probably', 'without', 'everything', 'another', 'year',
    'stuff', 'around', 'wish', 'life', 'stop', 'wait', 'tell', 'start', 'leave',
    'hear', 'saw', 'call', 'change', 'remember', 'anyone', 'probably', 'maybe',
    'anyway', 'already', 'yet', 'still', 'even', 'also', 'else', 'whole',
    'point', 'true', 'real', 'finally', 'little', 'big', 'long', 'short',
    'high', 'low', 'fast', 'slow', 'try', 'find', 'get', 'got', 'make', 'take',
    'put', 'use', 'using', 'see', 'look', 'watch', 'watching', 'know', 'think',
    'say', 'said', 'want', 'need'
}

# Franchise/game-specific terms (optional based on analysis goals)
FRANCHISE_TOKENS = {
    'fortnite', 'apex', 'valorant', 'rocket_league', 'dota', 'zelda',
    'elden_ring', 'hollow_knight', 'red_dead_redemption', 'red_dead_redemption_2',
    'baldur', 'baldur_gate', 'baldur_gate_3', 'rdr', 'rdr2'
}


def get_stopwords(include_franchise=False, include_characters=True, include_nltk=True):
    """
    Get combined stopword set for preprocessing.

    Parameters
    ----------
    include_franchise : bool, default=False
        If True, include franchise/game-specific tokens in stopwords.
        Set to False (default) to allow game names in topics for cross-game analysis.
    include_characters : bool, default=True
        If True, include in-game character names in stopwords.
        Set to False to allow character names in topics (useful for narrative analysis).
    include_nltk : bool, default=True
        If True, include NLTK's standard English stopwords (~179 words).

    Returns
    -------
    set
        Combined stopword set.

    Examples
    --------
    >>> stopwords = get_stopwords()  # Default: exclude franchise, include characters
    >>> stopwords_with_chars = get_stopwords(include_characters=False)  # Keep character names
    """
    stop_set = set()

    if include_nltk:
        stop_set.update(stopwords.words("english"))

    stop_set.update(GENERIC_CHAT)
    stop_set.update(PLATFORM_TERMS)
    stop_set.update(CREATOR_NAMES)
    stop_set.update(GAMING_METADATA)
    stop_set.update(EXTENDED_COMMON)

    if include_franchise:
        stop_set.update(FRANCHISE_TOKENS)

    if include_characters:
        stop_set.update(CHARACTER_NAMES)

    return stop_set


def print_stopword_stats():
    """Print statistics about stopword categories."""
    print("Stopword Statistics:")
    print(f"  NLTK English: {len(stopwords.words('english'))} words")
    print(f"  Generic chat: {len(GENERIC_CHAT)} words")
    print(f"  Platform terms: {len(PLATFORM_TERMS)} words")
    print(f"  Creator names: {len(CREATOR_NAMES)} words")
    print(f"  Character names: {len(CHARACTER_NAMES)} words")
    print(f"  Gaming metadata: {len(GAMING_METADATA)} words")
    print(f"  Extended common: {len(EXTENDED_COMMON)} words")
    print(f"  Franchise tokens: {len(FRANCHISE_TOKENS)} words")
    print(f"\nTotal (default - with characters, no franchise): {len(get_stopwords())} words")
    print(f"Total (no characters, no franchise): {len(get_stopwords(include_characters=False))} words")
    print(f"Total (with characters + franchise): {len(get_stopwords(include_franchise=True, include_characters=True))} words")


if __name__ == "__main__":
    print_stopword_stats()
