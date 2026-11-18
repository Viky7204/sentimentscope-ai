"""
Text Preprocessing for SentimentScope AI
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean input text for sentiment analysis.
    - Lowercase
    - Remove URLs, mentions, hashtags
    - Remove punctuation and numbers
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords if NLTK is available."""
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
        return ' '.join(filtered)
    except Exception:
        # Fallback: just return cleaned text
        return text


def full_clean(text: str) -> str:
    """Complete cleaning pipeline."""
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
