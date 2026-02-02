import re
import string
from typing import List, Set

# Import configuration
from library.config import FeatureConfig

# Import NLTK for stemming
from nltk.stem import PorterStemmer

# Import Scikit-learn for a reliable stopword list that doesn't require external downloads
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextPreprocessor:
    """
    Handles text normalization, tokenization, stopword removal, and stemming.
    """

    def __init__(self):
        self.use_stemming = FeatureConfig.USE_STEMMING
        self.remove_stops = FeatureConfig.REMOVE_STOPWORDS

        self.stemmer = PorterStemmer() if self.use_stemming else None
        self.stopwords: Set[str] = (
            set(ENGLISH_STOP_WORDS) if self.remove_stops else set()
        )

        # Regex for tokenization: matches alphanumeric sequences
        # This effectively splits by whitespace and punctuation
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def tokenize(self, text: str) -> List[str]:
        """
        Converts text to lowercase and splits into tokens using a regex that
        keeps words with 2 or more alphanumeric characters.
        """
        if not text:
            return []

        # Lowercase and find all matching tokens
        # This handles punctuation by treating it as a delimiter (unless part of the word boundary)
        return self.token_pattern.findall(text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filters out tokens that are in the stopword list.
        """
        if not self.remove_stops:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Applies Porter Stemming to reduce words to their root form.
        """
        if not self.use_stemming:
            return tokens
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str) -> List[str]:
        """
        Full preprocessing pipeline: Tokenize -> Remove Stopwords -> Stem.
        """
        if text is None:
            return []

        tokens = self.tokenize(text)

        if self.remove_stops:
            tokens = self.remove_stopwords(tokens)

        if self.use_stemming:
            tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_sentence(self, text: str) -> str:
        """
        Preprocesses text and joins tokens back into a single string.
        Useful for debugging or specific feature generation steps.
        """
        tokens = self.preprocess(text)
        return " ".join(tokens)
