import re
from typing import List, Tuple, Optional, Set
from library.text_utils import TextUtils
from library.config import Config


class SlidingWindowExtractor:
    """
    Extracts a short answer span from a long answer text using a sliding window
    heuristic based on n-gram overlap with the question.
    """

    def __init__(
        self, window_size: int = 10, threshold: int = Config.SHORT_OVERLAP_THRESHOLD
    ):
        """
        Args:
            window_size: The number of tokens in the sliding window.
            threshold: Minimum overlap count required to return a valid span.
        """
        self.window_size = window_size
        self.threshold = threshold

    def extract(
        self, question_text: str, long_answer_text: str
    ) -> Tuple[int, int, str]:
        """
        Finds the best window in the long answer that overlaps with the question.

        Args:
            question_text: The raw text of the question.
            long_answer_text: The raw text of the predicted long answer.

        Returns:
            A tuple containing:
            - start_token_offset (int): Relative start index in the long answer tokens.
            - end_token_offset (int): Relative end index in the long answer tokens.
            - text (str): The extracted text span.
            Returns (-1, -1, "") if no suitable span is found.
        """
        # Tokenize inputs
        q_tokens = TextUtils.tokenize(question_text)
        la_tokens = TextUtils.tokenize(long_answer_text)

        if not la_tokens or not q_tokens:
            return -1, -1, ""

        # Prepare sets for fast lookup
        q_set = set(q_tokens)

        # Prepare bigrams if possible
        q_bigrams: Set[Tuple[str, str]] = set()
        if len(q_tokens) > 1:
            q_bigrams = set(zip(q_tokens, q_tokens[1:]))

        max_overlap = -1
        best_window = (-1, -1)

        # Slide window over long answer tokens
        n_tokens = len(la_tokens)
        for i in range(n_tokens):
            end = min(i + self.window_size, n_tokens)
            window_tokens = la_tokens[i:end]

            # 1. Count unigram overlap
            overlap = sum(1 for t in window_tokens if t in q_set)

            # 2. Count bigram overlap (weighted equally here, effectively boosting score)
            if len(window_tokens) > 1 and q_bigrams:
                w_bigrams = set(zip(window_tokens, window_tokens[1:]))
                overlap += len(q_bigrams.intersection(w_bigrams))

            if overlap > max_overlap:
                max_overlap = overlap
                best_window = (i, end)

        # Check against threshold
        if max_overlap >= self.threshold:
            s, e = best_window
            # Reconstruct text from tokens (approximation, as reconstruction from tokens loses original whitespace)
            # In a strict pipeline, one might map tokens back to original character offsets if available.
            # Here we join with spaces as per the logic in the provided neural_ranker.py reference.
            span_text = " ".join(la_tokens[s:e])
            return s, e, span_text

        return -1, -1, ""


def detect_yes_no(span_text: str) -> str:
    """
    Determines if the extracted short answer span represents a Yes or No answer.

    Args:
        span_text: The text of the potential short answer.

    Returns:
        'YES', 'NO', or 'NONE'.
    """
    if not span_text:
        return "NONE"

    clean_s = span_text.lower().strip()

    # Heuristic: Starts with yes/no and is sufficiently short to be a direct answer rather than a sentence containing the word.
    # Length check < 10 chars handles cases like "Yes, it is." vs "Yesterday was..."
    if clean_s.startswith("yes") and len(clean_s) < 10:
        return "YES"
    elif clean_s.startswith("no") and len(clean_s) < 10:
        return "NO"

    return "NONE"
