import numpy as np
from collections import Counter
from library.config import Config
from library.data_utils import build_tokenizer, build_idf_weights


class TFIDFExtractor:
    """
    Heuristic-based short answer extractor using TF-IDF and sliding window cosine similarity.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the extractor by loading the tokenizer and IDF weights.

        Args:
            load_cached_data (bool): Whether to load cached vocabulary and IDF weights.
        """
        # Load tokenizer
        self.tokenizer = build_tokenizer(
            load_cached_data=load_cached_data, data_path=Config.TRAIN_DATA_PATH
        )

        # Load IDF weights
        self.idf_weights = build_idf_weights(
            self.tokenizer,
            load_cached_data=load_cached_data,
            data_path=Config.TRAIN_DATA_PATH,
        )

        # Auxiliary verbs for Yes/No detection
        self.auxiliary_verbs = {
            "is",
            "are",
            "was",
            "were",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
            "may",
            "might",
            "must",
            "will",
            "shall",
        }

    def _get_tfidf_vector(self, tokens):
        """
        Computes the L2-normalized TF-IDF vector for a list of tokens.

        Args:
            tokens (list): List of string tokens.

        Returns:
            np.array: Normalized TF-IDF vector of shape (vocab_size,).
        """
        # Calculate Term Frequency (TF)
        tf_vector = np.zeros(self.tokenizer.vocab_size, dtype=np.float32)
        token_indices = self.tokenizer.text_to_sequence(tokens)

        # We use raw counts for TF here
        counts = Counter(token_indices)

        for idx, count in counts.items():
            # Skip PAD and UNK if desired, but UNK might hold info.
            # Usually PAD (0) is ignored.
            if idx == 0:
                continue
            if idx < self.tokenizer.vocab_size:
                tf_vector[idx] = count

        # Apply Global IDF
        tfidf_vector = tf_vector * self.idf_weights

        # L2 Normalization
        norm = np.linalg.norm(tfidf_vector)
        if norm > 0:
            tfidf_vector = tfidf_vector / norm

        return tfidf_vector

    def sliding_window_search(self, question_text, candidate_text, candidate_start_idx):
        """
        Finds the best short answer span within a long answer candidate.

        Args:
            question_text (str): The question.
            candidate_text (str): The text of the long answer candidate.
            candidate_start_idx (int): The starting token index of the candidate in the document.

        Returns:
            dict: Dictionary containing:
                - 'score': Cosine similarity score.
                - 'text': The extracted short answer text.
                - 'start_token': Absolute start token index in document.
                - 'end_token': Absolute end token index in document.
        """
        # Tokenize inputs (simple split to match data_utils logic)
        q_tokens = question_text.split()
        c_tokens = candidate_text.split()

        if not q_tokens or not c_tokens:
            return {"score": 0.0, "text": "", "start_token": -1, "end_token": -1}

        # Vectorize Question
        q_vec = self._get_tfidf_vector(q_tokens)

        best_score = -1.0
        best_span = (0, 0)

        # Sliding Window
        window_size = Config.SHORT_ANSWER_WINDOW_SIZE
        stride = Config.SHORT_ANSWER_STRIDE

        # If candidate is shorter than window, take the whole thing
        if len(c_tokens) <= window_size:
            c_vec = self._get_tfidf_vector(c_tokens)
            score = np.dot(q_vec, c_vec)
            return {
                "score": score,
                "text": candidate_text,
                "start_token": candidate_start_idx,
                "end_token": candidate_start_idx + len(c_tokens),
            }

        # Slide
        for i in range(0, len(c_tokens) - window_size + 1, stride):
            window_tokens = c_tokens[i : i + window_size]
            w_vec = self._get_tfidf_vector(window_tokens)

            # Cosine similarity (vectors are already normalized)
            score = np.dot(q_vec, w_vec)

            if score > best_score:
                best_score = score
                best_span = (i, i + window_size)

        # Handle the tail if stride skips the very end, though usually not critical for heuristics

        # Construct result
        rel_start, rel_end = best_span
        short_text = " ".join(c_tokens[rel_start:rel_end])

        return {
            "score": best_score,
            "text": short_text,
            "start_token": candidate_start_idx + rel_start,
            "end_token": candidate_start_idx + rel_end,
        }

    def determine_yes_no(self, question_text, short_answer_text):
        """
        Determines if the answer is YES or NO based on heuristics.

        Args:
            question_text (str): The question.
            short_answer_text (str): The extracted short answer text.

        Returns:
            str: 'YES', 'NO', or 'NONE'.
        """
        if not question_text or not short_answer_text:
            return "NONE"

        q_tokens = question_text.lower().split()
        a_tokens = short_answer_text.lower().split()

        if not q_tokens or not a_tokens:
            return "NONE"

        # Check if question starts with an auxiliary verb
        first_word = q_tokens[0]
        if first_word in self.auxiliary_verbs:
            # Check answer start
            ans_start = a_tokens[0]
            # Simple check for yes/no words (could be expanded with punctuation handling)
            # e.g. "Yes," -> "yes," -> "yes"
            clean_start = ans_start.strip(",.")

            if clean_start == "yes":
                return "YES"
            elif clean_start == "no":
                return "NO"

        return "NONE"
