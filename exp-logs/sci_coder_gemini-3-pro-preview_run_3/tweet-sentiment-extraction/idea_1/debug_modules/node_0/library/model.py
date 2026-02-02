import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    SUBMISSION_PATH,
    SEED,
    SMOOTHING_ALPHA,
    MIN_FREQ,
    SCORE_SHIFT,
)
from library.utils import tokenize, preprocess, jaccard, set_seed, load_processed_data


class SentimentRelevanceModel:
    """
    A statistical model that predicts the selected text based on word-level
    relevance probabilities derived from the training data.
    """

    def __init__(self):
        self.pos_weights = {}
        self.neg_weights = {}
        set_seed(SEED)

    def fit(self, train_df, load_cached_data=True):
        """
        Learns the sentiment-specific word probabilities.

        Args:
            train_df (pd.DataFrame): Training data.
            load_cached_data (bool): Whether to try loading weights from cache.
        """
        # Define cache paths
        os.makedirs(CACHE_DIR, exist_ok=True)
        pos_cache_path = os.path.join(CACHE_DIR, "pos_weights.parquet")
        neg_cache_path = os.path.join(CACHE_DIR, "neg_weights.parquet")

        # Attempt to load from cache
        if (
            load_cached_data
            and os.path.exists(pos_cache_path)
            and os.path.exists(neg_cache_path)
        ):
            print("Loading learned weights from cache...")
            self.pos_weights = self._load_weights(pos_cache_path)
            self.neg_weights = self._load_weights(neg_cache_path)
        else:
            print("Computing weights from training data...")
            # Filter data by sentiment
            pos_df = train_df[train_df["sentiment"] == "positive"]
            neg_df = train_df[train_df["sentiment"] == "negative"]

            # Compute weights
            self.pos_weights = self._compute_weights(pos_df)
            self.neg_weights = self._compute_weights(neg_df)

            # Save to cache
            self._save_weights(self.pos_weights, pos_cache_path)
            self._save_weights(self.neg_weights, neg_cache_path)
            print("Weights computed and cached.")

    def predict(self, test_df):
        """
        Generates predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data.

        Returns:
            pd.DataFrame: DataFrame containing 'textID' and 'selected_text'.
        """
        predictions = []

        for _, row in test_df.iterrows():
            text = row["text"]
            sentiment = row["sentiment"]

            # Strategy:
            # 1. Neutral tweets have high overlap -> predict full text.
            # 2. Positive/Negative tweets -> extract span based on learned weights.
            if sentiment == "neutral":
                pred = text
            else:
                weights = (
                    self.pos_weights if sentiment == "positive" else self.neg_weights
                )
                pred = self._extract_span(text, weights)

            predictions.append(pred)

        return pd.DataFrame({"textID": test_df["textID"], "selected_text": predictions})

    def evaluate(self, val_df):
        """
        Evaluates the model on the validation set using the Jaccard metric.

        Args:
            val_df (pd.DataFrame): Validation data.

        Returns:
            float: Mean Jaccard score.
        """
        print("Evaluating model on validation set...")
        preds_df = self.predict(val_df)

        scores = []
        for i in range(len(val_df)):
            target = val_df.iloc[i]["selected_text"]
            pred = preds_df.iloc[i]["selected_text"]
            scores.append(jaccard(target, pred))

        mean_score = np.mean(scores)
        print(f"Validation Jaccard Score: {mean_score}")
        return mean_score

    def generate_submission(self, test_df):
        """
        Generates predictions for the test set and saves them to the submission file.
        """
        print("Generating submission...")
        preds_df = self.predict(test_df)

        # Ensure submission directory exists (handled in config, but good practice)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        preds_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    def _compute_weights(self, df):
        """
        Computes P(selected | word) for a given dataframe subset.
        """
        total_counts = Counter()
        selected_counts = Counter()

        for _, row in df.iterrows():
            # Use preprocessed (lowercased) text for counting to ensure matching
            # Note: tokenize() splits on whitespace, preserving punctuation
            text_tokens = tokenize(preprocess(row["text"]))
            sel_tokens = set(tokenize(preprocess(row["selected_text"])))

            for token in text_tokens:
                total_counts[token] += 1
                if token in sel_tokens:
                    selected_counts[token] += 1

        weights = {}
        for token, count in total_counts.items():
            if count >= MIN_FREQ:
                # Laplace smoothing for probability estimation
                # P(selected|word) = (count_selected + alpha) / (count_total + 2*alpha)
                # We use 2*alpha because the outcome is binary (selected vs not selected)
                prob = (selected_counts[token] + SMOOTHING_ALPHA) / (
                    count + 2 * SMOOTHING_ALPHA
                )
                weights[token] = prob

        return weights

    def _extract_span(self, text, weights):
        """
        Extracts the best substring from text based on token weights.
        """
        # Get original tokens to reconstruct text later
        orig_tokens = tokenize(text)
        # Get lowercased tokens for weight lookup
        lower_tokens = tokenize(preprocess(text))

        if not orig_tokens:
            return text

        # Calculate scores: P(word) - tau
        scores = []
        for token in lower_tokens:
            prob = weights.get(token, 0.0)  # Default 0.0 for unknown words
            scores.append(prob - SCORE_SHIFT)

        # Find the contiguous sub-sequence with maximum sum
        best_start, best_end = self._find_max_subarray(scores)

        # Reconstruct the string from the original text using the indices
        return self._reconstruct_text(text, orig_tokens, best_start, best_end)

    def _find_max_subarray(self, scores):
        """
        Finds the start and end indices of the subarray with the maximum sum.
        """
        max_so_far = -float("inf")
        current_max = 0

        start_index = 0
        end_index = 0
        temp_start_index = 0

        for i, x in enumerate(scores):
            current_max += x

            if current_max > max_so_far:
                max_so_far = current_max
                start_index = temp_start_index
                end_index = i

            if current_max < 0:
                current_max = 0
                temp_start_index = i + 1

        # If all scores are negative and max_so_far is very small,
        # we might return the single best token (least negative).
        # However, the logic above handles standard cases.
        # If max_so_far remains -inf (empty list), we return 0,0.
        if max_so_far == -float("inf"):
            return 0, 0

        return start_index, end_index

    def _reconstruct_text(self, text, tokens, start_idx, end_idx):
        """
        Reconstructs the substring from the original text corresponding to the
        token range [start_idx, end_idx].
        """
        # We need to find the character positions of the selected tokens in the original text.
        # Since tokens are split by whitespace, we can search for them sequentially.

        current_pos = 0
        char_start = 0
        char_end = len(text)

        # Find the starting character of the first selected token
        for i in range(start_idx + 1):
            token = tokens[i]
            # Find token starting from current_pos
            loc = text.find(token, current_pos)
            if loc == -1:
                # Fallback: if exact reconstruction fails, join tokens with space
                return " ".join(tokens[start_idx : end_idx + 1])

            if i == start_idx:
                char_start = loc

            # Move past this token
            current_pos = loc + len(token)

        # Find the ending character of the last selected token
        # We continue from where the start token ended (or started)
        # Reset current_pos to char_start to ensure continuity
        current_pos = char_start
        for i in range(start_idx, end_idx + 1):
            token = tokens[i]
            loc = text.find(token, current_pos)
            if loc == -1:
                return " ".join(tokens[start_idx : end_idx + 1])

            current_pos = loc + len(token)
            if i == end_idx:
                char_end = current_pos

        return text[char_start:char_end]

    def _save_weights(self, weights, path):
        """Saves a weight dictionary to a Parquet file."""
        df = pd.DataFrame(list(weights.items()), columns=["token", "prob"])
        df.to_parquet(path, index=False)

    def _load_weights(self, path):
        """Loads a weight dictionary from a Parquet file."""
        df = pd.read_parquet(path)
        return dict(zip(df["token"], df["prob"]))


def run_model_pipeline(debug=False, debug_size=500):
    """
    Helper function to run the full pipeline: Load -> Fit -> Evaluate -> Submit.
    """
    # 1. Load Data
    train_df, val_df, test_df = load_processed_data(
        load_cached_data=True, debug=debug, debug_size=debug_size
    )

    # 2. Initialize and Fit Model
    model = SentimentRelevanceModel()
    model.fit(train_df, load_cached_data=True)

    # 3. Evaluate
    model.evaluate(val_df)

    # 4. Generate Submission
    model.generate_submission(test_df)
