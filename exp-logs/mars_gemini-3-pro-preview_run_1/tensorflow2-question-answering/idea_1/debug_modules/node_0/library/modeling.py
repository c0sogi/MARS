import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from sklearn.metrics import log_loss, roc_auc_score
from library.config import Config
from library.utils import setup_logger, timer

# Initialize logger
logger = setup_logger("modeling")


class LongAnswerClassifier:
    """
    Wraps LightGBM for the binary classification task of identifying
    whether a candidate span is the correct long answer.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.feature_cols = None

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Trains the LightGBM model using the provided training and validation dataframes.
        Implements Early Stopping.

        Args:
            train_df (pd.DataFrame): Training data containing features and 'label'.
            val_df (pd.DataFrame): Validation data containing features and 'label'.
        """
        logger.info("Preparing data for LightGBM training...")

        # Identify feature columns (exclude metadata and labels)
        exclude_cols = [
            "example_id",
            "candidate_index",
            "question_text",
            "candidate_text",
            "document_url",
            "label",
            "start_token",
            "end_token",
        ]
        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        logger.info(
            f"Training with {len(self.feature_cols)} features: {self.feature_cols}"
        )

        # Create LightGBM Datasets
        dtrain = lgb.Dataset(
            train_df[self.feature_cols], label=train_df["label"], free_raw_data=False
        )
        dval = lgb.Dataset(
            val_df[self.feature_cols],
            label=val_df["label"],
            reference=dtrain,
            free_raw_data=False,
        )

        # Training
        logger.info("Starting LightGBM training...")

        # Callback for logging
        def log_metrics(env):
            if (env.iteration + 1) % self.config.VERBOSE_EVAL == 0:
                msg = f"[{env.iteration + 1}] "
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    msg += f"{data_name}-{eval_name}: {result:.8f} "
                logger.info(msg)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            log_metrics,
        ]

        self.model = lgb.train(
            self.config.LGBM_PARAMS,
            dtrain,
            num_boost_round=self.config.NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log final metrics
        train_preds = self.model.predict(train_df[self.feature_cols])
        val_preds = self.model.predict(val_df[self.feature_cols])

        train_loss = log_loss(train_df["label"], train_preds)
        val_loss = log_loss(val_df["label"], val_preds)

        try:
            train_auc = roc_auc_score(train_df["label"], train_preds)
            val_auc = roc_auc_score(val_df["label"], val_preds)
        except ValueError:
            train_auc = 0.0
            val_auc = 0.0

        logger.info(f"Final Training LogLoss: {train_loss:.8f}")
        logger.info(f"Final Validation LogLoss: {val_loss:.8f}")
        logger.info(f"Final Training AUC: {train_auc:.8f}")
        logger.info(f"Final Validation AUC: {val_auc:.8f}")

        # Save model
        self.save_model()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates probabilities for the input dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Model not trained and no model file found.")

        # Ensure feature columns match
        if self.feature_cols is None:
            # Try to infer from config or assume all numeric columns in df match training
            # For safety in this pipeline, we re-derive exclusion list
            exclude_cols = [
                "example_id",
                "candidate_index",
                "question_text",
                "candidate_text",
                "document_url",
                "label",
                "start_token",
                "end_token",
            ]
            self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        return self.model.predict(df[self.feature_cols])

    def save_model(self):
        """Saves the trained model to the cache directory."""
        model_path = self.config.get_cache_path("lgbm_model.txt")
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self) -> bool:
        """Loads the model from the cache directory."""
        model_path = self.config.get_cache_path("lgbm_model.txt")
        if os.path.exists(model_path):
            self.model = lgb.Booster(model_file=model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        return False


class ShortAnswerExtractor:
    """
    Extracts a short answer span from a long answer text using heuristic N-gram overlap
    and boolean logic.
    """

    def __init__(self, config: Config):
        self.config = config
        # Patterns for Yes/No questions
        self.yes_no_pattern = re.compile(
            r"^(did|do|does|was|were|is|are|can|could|would|should|has|have|had)\b",
            re.IGNORECASE,
        )

    def _tokenize(self, text):
        """Simple whitespace tokenization."""
        return str(text).lower().split()

    def _compute_overlap(self, query_tokens, span_tokens):
        """Computes Jaccard similarity between query and span."""
        s1 = set(query_tokens)
        s2 = set(span_tokens)
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def extract(
        self, question_text: str, candidate_text: str, candidate_start_token: int
    ):
        """
        Extracts the short answer.

        Args:
            question_text (str): The question.
            candidate_text (str): The text of the selected long answer.
            candidate_start_token (int): The starting token index of the long answer in the doc.

        Returns:
            dict: {
                "short_answer_type": "span" or "yes_no",
                "start_token": int (absolute),
                "end_token": int (absolute),
                "yes_no_answer": "YES" or "NO" or None
            }
        """
        q_tokens = self._tokenize(question_text)
        c_tokens = (
            candidate_text.split()
        )  # Preserve case for reconstruction, use lower for matching

        # 1. Check for Yes/No
        # Logic: If question starts with boolean indicator, check if answer contains explicit Yes/No
        # This is a weak heuristic for NQ but serves as a baseline.
        # A stronger heuristic: NQ dataset usually marks YES/NO if the short answer is literally YES/NO
        # or if the intent is boolean. Here we stick to a simplified extraction.
        if self.yes_no_pattern.match(question_text):
            # Simple check in first few tokens of candidate
            intro = " ".join(c_tokens[:10]).lower()
            if "yes" in intro.split():
                return {"short_answer_type": "yes_no", "yes_no_answer": "YES"}
            if "no" in intro.split():
                return {"short_answer_type": "yes_no", "yes_no_answer": "NO"}

        # 2. Span Extraction (N-gram overlap)
        # We slide a window over the candidate text to find the best overlap with the question.
        # Window size is dynamic or fixed max length.

        best_score = -1
        best_span = (0, 0)

        # We treat the candidate as a sequence of tokens.
        # We look for sentences or chunks. For simplicity, we use a sliding window.
        max_len = self.config.SHORT_ANSWER_MAX_TOKENS
        c_len = len(c_tokens)

        # Optimization: Only check windows that start at sentence boundaries or punctuation?
        # For baseline, stride of 1 is fine given short candidate lengths (paragraphs).

        stride = 1
        # Limit window sizes to search: [5, 10, 20, 30]
        window_sizes = [min(x, c_len) for x in [5, 10, 20, max_len] if x <= c_len]
        if not window_sizes and c_len > 0:
            window_sizes = [c_len]

        for w_size in window_sizes:
            for i in range(0, c_len - w_size + 1, stride):
                span = c_tokens[i : i + w_size]
                # Normalize for comparison
                span_norm = [t.lower() for t in span]
                score = self._compute_overlap(q_tokens, span_norm)

                if score > best_score:
                    best_score = score
                    best_span = (i, i + w_size)

        # If no overlap found, return the whole thing or first sentence?
        # Or return nothing?
        # If score is too low, maybe no short answer exists.
        if best_score <= 0.0:
            return {"short_answer_type": "null"}

        # Convert relative indices to absolute
        rel_start, rel_end = best_span
        abs_start = candidate_start_token + rel_start
        abs_end = candidate_start_token + rel_end

        return {
            "short_answer_type": "span",
            "start_token": abs_start,
            "end_token": abs_end,
            "yes_no_answer": None,
        }


def generate_submission(
    config: Config, classifier: LongAnswerClassifier, test_df: pd.DataFrame
):
    """
    Generates the submission file for the test set.

    Args:
        config (Config): Configuration object.
        classifier (LongAnswerClassifier): Trained model.
        test_df (pd.DataFrame): Test dataframe with features.
    """
    logger.info("Generating predictions for submission...")

    # 1. Predict Probabilities
    probs = classifier.predict(test_df)
    test_df["pred_score"] = probs

    # 2. Initialize Extractor
    extractor = ShortAnswerExtractor(config)

    # 3. Process by Example ID
    submission_rows = []

    # Group by example_id to find the best candidate per question
    grouped = test_df.groupby("example_id")

    for example_id, group in grouped:
        # Find candidate with max score
        best_idx = group["pred_score"].idxmax()
        best_row = group.loc[best_idx]
        max_score = best_row["pred_score"]

        long_pred_str = ""
        short_pred_str = ""

        # Thresholding
        if max_score >= config.LONG_ANSWER_THRESHOLD:
            # Long Answer Found
            long_start = best_row["start_token"]
            long_end = best_row["end_token"]
            long_pred_str = f"{long_start}:{long_end}"

            # Extract Short Answer
            short_res = extractor.extract(
                best_row["question_text"], best_row["candidate_text"], long_start
            )

            if short_res.get("short_answer_type") == "yes_no":
                short_pred_str = short_res["yes_no_answer"]
            elif short_res.get("short_answer_type") == "span":
                s_start = short_res["start_token"]
                s_end = short_res["end_token"]
                short_pred_str = f"{s_start}:{s_end}"
            else:
                short_pred_str = ""  # No short answer found within long answer

        # Append rows
        submission_rows.append(
            {"example_id": f"{example_id}_long", "PredictionString": long_pred_str}
        )
        submission_rows.append(
            {"example_id": f"{example_id}_short", "PredictionString": short_pred_str}
        )

    # 4. Create Submission DataFrame
    sub_df = pd.DataFrame(submission_rows)

    # Ensure correct column order
    sub_df = sub_df[["example_id", "PredictionString"]]

    # Save
    sub_df.to_csv(config.SUBMISSION_FILE, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_FILE}")
