import os
import pandas as pd
import numpy as np
from library.utils import setup_logger, set_seed
from library.text_processing import RegexTransducer
from library.data_loader import load_dataset


class HierarchicalLookupModel:
    """
    Hierarchical Symbolic Retrieval Model for Text Normalization.

    Architecture:
    1. L2 Cache (Bigram): (prev_token, token) -> normalized_text
       Captures context-dependent normalizations.
    2. L1 Cache (Unigram): token -> normalized_text
       Captures global most frequent normalizations.
    3. Regex Transducer: Heuristic rules for OOV numerical patterns.
    4. Identity: Fallback to original text.
    """

    def __init__(self, cache_dir="./working/idea_1"):
        self.logger = setup_logger("HierarchicalLookupModel")
        self.cache_dir = cache_dir
        self.regex_engine = RegexTransducer()

        # Lookup tables
        self.l1_lookup = {}  # Map[token, normalized]
        self.l2_lookup = {}  # Map[(prev_token, token), normalized]

    def fit(self, train_split="train", load_cached_data=True, sample_ratio=1.0):
        """
        Learns the unigram and bigram mappings from the training data.
        Implements strict caching for the learned statistics.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        l1_path = os.path.join(self.cache_dir, "l1_stats.parquet")
        l2_path = os.path.join(self.cache_dir, "l2_stats.parquet")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(l1_path) and os.path.exists(l2_path):
            self.logger.info("Loading cached model statistics...")
            try:
                df_l1 = pd.read_parquet(l1_path)
                df_l2 = pd.read_parquet(l2_path)

                # Build L1 Dictionary
                self.l1_lookup = dict(zip(df_l1["before"], df_l1["after"]))

                # Build L2 Dictionary
                # Zip creates tuples of (prev, curr) for keys
                self.l2_lookup = dict(
                    zip(zip(df_l2["prev_before"], df_l2["before"]), df_l2["after"])
                )

                self.logger.info(
                    f"Loaded {len(self.l1_lookup)} unigram rules and {len(self.l2_lookup)} bigram rules."
                )
                return
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute Statistics from Scratch
        self.logger.info("Computing statistics from training data...")

        # Load data with context processing
        df = load_dataset(
            split=train_split,
            process_context=True,
            load_cached_data=load_cached_data,
            sample_ratio=sample_ratio,
        )

        # A. Unigram Statistics (L1)
        self.logger.info("Aggregating Unigram Statistics...")
        # Group by input and target, count occurrences
        l1_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
        # Select the 'after' with the highest count for each 'before'
        l1_best = l1_counts.sort_values(
            ["before", "count"], ascending=[True, False]
        ).drop_duplicates(["before"])

        # Update internal state
        self.l1_lookup = dict(zip(l1_best["before"], l1_best["after"]))

        # Save to cache
        l1_best[["before", "after"]].to_parquet(l1_path, index=False)

        # B. Bigram Statistics (L2)
        self.logger.info("Aggregating Bigram Statistics...")
        # Group by prev, input, target
        l2_counts = (
            df.groupby(["prev_before", "before", "after"])
            .size()
            .reset_index(name="count")
        )
        # Select best target
        l2_best = l2_counts.sort_values(
            ["prev_before", "before", "count"], ascending=[True, True, False]
        ).drop_duplicates(["prev_before", "before"])

        # Update internal state
        self.l2_lookup = dict(
            zip(zip(l2_best["prev_before"], l2_best["before"]), l2_best["after"])
        )

        # Save to cache
        l2_best[["prev_before", "before", "after"]].to_parquet(l2_path, index=False)

        self.logger.info(
            f"Training complete. Learned {len(self.l1_lookup)} unigram rules and {len(self.l2_lookup)} bigram rules."
        )

    def predict(self, df):
        """
        Predicts normalized text for a DataFrame containing 'before' and 'prev_before' columns.
        Returns a list of predictions.
        """
        tokens = df["before"].tolist()
        prev_tokens = df["prev_before"].tolist()

        preds = []

        # Iterate through tokens (List comprehension/loop is generally efficient for this lookup logic)
        for token, prev in zip(tokens, prev_tokens):
            # 1. L2 Lookup (Contextual)
            l2_key = (prev, token)
            if l2_key in self.l2_lookup:
                preds.append(self.l2_lookup[l2_key])
                continue

            # 2. L1 Lookup (Global)
            if token in self.l1_lookup:
                preds.append(self.l1_lookup[token])
                continue

            # 3. Regex Fallback (Heuristic)
            regex_pred = self.regex_engine.normalize(token)
            if regex_pred is not None:
                preds.append(regex_pred)
                continue

            # 4. Identity (Default)
            preds.append(token)

        return preds

    def evaluate(self, val_split="val", load_cached_data=True):
        """
        Evaluates the model on the validation set and prints accuracy.
        """
        self.logger.info("Starting evaluation...")

        # Load validation data with context
        df_val = load_dataset(
            split=val_split, process_context=True, load_cached_data=load_cached_data
        )

        # Generate predictions
        predictions = self.predict(df_val)
        targets = df_val["after"].tolist()

        # Calculate Accuracy
        correct = sum(p == t for p, t in zip(predictions, targets))
        total = len(targets)
        accuracy = correct / total

        # Print full precision as requested
        print(f"Validation Accuracy: {accuracy}")

        return accuracy

    def generate_submission(
        self,
        test_split="test",
        output_file="./submission/submission.csv",
        load_cached_data=True,
    ):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        self.logger.info("Generating submission...")

        # Load test data with context
        df_test = load_dataset(
            split=test_split, process_context=True, load_cached_data=load_cached_data
        )

        # Generate predictions
        predictions = self.predict(df_test)

        # Prepare submission DataFrame
        sub_df = pd.DataFrame({"id": df_test["id"], "after": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        # quoting=1 corresponds to csv.QUOTE_ALL (or similar depending on pandas version mapping,
        # but usually 1 is QUOTE_ALL in python csv module. In pandas to_csv, quoting follows csv module constants).
        # However, pandas to_csv default is usually sufficient. The sample submission uses quotes for strings.
        # We will use default pandas behavior which quotes when necessary, or force if needed.
        # Given the sample: "the", "quick" -> it seems all text fields are quoted.
        import csv

        sub_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

        self.logger.info(f"Submission saved to {output_file}")
