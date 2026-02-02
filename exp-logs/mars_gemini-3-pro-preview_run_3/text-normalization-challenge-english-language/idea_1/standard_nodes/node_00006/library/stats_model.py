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
        self.l2_prev_lookup = {}  # Map[(prev_token, token), normalized]
        self.l2_next_lookup = {}  # Map[(token, next_token), normalized]
        self.l3_lookup = {}  # Map[(prev_token, token, next_token), normalized]

    def fit(self, train_split="train", load_cached_data=True, sample_ratio=1.0):
        """
        Learns the unigram, bigram, and trigram mappings from the training data.
        Implements strict caching for the learned statistics.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        l1_path = os.path.join(self.cache_dir, "l1_stats.parquet")
        l2_prev_path = os.path.join(self.cache_dir, "l2_prev_stats.parquet")
        l2_next_path = os.path.join(self.cache_dir, "l2_next_stats.parquet")
        l3_path = os.path.join(self.cache_dir, "l3_stats.parquet")

        # 1. Try Loading from Cache
        if (
            load_cached_data
            and os.path.exists(l1_path)
            and os.path.exists(l2_prev_path)
            and os.path.exists(l2_next_path)
            and os.path.exists(l3_path)
        ):
            self.logger.info("Loading cached model statistics...")
            try:
                df_l1 = pd.read_parquet(l1_path)
                df_l2_prev = pd.read_parquet(l2_prev_path)
                df_l2_next = pd.read_parquet(l2_next_path)
                df_l3 = pd.read_parquet(l3_path)

                # Build Dictionaries
                self.l1_lookup = dict(zip(df_l1["before"], df_l1["after"]))
                self.l2_prev_lookup = dict(
                    zip(
                        zip(df_l2_prev["prev_before"], df_l2_prev["before"]),
                        df_l2_prev["after"],
                    )
                )
                self.l2_next_lookup = dict(
                    zip(
                        zip(df_l2_next["before"], df_l2_next["next_before"]),
                        df_l2_next["after"],
                    )
                )
                self.l3_lookup = dict(
                    zip(
                        zip(
                            df_l3["prev_before"], df_l3["before"], df_l3["next_before"]
                        ),
                        df_l3["after"],
                    )
                )

                self.logger.info(
                    f"Loaded L1:{len(self.l1_lookup)}, L2_prev:{len(self.l2_prev_lookup)}, "
                    f"L2_next:{len(self.l2_next_lookup)}, L3:{len(self.l3_lookup)} rules."
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

        # Helper to compute and save best mapping
        def compute_stats(groupby_cols, output_path):
            self.logger.info(f"Aggregating stats for {groupby_cols}...")
            counts = (
                df.groupby(groupby_cols + ["after"]).size().reset_index(name="count")
            )
            # Sort by count desc, then drop duplicates to keep most frequent
            best = counts.sort_values(
                groupby_cols + ["count"],
                ascending=([True] * len(groupby_cols)) + [False],
            ).drop_duplicates(groupby_cols)
            best[groupby_cols + ["after"]].to_parquet(output_path, index=False)
            return best

        # A. Unigram (L1)
        l1_best = compute_stats(["before"], l1_path)
        self.l1_lookup = dict(zip(l1_best["before"], l1_best["after"]))

        # B. Bigram Left (L2 Prev)
        l2_prev_best = compute_stats(["prev_before", "before"], l2_prev_path)
        self.l2_prev_lookup = dict(
            zip(
                zip(l2_prev_best["prev_before"], l2_prev_best["before"]),
                l2_prev_best["after"],
            )
        )

        # C. Bigram Right (L2 Next)
        l2_next_best = compute_stats(["before", "next_before"], l2_next_path)
        self.l2_next_lookup = dict(
            zip(
                zip(l2_next_best["before"], l2_next_best["next_before"]),
                l2_next_best["after"],
            )
        )

        # D. Trigram (L3)
        l3_best = compute_stats(["prev_before", "before", "next_before"], l3_path)
        self.l3_lookup = dict(
            zip(
                zip(l3_best["prev_before"], l3_best["before"], l3_best["next_before"]),
                l3_best["after"],
            )
        )

        self.logger.info("Training complete.")

    def predict(self, df):
        """
        Predicts normalized text for a DataFrame containing 'before', 'prev_before', and 'next_before'.
        Returns a list of predictions.
        """
        tokens = df["before"].tolist()
        prev_tokens = df["prev_before"].tolist()
        next_tokens = df["next_before"].tolist()

        preds = []

        # Iterate through tokens
        for token, prev, next_tok in zip(tokens, prev_tokens, next_tokens):
            # 1. L3 Lookup (Trigram) - Cite solution_lesson_node_00001 (Hierarchical Backoff)
            l3_key = (prev, token, next_tok)
            if l3_key in self.l3_lookup:
                preds.append(self.l3_lookup[l3_key])
                continue

            # 2. L2 Prev Lookup (Left Context)
            l2_prev_key = (prev, token)
            if l2_prev_key in self.l2_prev_lookup:
                preds.append(self.l2_prev_lookup[l2_prev_key])
                continue

            # 3. L2 Next Lookup (Right Context)
            l2_next_key = (token, next_tok)
            if l2_next_key in self.l2_next_lookup:
                preds.append(self.l2_next_lookup[l2_next_key])
                continue

            # 4. L1 Lookup (Global)
            if token in self.l1_lookup:
                preds.append(self.l1_lookup[token])
                continue

            # 5. Regex Fallback (Heuristic)
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
