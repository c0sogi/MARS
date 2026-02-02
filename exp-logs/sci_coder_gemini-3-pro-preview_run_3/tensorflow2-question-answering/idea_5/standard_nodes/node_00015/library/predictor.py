import os
import json
import pandas as pd
import numpy as np
import torch
from library.configuration import Config
from library.model_ranker import GradientBoostingRanker
from library.model_reader import ReaderTrainer
from library.data_loader import RankerDatasetBuilder
from library.feature_engineering import get_candidates_from_json
from library.text_utils import (
    tokenize,
    strip_html_tags,
    text_to_indices,
    map_clean_to_raw_span,
)


class Evaluator:
    """
    Orchestrates the inference pipeline for Question Answering.
    Segments documents, ranks candidates, extracts spans, and formats submissions.
    """

    def __init__(self):
        self.ranker = GradientBoostingRanker()
        self.reader_trainer = ReaderTrainer()
        self.vocab = None
        self.device = Config.DEVICE

    def load_models(self):
        """
        Loads the trained Ranker and Reader models from disk.
        """
        print("Loading Ranker model...")
        self.ranker.load_model()

        print("Loading Reader model and vocabulary...")
        self.reader_trainer.load_model()
        self.vocab = self.reader_trainer.vocab

    def generate_submission(self, load_cached_data=True, sample_size=None):
        """
        Generates the submission file for the test set.

        Args:
            load_cached_data (bool): Whether to use cached ranker features for test set.
            sample_size (int, optional): Limit number of test examples for debugging.
        """
        print("--- Starting Submission Generation ---")

        # Ensure models are loaded
        if self.ranker.model is None or self.reader_trainer.model is None:
            self.load_models()

        # ---------------------------------------------------------
        # 1. Ranker Inference
        # ---------------------------------------------------------
        print("Building Ranker Test Features...")
        # This handles feature extraction and caching internally
        test_features_df = RankerDatasetBuilder.build_test_set(
            load_cached_data=load_cached_data, sample_size=sample_size
        )

        print("Predicting Long Answer scores...")
        scores = self.ranker.predict(test_features_df)
        test_features_df["score"] = scores

        # Select best candidate per example
        # Sort by score descending and take the first one for each example_id
        best_candidates = (
            test_features_df.sort_values("score", ascending=False)
            .groupby("example_id")
            .first()
            .reset_index()
        )

        # Create a lookup dictionary: example_id -> (candidate_index, score)
        best_cand_map = dict(
            zip(
                best_candidates["example_id"],
                zip(best_candidates["candidate_index"], best_candidates["score"]),
            )
        )

        # ---------------------------------------------------------
        # 2. Reader Inference & Result Assembly
        # ---------------------------------------------------------

        # Load Test Metadata to access raw text efficiently
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA_PATH}"
            )

        test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)
        if sample_size is not None:
            test_metadata = test_metadata.head(sample_size)

        results = []
        print("Processing Reader Inference...")

        # Group by file to optimize IO (open file once per group)
        for file_name, group in test_metadata.groupby("file_path"):
            file_path = os.path.join(Config.INPUT_DIR, file_name)
            if not os.path.exists(file_path):
                continue

            with open(file_path, "rb") as f:
                for _, row in group.iterrows():
                    example_id = row["example_id"]

                    # Default predictions (empty/null)
                    long_ans_str = ""
                    short_ans_str = ""

                    # Check if we have a ranker decision for this example
                    if example_id in best_cand_map:
                        cand_idx, score = best_cand_map[example_id]

                        # Apply Thresholding Logic
                        if score > Config.RANKER_THRESHOLD:
                            # Read raw JSON data
                            f.seek(row["byte_offset"])
                            line = f.readline()

                            if line:
                                try:
                                    data = json.loads(line)
                                    candidates = get_candidates_from_json(data)

                                    # Validate candidate index
                                    if 0 <= int(cand_idx) < len(candidates):
                                        selected_cand = candidates[int(cand_idx)]

                                        # --- Long Answer Prediction ---
                                        long_start = selected_cand["start_token"]
                                        long_end = selected_cand["end_token"]
                                        long_ans_str = f"{long_start}:{long_end}"

                                        # --- Short Answer Extraction ---
                                        q_text = data.get("question_text", "")
                                        raw_cand_tokens = selected_cand["tokens"]

                                        # Clean tokens for Reader (remove HTML tags)
                                        clean_cand_tokens, clean_map = strip_html_tags(
                                            raw_cand_tokens
                                        )
                                        clean_cand_text = " ".join(clean_cand_tokens)

                                        # Vectorize
                                        q_indices = text_to_indices(
                                            q_text, self.vocab, max_len=Config.MAX_Q_LEN
                                        )
                                        ctx_indices = text_to_indices(
                                            clean_cand_text,
                                            self.vocab,
                                            max_len=Config.MAX_CTX_LEN,
                                        )

                                        # Create batch (size 1)
                                        q_batch = np.array([q_indices])
                                        ctx_batch = np.array([ctx_indices])

                                        # Predict probabilities
                                        start_probs, end_probs = (
                                            self.reader_trainer.predict(
                                                q_batch, ctx_batch
                                            )
                                        )

                                        # Decode best span (greedy approach)
                                        # Maximize joint probability P(start) * P(end) s.t. start <= end
                                        score_mat = np.outer(
                                            start_probs[0], end_probs[0]
                                        )
                                        # Mask invalid spans where end index < start index
                                        score_mat = np.triu(score_mat)

                                        flat_idx = np.argmax(score_mat)
                                        best_start_clean, best_end_clean = (
                                            np.unravel_index(flat_idx, score_mat.shape)
                                        )

                                        # Map clean indices back to raw document indices
                                        # Note: The model predicts the inclusive end index of the token in the clean sequence.
                                        # map_clean_to_raw_span expects exclusive end index for slicing.
                                        # Therefore, we pass best_end_clean + 1.
                                        raw_rel_start, raw_rel_end = (
                                            map_clean_to_raw_span(
                                                best_start_clean,
                                                best_end_clean + 1,
                                                clean_map,
                                            )
                                        )

                                        if raw_rel_start != -1 and raw_rel_end != -1:
                                            # Convert relative indices to absolute document indices
                                            final_short_start = (
                                                long_start + raw_rel_start
                                            )
                                            final_short_end = long_start + raw_rel_end
                                            short_ans_str = (
                                                f"{final_short_start}:{final_short_end}"
                                            )

                                except json.JSONDecodeError:
                                    pass

                    # Append formatted results
                    results.append(
                        {
                            "example_id": str(example_id) + "_long",
                            "PredictionString": long_ans_str,
                        }
                    )
                    results.append(
                        {
                            "example_id": str(example_id) + "_short",
                            "PredictionString": short_ans_str,
                        }
                    )

        # ---------------------------------------------------------
        # 3. Save Submission
        # ---------------------------------------------------------
        submission_df = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(
            f"Submission saved to {Config.SUBMISSION_PATH}. Rows: {len(submission_df)}"
        )
