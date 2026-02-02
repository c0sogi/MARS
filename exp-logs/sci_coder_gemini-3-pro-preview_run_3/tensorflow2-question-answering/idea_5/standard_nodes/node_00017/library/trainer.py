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


def train_ranker_model(load_cached_data=True, sample_size=None):
    """
    Trains the Gradient Boosting Ranker model.

    Args:
        load_cached_data (bool): Whether to load pre-computed features from cache.
        sample_size (int, optional): Limit dataset size for debugging.
    """
    print("--- Starting Ranker Training ---")
    ranker = GradientBoostingRanker()
    ranker.train(load_cached_data=load_cached_data, sample_size=sample_size)
    print("--- Ranker Training Completed ---")


def train_reader_model(load_cached_data=True, sample_size=None):
    """
    Trains the Stacked Bi-GRU Reader model.

    Args:
        load_cached_data (bool): Whether to load pre-computed data from cache.
        sample_size (int, optional): Limit dataset size for debugging.
    """
    print("--- Starting Reader Training ---")
    trainer = ReaderTrainer()
    trainer.train(load_cached_data=load_cached_data, sample_size=sample_size)
    print("--- Reader Training Completed ---")


def generate_submission(load_cached_data=True, sample_size=None):
    """
    Generates the submission file for the test set using trained Ranker and Reader models.

    Args:
        load_cached_data (bool): Whether to use cached ranker features for test set.
        sample_size (int, optional): Limit number of test examples for debugging.
    """
    print("--- Starting Submission Generation ---")

    # 1. Ranker Inference
    # Build or load test features
    print("Building Ranker Test Features...")
    test_features_df = RankerDatasetBuilder.build_test_set(
        load_cached_data=load_cached_data, sample_size=sample_size
    )

    # Load Ranker and Predict
    ranker = GradientBoostingRanker()
    ranker.load_model()

    print("Predicting Long Answer scores...")
    scores = ranker.predict(test_features_df)
    test_features_df["score"] = scores

    # Select best candidate per example
    # We sort by score descending and take the first one for each example_id
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

    # 2. Reader Inference Setup
    reader_trainer = ReaderTrainer()
    reader_trainer.load_model()  # Loads model and vocab
    vocab = reader_trainer.vocab
    device = Config.DEVICE

    # Load Test Metadata to access raw text
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    test_metadata = pd.read_csv(Config.TEST_METADATA_PATH)
    if sample_size is not None:
        test_metadata = test_metadata.head(sample_size)

    results = []

    print("Processing Reader Inference...")

    # Group by file to optimize IO
    for file_name, group in test_metadata.groupby("file_path"):
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "rb") as f:
            for _, row in group.iterrows():
                example_id = row["example_id"]

                # Default predictions (empty)
                long_ans_str = ""
                short_ans_str = ""

                # Retrieve Ranker decision
                if example_id in best_cand_map:
                    cand_idx, score = best_cand_map[example_id]

                    # Check Ranker Threshold
                    if score > Config.RANKER_THRESHOLD:
                        # Read raw data
                        f.seek(row["byte_offset"])
                        line = f.readline()
                        if line:
                            try:
                                data = json.loads(line)
                                candidates = get_candidates_from_json(data)

                                if 0 <= int(cand_idx) < len(candidates):
                                    selected_cand = candidates[int(cand_idx)]

                                    # Set Long Answer Prediction
                                    long_start = selected_cand["start_token"]
                                    long_end = selected_cand["end_token"]
                                    long_ans_str = f"{long_start}:{long_end}"

                                    # Prepare Reader Input
                                    q_text = data.get("question_text", "")
                                    # The candidate tokens are raw (with tags)
                                    raw_cand_tokens = selected_cand["tokens"]

                                    # Clean tokens for Reader
                                    clean_cand_tokens, clean_map = strip_html_tags(
                                        raw_cand_tokens
                                    )
                                    clean_cand_text = " ".join(clean_cand_tokens)

                                    # Vectorize
                                    q_indices = text_to_indices(
                                        q_text, vocab, max_len=Config.MAX_Q_LEN
                                    )
                                    ctx_indices = text_to_indices(
                                        clean_cand_text,
                                        vocab,
                                        max_len=Config.MAX_CTX_LEN,
                                    )

                                    # Batch dimension
                                    q_batch = np.array([q_indices])
                                    ctx_batch = np.array([ctx_indices])

                                    # Predict
                                    start_probs, end_probs = reader_trainer.predict(
                                        q_batch, ctx_batch
                                    )

                                    # Decode best span (greedy)
                                    # We want max(P(start=i) * P(end=j)) for i <= j
                                    # To simplify, we can take argmax of outer product
                                    score_mat = np.outer(start_probs[0], end_probs[0])
                                    # Mask invalid spans (j < i)
                                    score_mat = np.triu(score_mat)

                                    flat_idx = np.argmax(score_mat)
                                    best_start_clean, best_end_clean = np.unravel_index(
                                        flat_idx, score_mat.shape
                                    )

                                    # Map clean indices back to raw indices relative to candidate start
                                    # Note: best_end_clean is inclusive in the matrix index, but map function expects exclusive end index?
                                    # Actually, reader output is logits for specific positions.
                                    # If model predicts index K, it means the K-th token.
                                    # map_clean_to_raw_span expects exclusive end.
                                    # Let's assume the model predicts the *inclusive* end index of the answer.
                                    # So we pass best_end_clean + 1 to the mapper.

                                    raw_rel_start, raw_rel_end = map_clean_to_raw_span(
                                        best_start_clean, best_end_clean + 1, clean_map
                                    )

                                    if raw_rel_start != -1 and raw_rel_end != -1:
                                        # Convert to document absolute indices
                                        final_short_start = long_start + raw_rel_start
                                        final_short_end = long_start + raw_rel_end
                                        short_ans_str = (
                                            f"{final_short_start}:{final_short_end}"
                                        )

                            except json.JSONDecodeError:
                                pass

                # Append results
                # Format requires string ID
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

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}. Rows: {len(submission_df)}")
