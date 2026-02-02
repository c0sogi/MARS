import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.model import get_model
from library.data_loader import process_split, NQDataset, collate_fn
from library.text_processing import build_vocab


def get_best_span(start_logits, end_logits):
    """
    Finds the optimal span (start, end) such that start <= end
    and score = start_logits[start] + end_logits[end] is maximized.

    Args:
        start_logits (np.array): Logits for start position.
        end_logits (np.array): Logits for end position.

    Returns:
        tuple: ((best_start, best_end), best_score)
    """
    seq_len = len(start_logits)

    # Create score matrix (seq_len, seq_len) where grid[i, j] = start[i] + end[j]
    s_matrix = start_logits.reshape(-1, 1)
    e_matrix = end_logits.reshape(1, -1)
    score_matrix = s_matrix + e_matrix

    # Mask out invalid spans where start_index > end_index
    # np.triu returns upper triangle (including diagonal), which corresponds to i <= j
    valid_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool))

    # Apply mask (set invalid to -inf)
    score_matrix[~valid_mask] = -float("inf")

    # Find indices of maximum score
    flat_idx = np.argmax(score_matrix)
    best_start = flat_idx // seq_len
    best_end = flat_idx % seq_len
    best_score = score_matrix[best_start, best_end]

    return (best_start, best_end), best_score


def generate_predictions():
    """
    Main inference function.
    Loads data, runs model, processes results, and writes submission file.
    """
    print("[Predictor] Starting inference pipeline...")

    # 1. Load Resources
    # Ensure vocab is built (should be available from training, or built here)
    vocab = build_vocab(load_cached_data=True)

    # Process test data (flattened candidates)
    # This will create/load processed_test.parquet
    test_df = process_split("test", load_cached_data=True)

    if test_df.empty:
        print("[Predictor] No test data found. Generating empty submission.")
        # Create dummy submission structure
        pd.DataFrame(columns=["example_id", "PredictionString"]).to_csv(
            Config.SUBMISSION_FILE_PATH, index=False
        )
        return

    dataset = NQDataset(test_df, vocab)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    model = get_model(load_weights=True)
    model.to(Config.DEVICE)
    model.eval()

    # Dictionary to store raw model outputs per example
    # Structure: example_id -> list of dicts with candidate info
    raw_results = {}

    print(f"[Predictor] Running inference on {len(dataset)} samples...")

    with torch.no_grad():
        for batch in loader:
            q_ids = batch["question_ids"].to(Config.DEVICE)
            c_ids = batch["candidate_ids"].to(Config.DEVICE)

            # Forward pass
            long_probs, start_logits, end_logits = model(q_ids, c_ids)

            # Move to CPU
            long_probs = long_probs.cpu().numpy().flatten()
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            example_ids = batch["example_ids"]
            candidate_indices = batch["candidate_indices"]

            for i, ex_id in enumerate(example_ids):
                if ex_id not in raw_results:
                    raw_results[ex_id] = []

                raw_results[ex_id].append(
                    {
                        "cand_idx": candidate_indices[i],
                        "long_prob": long_probs[i],
                        "start_logits": start_logits[i],
                        "end_logits": end_logits[i],
                    }
                )

    print("[Predictor] Inference complete. Processing results...")

    # 2. Select Answers
    final_decisions = {}  # ex_id -> {long_cand_idx, short_span_rel}

    for ex_id, candidates in raw_results.items():
        # Find candidate with max long probability for this example
        best_cand = max(candidates, key=lambda x: x["long_prob"])

        decision = {"long_cand_idx": None, "short_span_rel": None}

        # Thresholding for Long Answer
        if best_cand["long_prob"] > Config.LONG_ANSWER_THRESHOLD:
            decision["long_cand_idx"] = best_cand["cand_idx"]

            # Check for Short Answer within this best candidate
            span, score = get_best_span(
                best_cand["start_logits"], best_cand["end_logits"]
            )

            if score > Config.SHORT_ANSWER_THRESHOLD:
                decision["short_span_rel"] = span

        final_decisions[ex_id] = decision

    # 3. Generate Submission File
    # We need to read the original test file to map candidate indices to global token offsets
    print(f"[Predictor] Writing submission to {Config.SUBMISSION_FILE_PATH}...")

    submission_rows = []

    # Stream the raw test file to avoid memory issues
    # We rely on the fact that example_ids in raw_results match those in the file
    if os.path.exists(Config.TEST_DATA_PATH):
        with open(Config.TEST_DATA_PATH, "rb") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ex_id = entry["example_id"]

                # Default predictions (empty)
                long_pred_str = ""
                short_pred_str = ""

                if ex_id in final_decisions:
                    dec = final_decisions[ex_id]

                    if dec["long_cand_idx"] is not None:
                        # Retrieve the specific candidate object to get global offsets
                        candidates_list = entry["long_answer_candidates"]
                        c_idx = dec["long_cand_idx"]

                        if c_idx < len(candidates_list):
                            cand_obj = candidates_list[c_idx]
                            global_start = cand_obj["start_token"]
                            global_end = cand_obj["end_token"]

                            # Format Long Answer: "start:end"
                            long_pred_str = f"{global_start}:{global_end}"

                            # Process Short Answer
                            if dec["short_span_rel"] is not None:
                                rel_s, rel_e = dec["short_span_rel"]

                                # Map relative indices to global indices
                                s_global = global_start + rel_s
                                e_global = global_start + rel_e

                                # Sanity check: Ensure short answer is within the long answer bounds
                                # Note: rel_e is the index of the last token.
                                # The submission format usually expects "start:end" where end is exclusive?
                                # Looking at sample_submission: "-545833482873225036_long,105:200"
                                # NQ dataset defines end_token as exclusive.
                                # Our model predicts the index of the last token (inclusive).
                                # So we should add 1 to the end index for the string format.

                                if s_global < global_end and e_global < global_end:
                                    short_pred_str = f"{s_global}:{e_global + 1}"

                submission_rows.append(
                    {"example_id": f"{ex_id}_long", "PredictionString": long_pred_str}
                )
                submission_rows.append(
                    {"example_id": f"{ex_id}_short", "PredictionString": short_pred_str}
                )
    else:
        print(f"[Predictor] Warning: Test data path {Config.TEST_DATA_PATH} not found.")

    # Create DataFrame and save
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"[Predictor] Submission saved. Total rows: {len(sub_df)}")
