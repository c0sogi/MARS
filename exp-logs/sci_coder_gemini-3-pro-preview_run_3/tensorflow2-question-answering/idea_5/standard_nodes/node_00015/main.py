import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import warnings
from collections import Counter

# Import library modules
from library.configuration import Config
from library.trainer import train_ranker_model, train_reader_model
from library.predictor import Evaluator
from library.data_loader import RankerDatasetBuilder
from library.model_ranker import GradientBoostingRanker
from library.model_reader import ReaderTrainer
from library.feature_engineering import get_candidates_from_json
from library.text_utils import (
    tokenize,
    strip_html_tags,
    text_to_indices,
    map_clean_to_raw_span,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_fast_config():
    """
    Overrides configuration parameters to ensure the script runs within the time limit.
    """
    # Reduce training data size for speed
    Config.TRAIN_SAMPLE_SIZE = 15000

    # Ranker settings
    Config.RANKER_NUM_BOOST_ROUND = 200
    Config.RANKER_EARLY_STOPPING_ROUNDS = 20

    # Reader settings
    Config.READER_EPOCHS = 3
    Config.READER_BATCH_SIZE = 128

    print("Configuration updated for fast baseline execution.")


def compute_f1(tp, fp, fn):
    """Calculates F1 score from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def get_ground_truth(data):
    """Parses ground truth spans from JSON data."""
    long_answers = []
    short_answers = []

    annotations = data.get("annotations", [])
    for ann in annotations:
        # Long Answer
        la = ann.get("long_answer", {})
        if la.get("start_token", -1) != -1:
            long_answers.append(f"{la['start_token']}:{la['end_token']}")

        # Short Answer
        sas = ann.get("short_answers", [])
        for sa in sas:
            short_answers.append(f"{sa['start_token']}:{sa['end_token']}")

        # Yes/No Answer (treated as short answer text)
        yes_no = ann.get("yes_no_answer", "NONE")
        if yes_no != "NONE":
            short_answers.append(yes_no)

    return set(long_answers), set(short_answers)


def validate_and_analyze():
    """
    Performs validation on the hold-out set, computes the official metric,
    and runs failure analysis.
    """
    print("\n--- Starting Validation & Failure Analysis ---")

    # 1. Load Models
    ranker = GradientBoostingRanker()
    ranker.load_model()

    reader_trainer = ReaderTrainer()
    reader_trainer.load_model()
    vocab = reader_trainer.vocab

    # 2. Prepare Ranker Validation Features
    print("Building Ranker Validation Features...")
    val_features_df = RankerDatasetBuilder.build_val_set(load_cached_data=True)

    # Predict Scores
    print("Predicting Ranker Scores...")
    scores = ranker.predict(val_features_df)
    val_features_df["score"] = scores

    # Get best candidate per example
    best_candidates = (
        val_features_df.sort_values("score", ascending=False)
        .groupby("example_id")
        .first()
        .reset_index()
    )
    best_cand_map = dict(
        zip(
            best_candidates["example_id"],
            zip(best_candidates["candidate_index"], best_candidates["score"]),
        )
    )

    # 3. Validation Loop
    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA_PATH}"
        )

    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    tp = 0
    fp = 0
    fn = 0

    # Data for failure analysis
    analysis_data = []

    print(f"Validating on {len(val_metadata)} examples...")

    # Group by file to optimize IO
    for file_name, group in val_metadata.groupby("file_path"):
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            continue

        with open(file_path, "rb") as f:
            for _, row in group.iterrows():
                example_id = row["example_id"]

                # Ground Truth
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                gt_long, gt_short = get_ground_truth(data)

                # Prediction
                pred_long = ""
                pred_short = ""
                ranker_score = 0.0

                if example_id in best_cand_map:
                    cand_idx, ranker_score = best_cand_map[example_id]

                    if ranker_score > Config.RANKER_THRESHOLD:
                        candidates = get_candidates_from_json(data)
                        if 0 <= int(cand_idx) < len(candidates):
                            selected_cand = candidates[int(cand_idx)]

                            # Long Answer
                            pred_long = f"{selected_cand['start_token']}:{selected_cand['end_token']}"

                            # Short Answer Inference
                            q_text = data.get("question_text", "")
                            raw_cand_tokens = selected_cand["tokens"]
                            clean_cand_tokens, clean_map = strip_html_tags(
                                raw_cand_tokens
                            )
                            clean_cand_text = " ".join(clean_cand_tokens)

                            q_indices = text_to_indices(
                                q_text, vocab, max_len=Config.MAX_Q_LEN
                            )
                            ctx_indices = text_to_indices(
                                clean_cand_text, vocab, max_len=Config.MAX_CTX_LEN
                            )

                            q_batch = np.array([q_indices])
                            ctx_batch = np.array([ctx_indices])

                            start_probs, end_probs = reader_trainer.predict(
                                q_batch, ctx_batch
                            )

                            score_mat = np.outer(start_probs[0], end_probs[0])
                            score_mat = np.triu(score_mat)
                            flat_idx = np.argmax(score_mat)
                            best_start_clean, best_end_clean = np.unravel_index(
                                flat_idx, score_mat.shape
                            )

                            raw_rel_start, raw_rel_end = map_clean_to_raw_span(
                                best_start_clean, best_end_clean + 1, clean_map
                            )

                            if raw_rel_start != -1 and raw_rel_end != -1:
                                final_start = (
                                    selected_cand["start_token"] + raw_rel_start
                                )
                                final_end = selected_cand["start_token"] + raw_rel_end
                                pred_short = f"{final_start}:{final_end}"

                # Metric Computation (Micro F1 Logic)
                # Check Long
                is_correct_long = False
                if pred_long:
                    if pred_long in gt_long:
                        tp += 1
                        is_correct_long = True
                    else:
                        fp += 1
                else:
                    if gt_long:
                        fn += 1
                    # else: True Negative (ignored in F1)

                # Check Short
                is_correct_short = False
                if pred_short:
                    if pred_short in gt_short:
                        tp += 1
                        is_correct_short = True
                    else:
                        fp += 1
                else:
                    if gt_short:
                        fn += 1

                # Failure Analysis Data Collection
                # We define "Error" as failing to predict correctly when an answer exists,
                # or predicting incorrectly.
                # Simplified error metric: 1 if any part is wrong, 0 if perfect.
                # Or better: 1 - (local F1).

                # Calculate local F1 for this example
                local_tp = (1 if is_correct_long else 0) + (
                    1 if is_correct_short else 0
                )
                local_fp = (1 if pred_long and not is_correct_long else 0) + (
                    1 if pred_short and not is_correct_short else 0
                )
                local_fn = (1 if gt_long and not pred_long else 0) + (
                    1 if gt_short and not pred_short else 0
                )

                local_f1 = compute_f1(local_tp, local_fp, local_fn)
                error_magnitude = 1.0 - local_f1

                doc_len = len(data.get("document_text", "").split())
                q_len = len(data.get("question_text", "").split())

                analysis_data.append(
                    {
                        "error": error_magnitude,
                        "doc_len": doc_len,
                        "q_len": q_len,
                        "ranker_score": ranker_score,
                    }
                )

    # Final Metric
    final_f1 = compute_f1(tp, fp, fn)
    print(f"Final Validation Metric: {final_f1}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    df_analysis = pd.DataFrame(analysis_data)

    if not df_analysis.empty:
        correlations = df_analysis.corr()["error"].drop("error")
        print("Correlation between Model Error and Input Features:")
        print(correlations.to_string())

        # Identify strongest factor
        max_corr_feat = correlations.abs().idxmax()
        print(
            f"\nFeature most associated with error: {max_corr_feat} (corr: {correlations[max_corr_feat]:.4f})"
        )
    else:
        print("No analysis data available.")


def main():
    # 1. Setup
    set_fast_config()

    # 2. Training
    # Ranker
    print("\n=== Training Ranker ===")
    train_ranker_model(load_cached_data=False, sample_size=Config.TRAIN_SAMPLE_SIZE)

    # Reader
    print("\n=== Training Reader ===")
    train_reader_model(load_cached_data=False, sample_size=Config.TRAIN_SAMPLE_SIZE)

    # 3. Validation & Analysis
    validate_and_analyze()

    # 4. Submission
    print("\n=== Generating Submission ===")
    evaluator = Evaluator()
    evaluator.generate_submission(load_cached_data=False)

    print("\nExecution Completed.")


if __name__ == "__main__":
    main()
