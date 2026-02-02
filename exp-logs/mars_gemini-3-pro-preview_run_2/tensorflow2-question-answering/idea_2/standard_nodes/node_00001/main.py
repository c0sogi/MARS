import os
import sys
import numpy as np
import pandas as pd
import json
import torch  # For GPU detection requirement
from sklearn.metrics import f1_score

# Import provided libraries
from library.config import PathConfig, ModelConfig
from library.corpus_stats import IDFIndex
from library.data_loader import NQDataReader
from library.ranker_model import GradientBoostingRanker
from library.answer_selector import ShortAnswerHeuristic


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting runfile.py...")
    set_seed(ModelConfig.SEED)

    # 1. Initialize Components
    print("Initializing components...")
    # Build/Load IDF Index (needed for feature generation if not cached)
    idf_index = IDFIndex()
    idf_index.build_from_corpus(load_cached_data=True)

    data_loader = NQDataReader()
    ranker = GradientBoostingRanker()

    # 2. Load Data
    # Limit training size for fast baseline execution
    TRAIN_SAMPLE_SIZE = 50000
    VAL_SAMPLE_SIZE = 10000

    print(f"Loading training data (limit={TRAIN_SAMPLE_SIZE})...")
    train_df = data_loader.get_training_samples(
        sample_size=TRAIN_SAMPLE_SIZE, load_cached_data=True
    )

    print(f"Loading validation data (limit={VAL_SAMPLE_SIZE})...")
    val_df = data_loader.get_validation_samples(
        sample_size=VAL_SAMPLE_SIZE, load_cached_data=True
    )

    # 3. Train Model
    print("Training model...")
    ranker.train_model(train_df, val_df)

    # 4. Validation Assessment & Failure Analysis
    print("Performing validation assessment...")

    # Predict on validation set
    val_scores = ranker.predict_scores(val_df)
    val_df["score"] = val_scores

    # Calculate Micro F1 for Long Answers
    # Logic: Group by example_id, find max score.
    # If max_score > threshold, predict that candidate.
    # Compare with ground truth (label=1).

    # Group by example_id
    grouped = val_df.groupby("example_id")

    tp = 0
    fp = 0
    fn = 0

    feature_cols = [c for c in val_df.columns if c.startswith("f_")]

    print("Calculating validation metrics...")
    for name, group in grouped:
        # Ground Truth
        gt_candidates = group[group["label"] == 1]
        has_gt = len(gt_candidates) > 0
        gt_idx = gt_candidates.iloc[0]["candidate_index"] if has_gt else -1

        # Prediction
        best_row_idx = group["score"].idxmax()
        best_score = group.loc[best_row_idx, "score"]
        best_cand_idx = group.loc[best_row_idx, "candidate_index"]

        predicted_has_answer = best_score > ModelConfig.LONG_CONFIDENCE_THRESHOLD

        # F1 Logic
        if has_gt:
            if predicted_has_answer and best_cand_idx == gt_idx:
                tp += 1
            else:
                fn += 1  # Missed it or predicted wrong one
                if predicted_has_answer:
                    fp += 1  # Predicted wrong one is also a False Positive
        else:
            if predicted_has_answer:
                fp += 1

    # Calculate F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Final Validation Metric: {f1:.16f}")

    # Failure Analysis: Correlation
    print("Performing failure analysis...")
    # Row-level error for failure analysis
    # Error = |Label - Score|
    val_df["error"] = (val_df["label"] - val_df["score"]).abs()

    correlations = {}
    for col in feature_cols:
        corr = val_df["error"].corr(val_df[col])
        correlations[col] = corr

    print("Correlations between Error and Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:5]:  # Top 5
        print(f"{feat}: {corr:.4f}")

    # 5. Inference on Test Set
    print("Loading test candidates...")
    test_df = data_loader.get_test_candidates(load_cached_data=True)

    print("Predicting test scores...")
    # Check for GPU (Requirement compliance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device check: {device} available")

    test_scores = ranker.predict_scores(test_df)
    test_df["score"] = test_scores

    # 6. Generate Submission
    print("Generating submission...")

    # Create a map of example_id -> best candidate info
    # Sort by score descending to get best first
    test_df_sorted = test_df.sort_values("score", ascending=False)
    # Drop duplicates to keep top 1
    best_candidates = test_df_sorted.drop_duplicates(
        subset=["example_id"], keep="first"
    )

    # Create lookup dictionary
    # example_id -> {candidate_index, score}
    predictions_map = {}
    for _, row in best_candidates.iterrows():
        predictions_map[str(row["example_id"])] = {
            "candidate_index": int(row["candidate_index"]),
            "score": float(row["score"]),
        }

    short_answer_heuristic = ShortAnswerHeuristic()

    # We need to stream the test JSONL to get tokens and text
    submission_rows = []

    processed_ids = set()

    if os.path.exists(PathConfig.TEST_JSONL):
        with open(PathConfig.TEST_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    continue

                ex_id = str(entry["example_id"])
                processed_ids.add(ex_id)

                long_pred_str = ""
                short_pred_str = ""

                # Check if we have a prediction for this ID
                if ex_id in predictions_map:
                    pred_info = predictions_map[ex_id]

                    # Check Long Answer Threshold
                    if pred_info["score"] > ModelConfig.LONG_CONFIDENCE_THRESHOLD:
                        cand_idx = pred_info["candidate_index"]
                        candidates = entry.get("long_answer_candidates", [])

                        if 0 <= cand_idx < len(candidates):
                            cand = candidates[cand_idx]
                            start_token = cand["start_token"]
                            end_token = cand["end_token"]

                            long_pred_str = f"{start_token}:{end_token}"

                            # Short Answer Logic
                            # Get text for heuristic
                            doc_text = entry.get("document_text", "")
                            tokens = doc_text.split()
                            # Safety check on indices
                            if end_token <= len(tokens):
                                cand_text = " ".join(tokens[start_token:end_token])
                                question_text = entry.get("question_text", "")

                                best_sent, sa_score = (
                                    short_answer_heuristic.find_best_sentence(
                                        question_text, cand_text
                                    )
                                )

                                if best_sent:
                                    # Map text back to indices
                                    sent_tokens = (
                                        doc_text.split()
                                    )  # Re-split to be sure
                                    cand_tokens = sent_tokens[start_token:end_token]

                                    best_sent_tokens = best_sent.split()
                                    n_sent = len(best_sent_tokens)
                                    n_cand = len(cand_tokens)

                                    for i in range(n_cand - n_sent + 1):
                                        if (
                                            cand_tokens[i : i + n_sent]
                                            == best_sent_tokens
                                        ):
                                            sa_start = start_token + i
                                            sa_end = sa_start + n_sent
                                            short_pred_str = f"{sa_start}:{sa_end}"

                                            # Check Yes/No
                                            yn = short_answer_heuristic.check_yes_no(
                                                best_sent
                                            )
                                            if yn != "NONE":
                                                short_pred_str = yn
                                            break

                # Add rows
                submission_rows.append([f"{ex_id}_long", long_pred_str])
                submission_rows.append([f"{ex_id}_short", short_pred_str])

    # Create Submission DataFrame
    sub_df = pd.DataFrame(submission_rows, columns=["example_id", "PredictionString"])

    # Save
    print(f"Saving submission to {PathConfig.SUBMISSION_FILE}...")
    sub_df.to_csv(PathConfig.SUBMISSION_FILE, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
