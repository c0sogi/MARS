import os
import json
import torch
import numpy as np
import pandas as pd
import time
from library.config import Config
from library.trainer import RankerTrainer, ReaderTrainer
from library.inference import QuestionAnsweringPredictor


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


def train_models(config):
    print("--- Starting Model Training ---")

    # Train Ranker
    print("\n[Ranker] Starting training...")
    ranker_trainer = RankerTrainer(config)
    ranker_trainer.train(load_cached_data=False)

    # Train Reader
    print("\n[Reader] Starting training...")
    reader_trainer = ReaderTrainer(config)
    reader_trainer.train(load_cached_data=False)

    print("Training phase completed.")


def validate_and_analyze(config):
    print("\n--- Starting Validation & Failure Analysis ---")

    if not os.path.exists(config.VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {config.VAL_METADATA_PATH}")
        return

    val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    predictor = QuestionAnsweringPredictor(config)

    # Metrics
    tp = 0
    fp = 0
    fn = 0

    # Analysis Data
    analysis_records = []

    print(f"Validating on {len(val_meta)} examples...")
    start_time = time.time()

    with open(config.TRAIN_FILE, "rb") as f:
        for idx, row in val_meta.iterrows():
            # Load Data
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            # --- Ground Truth Extraction ---
            annotations = data.get("annotations", [])
            if not annotations:
                continue

            ann = annotations[0]  # Use first annotation

            # Long Answer GT
            la = ann["long_answer"]
            gt_long = (
                f"{la['start_token']}:{la['end_token']}"
                if la["start_token"] != -1
                else ""
            )

            # Short Answer GT
            gt_short = ""
            if ann["yes_no_answer"] != "NONE":
                gt_short = ann["yes_no_answer"]
            elif ann["short_answers"]:
                sa = ann["short_answers"][0]
                gt_short = f"{sa['start_token']}:{sa['end_token']}"

            # --- Prediction ---
            try:
                pred_long, pred_short = predictor.predict_single(data)
            except Exception:
                pred_long, pred_short = "", ""

            # --- Scoring (Micro F1 Logic) ---
            # Evaluate Long Answer
            is_long_correct = 0
            if pred_long == gt_long:
                if pred_long != "":  # Both non-empty match
                    tp += 1
                # else: Both empty (TN), ignore for F1
                is_long_correct = 1
            else:
                if pred_long != "":  # Predicted something, but wrong or GT was empty
                    fp += 1
                else:  # Predicted empty, but GT was not
                    fn += 1

            # Evaluate Short Answer
            is_short_correct = 0
            if pred_short == gt_short:
                if pred_short != "":
                    tp += 1
                is_short_correct = 1
            else:
                if pred_short != "":
                    fp += 1
                else:
                    fn += 1

            # --- Collect Analysis Data ---
            q_len = len(data.get("question_text", "").split())
            doc_len = len(data.get("document_text", "").split())

            analysis_records.append(
                {
                    "q_len": q_len,
                    "doc_len": doc_len,
                    "long_error": 1 - is_long_correct,
                    "short_error": 1 - is_short_correct,
                    "has_long_gt": 1 if gt_long else 0,
                    "has_short_gt": 1 if gt_short else 0,
                }
            )

            if (idx + 1) % 5000 == 0:
                print(f"Processed {idx + 1} samples...")

    # --- Compute Final Metric ---
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"Final Validation Metric: {f1}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis Report ---")
    df = pd.DataFrame(analysis_records)

    # Correlation Analysis
    if not df.empty:
        correlations = df[["long_error", "short_error", "q_len", "doc_len"]].corr()
        print("Correlation between Error and Input Features:")
        print(correlations[["long_error", "short_error"]].loc[["q_len", "doc_len"]])

        # Systematic Error Patterns
        print("\nSystematic Error Patterns:")
        long_err_rate = df[df["has_long_gt"] == 1]["long_error"].mean()
        short_err_rate = df[df["has_short_gt"] == 1]["short_error"].mean()
        print(f"Error Rate on samples with Long Answer GT: {long_err_rate:.4f}")
        print(f"Error Rate on samples with Short Answer GT: {short_err_rate:.4f}")
    else:
        print("No validation data processed.")


def main():
    # Initialize Config
    config = Config()

    # --- Fast Baseline Configuration ---
    config.DEBUG_SAMPLE_SIZE = 5000  # Limit training samples
    config.EPOCHS = 2  # Limit epochs
    config.BATCH_SIZE = 32  # Ensure memory safety
    config.NUM_WORKERS = 2  # Optimize for environment

    set_seed(config.SEED)

    # 1. Train Models
    train_models(config)

    # 2. Validate and Analyze
    # Note: Validation uses the full validation set as per requirements,
    # ignoring DEBUG_SAMPLE_SIZE logic inside the custom validation loop.
    validate_and_analyze(config)

    # 3. Generate Submission
    print("\n--- Generating Submission ---")
    # Reset debug size for submission to ensure full test set processing if needed
    # However, config is shared. We set it to None to process all test data.
    config.DEBUG_SAMPLE_SIZE = None

    predictor = QuestionAnsweringPredictor(config)
    predictor.generate_submission()
    print("Process completed successfully.")


if __name__ == "__main__":
    main()
