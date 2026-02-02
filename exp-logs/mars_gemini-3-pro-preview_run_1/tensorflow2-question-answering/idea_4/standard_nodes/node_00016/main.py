import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import shutil
import json

# Import library modules
from library.config import Config, set_seed
from library.training import train_model
from library.inference import run_inference, predict_on_test, select_answers
from library.data_processing import get_dataloaders
from library.model import DAAN
from library.utils import (
    load_checkpoint,
    setup_logger,
    compute_classification_metrics,
    format_submission_file,
)

# Setup Logger
logger = setup_logger("runfile")


def clean_working_dir():
    """Removes cached parquet files to ensure data is regenerated with correct settings."""
    files_to_remove = [
        Config.TRAIN_FLATTENED_PATH,
        Config.VAL_FLATTENED_PATH,
        Config.TEST_FLATTENED_PATH,
    ]
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                logger.info(f"Removed cached file: {f}")
            except OSError as e:
                logger.warning(f"Error removing {f}: {e}")


def compute_official_f1(predictions, metadata_path):
    """
    Computes the official Micro F1 score based on exact string matching.
    Cite debug_lesson_2: Align Validation Metrics with Official Scoring Logic.
    """
    meta = pd.read_parquet(metadata_path)

    tp = 0
    fp = 0
    fn = 0

    # Pre-process ground truth
    # Map example_id -> { 'long': set([str]), 'short': set([str]) }
    gt_map = {}

    for _, row in meta.iterrows():
        eid = row["example_id"]
        try:
            anns = json.loads(row["annotations"])
        except:
            anns = []

        long_gts = set()
        short_gts = set()

        for ann in anns:
            # Long Answer GT
            la = ann.get("long_answer", {})
            if la.get("start_token", -1) != -1:
                long_gts.add(f"{la['start_token']}:{la['end_token']}")

            # Short Answer GT
            # Check yes/no first
            yn = ann.get("yes_no_answer", "NONE")
            if yn in ["YES", "NO"]:
                short_gts.add(yn)
            else:
                # Spans
                sas = ann.get("short_answers", [])
                for sa in sas:
                    short_gts.add(f"{sa['start_token']}:{sa['end_token']}")

        gt_map[eid] = {"long": long_gts, "short": short_gts}

    # Evaluate
    for eid, gts in gt_map.items():
        # Long Answer Evaluation
        pred_long = predictions.get(f"{eid}_long", "")
        gt_long = gts["long"]

        if len(gt_long) > 0:
            if pred_long in gt_long:
                tp += 1
            else:
                fn += 1
                if pred_long != "":
                    fp += 1
        else:
            if pred_long != "":
                fp += 1

        # Short Answer Evaluation
        pred_short = predictions.get(f"{eid}_short", "")
        gt_short = gts["short"]

        if len(gt_short) > 0:
            if pred_short in gt_short:
                tp += 1
            else:
                fn += 1
                if pred_short != "":
                    fp += 1
        else:
            if pred_short != "":
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration for Full Training
    # -------------------------------------------------------------------------
    logger.info("Configuring pipeline for full training run...")
    # Train on full data for better performance and to utilize available time
    Config.DEBUG = False
    Config.EPOCHS = 5
    Config.BATCH_SIZE = 64

    # Clean cache to ensure we start with full data
    clean_working_dir()

    # -------------------------------------------------------------------------
    # 2. Training Phase (on small subset)
    # -------------------------------------------------------------------------
    logger.info("=== Starting Training Phase ===")
    # train_model handles data loading internally based on Config
    train_model(load_cached_data=False)

    # -------------------------------------------------------------------------
    # 3. Validation & Failure Analysis (on FULL validation set)
    # -------------------------------------------------------------------------
    logger.info("=== Starting Validation and Failure Analysis ===")

    # Switch to full data mode for validation and testing
    Config.DEBUG = False

    # Remove cached validation/test files so they are regenerated fully
    if os.path.exists(Config.VAL_FLATTENED_PATH):
        os.remove(Config.VAL_FLATTENED_PATH)
    if os.path.exists(Config.TEST_FLATTENED_PATH):
        os.remove(Config.TEST_FLATTENED_PATH)

    # Load DataLoaders (this will regenerate full val/test sets)
    # We reuse the vocab/embeddings generated during training (cached)
    logger.info("Loading full validation data...")
    _, val_loader, _, embedding_matrix = get_dataloaders(load_cached_data=True)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAAN(embedding_matrix)
    model.to(device)

    epoch, loss = load_checkpoint(Config.MODEL_PATH, model, device=device)
    logger.info(f"Loaded model checkpoint from Epoch {epoch}")

    # --- Official Metric Calculation ---
    logger.info("Running full inference on validation set for Official F1...")

    # Run inference using the pipeline logic to get raw probabilities
    val_preds_df = predict_on_test(model, val_loader, device)

    # Apply thresholds to get prediction strings
    val_submission_dict = select_answers(
        val_preds_df, tau_long=Config.TAU_LONG, tau_short=Config.TAU_SHORT
    )

    # Compute Official F1
    f1 = compute_official_f1(val_submission_dict, Config.VAL_META_PATH)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {f1}")

    # --- Failure Analysis (Correlation) ---
    logger.info("Performing failure analysis...")
    # Load the flattened validation data to get targets
    val_flat_df = pd.read_parquet(Config.VAL_FLATTENED_PATH)

    if len(val_preds_df) == len(val_flat_df):
        la_preds_arr = val_preds_df["la_prob"].values
        la_targets_arr = val_flat_df["label_long"].values

        # Calculate lengths (approximate from text to avoid re-tokenizing)
        q_lengths = val_flat_df["question_text"].apply(lambda x: len(x.split())).values
        c_lengths = val_flat_df["candidate_text"].apply(lambda x: len(x.split())).values

        errors = np.abs(la_preds_arr - la_targets_arr)

        if len(errors) > 1:
            corr_q, _ = pearsonr(errors, q_lengths)
            corr_c, _ = pearsonr(errors, c_lengths)
            print(f"Correlation Error vs Question Length: {corr_q}")
            print(f"Correlation Error vs Candidate Length: {corr_c}")
        else:
            print("Insufficient data for correlation analysis.")
    else:
        logger.warning(
            "Mismatch in validation data lengths. Skipping correlation analysis."
        )

    # -------------------------------------------------------------------------
    # 4. Submission Generation (on FULL test set)
    # -------------------------------------------------------------------------
    logger.info("=== Generating Submission ===")

    # Reload dataloaders to get the test_loader (Config.DEBUG is False)
    # Note: test_flattened.parquet was deleted earlier, so it will be generated fully now if not already
    _, _, test_loader, _ = get_dataloaders(load_cached_data=True)

    # Predict on Test
    raw_predictions_df = predict_on_test(model, test_loader, device)

    # Apply thresholds and formatting
    submission_dict = select_answers(
        raw_predictions_df, tau_long=Config.TAU_LONG, tau_short=Config.TAU_SHORT
    )

    # Save to the configured path (subdirectory)
    format_submission_file(submission_dict, Config.SUBMISSION_FILE)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
    # Legacy copy operation removed (Cite debug_lesson_4: Audit Legacy File Operations)
    # Config.SUBMISSION_FILE already points to root "submission.csv"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
