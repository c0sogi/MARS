import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import sys
import shutil

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


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # -------------------------------------------------------------------------
    logger.info("Configuring pipeline for fast baseline...")
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SIZE = 5000  # Small subset for fast training
    Config.BATCH_SIZE = 32  # Conservative batch size

    # Clean cache to ensure we start with debug data
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
    model.eval()

    # Inference Loop for Validation
    la_preds = []
    la_targets = []

    # Features for failure analysis
    q_lengths = []
    c_lengths = []

    logger.info("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            q_input = batch["q_input"].to(device)
            c_input = batch["c_input"].to(device)
            l_target = batch["label_long"].to(device)

            # Forward pass
            la_logits, _, _ = model(q_input, c_input)

            # Predictions
            la_prob = torch.sigmoid(la_logits).squeeze(-1).cpu().numpy()
            la_preds.extend(la_prob)
            la_targets.extend(l_target.cpu().numpy())

            # Extract features (length of non-padding tokens)
            q_lens = torch.sum(q_input != 0, dim=1).cpu().numpy()
            c_lens = torch.sum(c_input != 0, dim=1).cpu().numpy()
            q_lengths.extend(q_lens)
            c_lengths.extend(c_lens)

    # Compute Final Metric (Micro F1 on Long Answer Classification)
    # Using the threshold defined in Config
    la_preds_arr = np.array(la_preds)
    la_targets_arr = np.array(la_targets).astype(int)
    la_preds_bin = (la_preds_arr >= Config.TAU_LONG).astype(int)

    tp = np.sum((la_preds_bin == 1) & (la_targets_arr == 1))
    fp = np.sum((la_preds_bin == 1) & (la_targets_arr == 0))
    fn = np.sum((la_preds_bin == 0) & (la_targets_arr == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {f1}")

    # Failure Analysis
    logger.info("Performing failure analysis...")
    # Error magnitude: |Predicted_Prob - Target|
    errors = np.abs(la_preds_arr - la_targets_arr)

    # Correlations
    if len(errors) > 1:
        corr_q, _ = pearsonr(errors, q_lengths)
        corr_c, _ = pearsonr(errors, c_lengths)
        print(f"Correlation Error vs Question Length: {corr_q}")
        print(f"Correlation Error vs Candidate Length: {corr_c}")
    else:
        print("Insufficient data for correlation analysis.")

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

    # Save
    format_submission_file(submission_dict, Config.SUBMISSION_FILE)
    logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
