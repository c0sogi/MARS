import sys
import os
import torch
import numpy as np
import pandas as pd
import scipy.stats
import logging

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import config
from library.dataset import get_dataloaders
from library.engine import Trainer
from library.utils import (
    setup_logger,
    process_predictions_for_submission,
    levenshtein_distance,
)


def main():
    # 1. Setup
    logger = setup_logger()
    config.set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    # Using defaults from config (Batch Size 32)
    # load_cached_data=True ensures we use preprocessed .npz files if available
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Training
    logger.info("Initializing Trainer...")
    trainer = Trainer(device=device)

    logger.info("Starting Training...")
    # We use the epochs defined in config (50) which is sufficient and fast enough on A100
    trainer.fit(train_loader, val_loader)

    # 4. Final Validation Evaluation
    # Load the best model weights to ensure accurate metric calculation
    if os.path.exists(config.MODEL_SAVE_PATH):
        trainer.model.load_state_dict(
            torch.load(config.MODEL_SAVE_PATH, map_location=device)
        )

    logger.info("Computing Final Validation Metric...")
    final_metric = trainer.validate(val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    logger.info("Starting Failure Analysis on Validation Set...")
    trainer.model.eval()

    # Reconstruct probabilities for validation set
    val_probs = trainer._reconstruct_sequences(val_loader)
    val_dataset = val_loader.dataset

    errors = []
    durations = []
    num_gestures = []

    for i, probs in enumerate(val_probs):
        # Decode predictions
        frame_preds = torch.argmax(probs, dim=1).cpu().numpy()
        hyp_list = process_predictions_for_submission(frame_preds, background_class=0)

        # Get Ground Truth
        gt_frame_labels = val_dataset.samples[i]["labels"]
        ref_list = process_predictions_for_submission(
            gt_frame_labels, background_class=0
        )

        # Calculate Error (Levenshtein Distance)
        dist = levenshtein_distance(hyp_list, ref_list)

        # Collect Metrics
        errors.append(dist)
        durations.append(len(gt_frame_labels))  # Sequence length in frames
        num_gestures.append(len(ref_list))  # Complexity

    # Compute Correlations
    if len(errors) > 1:
        corr_duration, _ = scipy.stats.pearsonr(errors, durations)
        corr_complexity, _ = scipy.stats.pearsonr(errors, num_gestures)

        print("\nFailure Analysis - Error Correlation:")
        print(f"Correlation (Error vs Sequence Duration): {corr_duration:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_complexity:.4f}")
    else:
        print("\nInsufficient validation samples for correlation analysis.")

    # 6. Submission Generation
    # Threshold check as per requirements
    THRESHOLD = 0.2251

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        trainer.generate_submission(test_loader)
        logger.info(f"Submission saved to {config.SUBMISSION_FILE}")
    else:
        logger.warning(
            f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
