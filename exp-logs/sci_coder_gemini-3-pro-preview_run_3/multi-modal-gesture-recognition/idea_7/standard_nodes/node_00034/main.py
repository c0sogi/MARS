import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import logging

# Import provided library modules
from library import config, utils, model, data_loader, train, predict


def main():
    # 1. Setup and Configuration
    utils.set_seed(config.SEED)

    # Override config for fast baseline execution
    config.NUM_EPOCHS = 25

    # Setup logging
    log_path = os.path.join(config.WORKING_DIR, "run.log")
    logger = utils.setup_logger(log_path)

    device = config.DEVICE
    logger.info(f"Running on device: {device}")

    # 2. Data Loading
    logger.info("Loading datasets...")
    # Use cached data for speed
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached=True
    )

    # 3. Training
    logger.info("Initializing Trainer...")
    trainer = train.Trainer(device, logger)

    logger.info("Starting Training...")
    trainer.fit(
        train_loader, val_loader, num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE
    )

    # 4. Final Validation & Failure Analysis
    logger.info("Loading best model for analysis...")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        logger.error("Model file not found. Training might have failed.")
        return

    # Load best model
    trainer.model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=device)
    )
    trainer.model.eval()

    total_edit_distance = 0
    total_gt_gestures = 0

    # Lists for failure analysis
    error_rates = []
    seq_lengths = []
    gt_counts = []

    logger.info("Performing detailed validation analysis...")

    with torch.no_grad():
        for i, (features, labels, sample_ids) in enumerate(val_loader):
            features = features.to(device)
            # labels in validation loader are full sequence labels (1, Time)

            # Perform inference
            # Using the model directly since val_loader has batch_size=1 and full sequences
            # Note: If sequence is very long, we might need sliding window, but for validation
            # usually direct pass or the logic inside predict.sliding_window_inference is safer.
            # We'll use the sliding window function from predict library to be robust.

            probs = predict.sliding_window_inference(
                trainer.model,
                features,
                window_size=config.WINDOW_SIZE,
                stride=config.STRIDE,
                device=device,
            )

            # Decode predictions
            pred_labels = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
            pred_sequence = utils.decode_predictions_to_sequence(
                pred_labels, background_id=config.BACKGROUND_CLASS_ID, min_len=5
            )

            # Decode Ground Truth
            gt_frame_labels = labels.squeeze(0).cpu().numpy()
            gt_sequence = utils.decode_predictions_to_sequence(
                gt_frame_labels, background_id=config.BACKGROUND_CLASS_ID, min_len=1
            )

            # Metrics
            dist = utils.compute_levenshtein(pred_sequence, gt_sequence)
            n_gt = len(gt_sequence)

            total_edit_distance += dist
            total_gt_gestures += n_gt

            # Failure Analysis Data
            seq_len_frames = features.size(1)

            # Avoid division by zero for error rate calculation
            if n_gt > 0:
                err_rate = dist / n_gt
            else:
                # If there were no gestures and we predicted none, error is 0.
                # If we predicted some, error is technically infinite or undefined relative to GT,
                # but for correlation we can treat dist as the error magnitude.
                # Let's skip empty GT sequences for correlation to be safe, or set a proxy.
                err_rate = dist if dist > 0 else 0

            error_rates.append(err_rate)
            seq_lengths.append(seq_len_frames)
            gt_counts.append(n_gt)

    # Compute Final Metric
    final_metric = (
        total_edit_distance / total_gt_gestures if total_gt_gestures > 0 else 0.0
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    if len(error_rates) > 1:
        corr_len, _ = pearsonr(error_rates, seq_lengths)
        corr_count, _ = pearsonr(error_rates, gt_counts)

        print("-" * 30)
        print("Failure Analysis (Correlation with Error Rate):")
        print(f"  Sequence Length (Frames): {corr_len:.4f}")
        print(f"  Number of Gestures (GT):  {corr_count:.4f}")
        print("-" * 30)

    # 5. Submission
    THRESHOLD = 0.225114854517611

    if final_metric < THRESHOLD:
        logger.info(
            f"Metric ({final_metric:.4f}) is better than threshold ({THRESHOLD:.4f}). Generating submission..."
        )
        predict.predict_test_set(load_cached_data=True)
    else:
        logger.info(
            f"Metric ({final_metric:.4f}) did not meet threshold ({THRESHOLD:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
