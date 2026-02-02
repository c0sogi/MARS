import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.stats


# 1. Suppress tqdm before importing library modules that use it
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm

tqdm.tqdm = silent_tqdm

# 2. Imports from provided library
from library.config import Config
from library.utils import set_seed, levenshtein_distance, run_length_encoding
from library.data_loader import get_dataloaders
from library.model import RHCKN
from library.train import (
    train_one_epoch,
    validate,
    get_loss_criterion,
    aggregate_predictions,
)
from library.predict import generate_predictions


def main():
    # ==========================================
    # Setup & Configuration
    # ==========================================
    # Initialize Config and directories
    Config.setup()

    # Override Config for Fast Baseline
    Config.NUM_EPOCHS = 10  # Reduced from 50 for speed
    # We keep batch size and other params as defined in Config to ensure stability

    # Set seeds
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # Data Loading
    # ==========================================
    # Load cached data if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # ==========================================
    # Model Initialization
    # ==========================================
    model = RHCKN().to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Criteria
    ce_criterion, smooth_criterion = get_loss_criterion(device)

    # ==========================================
    # Training Loop
    # ==========================================
    best_score = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, smooth_criterion, device
        )

        # Validate
        val_score = validate(model, val_loader, device)

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            break

    # ==========================================
    # Final Validation & Metrics
    # ==========================================
    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Compute final metric on validation set
    final_metric = validate(model, val_loader, device)
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # Failure Analysis
    # ==========================================
    # Aggregate predictions to get per-sample details
    val_aggregated = aggregate_predictions(val_loader, model, device)

    errors = []
    lengths = []
    num_gestures = []

    for s_idx, data in val_aggregated.items():
        # 1. Compute Error (Levenshtein)
        frame_preds = np.argmax(data["probs"], axis=1)
        pred_seq = run_length_encoding(
            frame_preds,
            min_duration=Config.MIN_GESTURE_DURATION,
            background_class=Config.BACKGROUND_CLASS_ID,
        )
        gt_frame_labels = data["gt_labels"]
        gt_seq = run_length_encoding(
            gt_frame_labels,
            min_duration=1,
            background_class=Config.BACKGROUND_CLASS_ID,
        )
        dist = levenshtein_distance(pred_seq, gt_seq)

        # 2. Extract Features
        # Sequence length (frames)
        seq_len = len(gt_frame_labels)
        # Number of gestures in GT
        n_gestures = len(gt_seq)

        errors.append(dist)
        lengths.append(seq_len)
        num_gestures.append(n_gestures)

    # Compute Correlations
    if len(errors) > 1:
        corr_len, _ = scipy.stats.pearsonr(errors, lengths)
        corr_num, _ = scipy.stats.pearsonr(errors, num_gestures)

        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")
        print(f"Correlation (Error vs Num Gestures): {corr_num:.4f}")
    else:
        print("Insufficient validation samples for failure analysis.")

    # ==========================================
    # Submission
    # ==========================================
    THRESHOLD = 0.2251
    if final_metric < THRESHOLD:
        generate_predictions(model, test_loader, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
