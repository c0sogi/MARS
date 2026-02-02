import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from library.config import Config, seed_everything
from library.dataset import get_dataloader, load_df
from library.model import CatheterModel
from library.utils import (
    calculate_pos_weights,
    train_one_epoch,
    validate,
    generate_submission,
)


def perform_failure_analysis(model, dataloader, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with the presence of specific target labels.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()
    all_preds = []
    all_targets = []

    # Collect predictions and targets
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    if not all_preds:
        print("No validation data found for analysis.")
        return

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Calculate Mean Absolute Error per sample (Error Magnitude)
    # Shape: (N_samples,)
    error_magnitude = np.mean(np.abs(preds_arr - targets_arr), axis=1)

    # Calculate correlation between Error Magnitude and each Target Variable
    print("-" * 40)
    print(f"{'Feature (Target)':<30} | {'Correlation with Error':<20}")
    print("-" * 40)

    correlations = {}
    for i, col_name in enumerate(Config.TARGET_COLS):
        # Ground truth presence of the label
        feature_vector = targets_arr[:, i]

        # Calculate Pearson correlation
        # Handle constant columns (std=0) to avoid NaN
        if np.std(feature_vector) == 0 or np.std(error_magnitude) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feature_vector, error_magnitude)[0, 1]

        correlations[col_name] = corr
        print(f"{col_name:<30} | {corr:.4f}")
    print("-" * 40)


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline requirements
    Config.EPOCHS = 5
    Config.MAX_TRAIN_SAMPLES = 5000  # Limit training data for speed

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader("valid", batch_size=Config.BATCH_SIZE, shuffle=False)

    # Calculate class weights for loss function
    pos_weights = calculate_pos_weights(load_cached_data=True, device=device)

    # 3. Model Initialization
    print("Initializing Model...")
    model = CatheterModel(pretrained=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, pos_weights
        )

        # Validate
        avg_auc, auc_scores = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {train_loss:.4f} - Val AUC: {avg_auc:.4f}"
        )

        # Checkpoint
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! ({best_auc:.4f})")

    # 5. Final Evaluation & Metric
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Re-run validation on full validation set (though val_loader is already full here)
    final_auc, final_scores = validate(model, val_loader, device)

    # REQUIRED: Print Final Validation Metric
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    print("\nGenerating Submission...")
    test_loader = get_dataloader("test", batch_size=Config.BATCH_SIZE, shuffle=False)
    generate_submission(model, test_loader, device)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
