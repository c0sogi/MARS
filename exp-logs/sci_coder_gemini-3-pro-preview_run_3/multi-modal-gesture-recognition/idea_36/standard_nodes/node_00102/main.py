import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import provided library modules
from library import config, utils, data_loader, trainer


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    # Set seeds for reproducibility
    utils.set_seed(config.SEED)

    # Override configuration for a fast baseline run
    # Limiting epochs to ensure execution finishes well within time limits
    config.NUM_EPOCHS = 15

    # Ensure output directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    # Load loaders with caching enabled (default in data_loader)
    # debug=False ensures we use the full dataset, but low epochs keep it fast
    train_loader, val_loader, test_loader = data_loader.get_loaders(debug=False)

    # Access underlying datasets for analysis and inference
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    # ==========================================
    # 3. Training
    # ==========================================
    print("Initializing trainer...")
    model_trainer = trainer.Trainer()

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    model_trainer.fit(train_loader, val_loader, val_dataset, epochs=config.NUM_EPOCHS)

    # ==========================================
    # 4. Validation Reporting
    # ==========================================
    # Retrieve the best score recorded during training
    final_metric = model_trainer.best_score
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    print("Performing failure analysis on validation set...")

    # Generate predictions for the validation set using the best model
    # We use the trainer's predict method which loads the best model weights
    val_preds = model_trainer.predict(val_dataset)

    errors = []
    seq_lengths = []
    num_gestures_list = []

    # Iterate through validation sequences to correlate errors with features
    for i in range(len(val_dataset.sequences)):
        # Get metadata for the current sequence
        meta_row = val_dataset.metadata.iloc[i]
        sample_id = meta_row["sample_id"]

        # Get Ground Truth IDs from parsed labels
        gt_labels = meta_row["parsed_labels"]
        gt_ids = [l["id"] for l in gt_labels]

        # Get Predicted IDs
        pred_ids = val_preds.get(sample_id, [])

        # Compute Levenshtein Distance (Error Magnitude)
        dist = utils.levenshtein_distance(pred_ids, gt_ids)

        # Extract Features
        # 1. Sequence Length (frames)
        seq_data = val_dataset.sequences[i]
        num_frames = seq_data["label"].shape[0]

        # 2. Number of Ground Truth Gestures
        n_gestures = len(gt_ids)

        errors.append(dist)
        seq_lengths.append(num_frames)
        num_gestures_list.append(n_gestures)

    # Compute Correlations
    if len(errors) > 1:
        # Correlation between Error and Sequence Length
        corr_len, _ = pearsonr(errors, seq_lengths)
        print(f"Correlation (Error vs Sequence Length): {corr_len:.4f}")

        # Correlation between Error and Number of Gestures
        corr_gest, _ = pearsonr(errors, num_gestures_list)
        print(f"Correlation (Error vs Num Gestures): {corr_gest:.4f}")
    else:
        print("Insufficient validation samples for correlation analysis.")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    threshold = 0.2251

    if final_metric < threshold:
        print(
            f"Validation metric {final_metric} meets threshold ({threshold}). Generating submission..."
        )

        # Generate predictions for test set
        test_preds = model_trainer.predict(test_dataset)

        # Format lines for CSV
        lines = []
        for sample_id, pred_ids in test_preds.items():
            # Format: SessionID,Label1,Label2,...
            # If pred_ids is empty, it will just be "SessionID," which is valid (no gestures)
            pred_str = ",".join(map(str, pred_ids))
            line = f"{sample_id},{pred_str}"
            lines.append(line)

        # Write to file
        with open(config.SUBMISSION_PATH, "w") as f:
            f.write("\n".join(lines))

        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
