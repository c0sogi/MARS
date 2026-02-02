import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processing import DataProcessor
from library.dataset import ContactDataset
from library.train_eval import train_model, validate
from library.model import KinematicMLP


def main():
    # 1. Setup & Configuration
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Initialize processor
    processor = DataProcessor()

    # Load training and validation data (utilizing cache if available)
    # The processor handles loading metadata, tracking, and feature engineering
    X_train, y_train, X_val, y_val = processor.get_train_val_datasets(load_cached=True)

    # Cite Lesson 00023: Simplicity Enables Data Scale.
    # We remove the training limit to leverage the full dataset (3.4M rows).
    # The simple MLP architecture is efficient enough to train on the full set within time limits.

    # Create Datasets
    train_dataset = ContactDataset(X_train, y_train)
    val_dataset = ContactDataset(X_val, y_val)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Training
    # train_model handles the training loop, validation per epoch, threshold optimization,
    # and early stopping. It returns the model with the best weights loaded.
    model = train_model(train_loader, val_loader)

    # 4. Final Validation Assessment
    # Load the best threshold optimized during training
    best_threshold = 0.5
    if os.path.exists(Config.THRESHOLD_PATH):
        best_threshold = np.load(Config.THRESHOLD_PATH)[0]

    # Run inference on the full validation set
    # We use BCEWithLogitsLoss for consistency, though we only need probs for MCC
    criterion = nn.BCEWithLogitsLoss()
    avg_loss, val_probs, val_targets = validate(model, val_loader, criterion, device)

    # Compute Final MCC
    val_preds = (val_probs >= best_threshold).astype(int)
    final_mcc = compute_mcc(val_targets, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_targets - val_probs)

    # Correlate error with features from the center frame (t=0)
    # Total frames = 2 * WINDOW_SIZE + 1
    # Center frame index is WINDOW_SIZE
    n_feats_per_step = len(Config.STEP_FEATURES)
    center_frame_idx = Config.WINDOW_SIZE
    start_col = center_frame_idx * n_feats_per_step

    # Extract key features for analysis
    # Feature order in STEP_FEATURES:
    # [DX, DY, VX1, VY1, VX2, VY2, LOG_DIST, CLOSING_SPEED, S1, A1, S2, A2, IS_GROUND]

    # Indices based on Config.STEP_FEATURES
    # LOG_DIST is at index 6
    # CLOSING_SPEED is at index 7
    # S1 is at index 8
    # S2 is at index 10
    # IS_GROUND is at index 12

    feature_data = {
        "log_dist": X_val[:, start_col + 6],
        "closing_speed": X_val[:, start_col + 7],
        "speed_1": X_val[:, start_col + 8],
        "speed_2": X_val[:, start_col + 10],
        "is_ground": X_val[:, start_col + 12],
    }

    analysis_df = pd.DataFrame(feature_data)
    analysis_df["error"] = errors

    # Compute correlation
    corrs = analysis_df.corr()["error"].sort_values(key=abs, ascending=False)
    print("Correlation between Error Magnitude and Key Features (Center Frame):")
    print(corrs.drop("error").to_string())

    # 6. Submission Generation
    TARGET_METRIC = 0.62458462731896

    if final_mcc > TARGET_METRIC:
        print(
            f"\nMetric ({final_mcc}) > Threshold ({TARGET_METRIC}). Generating submission..."
        )

        # Load test data
        X_test, test_ids = processor.get_test_dataset(load_cached=True)
        test_dataset = ContactDataset(X_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        model.eval()
        all_probs = []

        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                logits = model(features)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        # Concatenate and Threshold
        if len(all_probs) > 0:
            all_probs = np.concatenate(all_probs)
            predictions = (all_probs >= best_threshold).astype(int)
        else:
            predictions = np.array([])

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"contact_id": test_ids, "contact": predictions.flatten()}
        )

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_mcc}) <= Threshold ({TARGET_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    main()
