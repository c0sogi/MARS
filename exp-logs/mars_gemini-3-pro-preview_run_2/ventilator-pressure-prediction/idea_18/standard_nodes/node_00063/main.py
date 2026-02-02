import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_factory import prepare_data, VentilatorDataset
from library.model_factory import WCMI_BiLSTM
from library.trainer import Trainer


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    (X_train, y_train, u_out_train), (X_val, y_val, u_out_val), (X_test, test_ids) = (
        prepare_data(load_cached_data=True)
    )

    # 3. Fast Baseline Configuration
    # Limit training data size and epochs for quick execution
    TRAIN_SUBSET_SIZE = 20000
    FAST_EPOCHS = 15

    if len(X_train) > TRAIN_SUBSET_SIZE:
        print(
            f"Subsampling training data to {TRAIN_SUBSET_SIZE} breaths for fast baseline."
        )
        X_train = X_train[:TRAIN_SUBSET_SIZE]
        y_train = y_train[:TRAIN_SUBSET_SIZE]
        u_out_train = u_out_train[:TRAIN_SUBSET_SIZE]

    # 4. Dataset & DataLoader Creation
    train_dataset = VentilatorDataset(X_train, y_train, u_out_train)
    val_dataset = VentilatorDataset(X_val, y_val, u_out_val)
    test_dataset = VentilatorDataset(X_test)

    # Auxiliary dataset for validation inference (features only) for failure analysis
    val_dataset_inference = VentilatorDataset(X_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader_inference = DataLoader(
        val_dataset_inference,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 5. Model Initialization
    input_dim = X_train.shape[2]
    print(f"Input feature dimension: {input_dim}")
    model = WCMI_BiLSTM(input_dim).to(device)

    # 6. Training
    trainer = Trainer(model, device)
    print(f"Starting training for {FAST_EPOCHS} epochs...")
    trainer.fit(train_loader, val_loader, epochs=FAST_EPOCHS, lr=Config.LR, patience=5)

    # 7. Validation Metric
    final_metric = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_metric}")

    # 8. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Generate predictions on validation set
    val_preds = trainer.predict(val_loader_inference)

    # Flatten targets and u_out for alignment
    y_val_flat = y_val.flatten()
    u_out_val_flat = u_out_val.flatten()
    X_val_flat = X_val.reshape(-1, input_dim)

    # Calculate absolute errors
    errors = np.abs(val_preds - y_val_flat)

    # Filter for inspiratory phase only (u_out == 0)
    insp_mask = u_out_val_flat == 0

    if np.sum(insp_mask) > 0:
        insp_errors = errors[insp_mask]
        insp_features = X_val_flat[insp_mask]

        # Calculate correlation between error magnitude and each feature
        correlations = {}
        for i in range(input_dim):
            # Check for zero variance to avoid division by zero in correlation
            if np.std(insp_features[:, i]) > 1e-9:
                corr = np.corrcoef(insp_features[:, i], insp_errors)[0, 1]
                correlations[f"Feature_{i}"] = corr
            else:
                correlations[f"Feature_{i}"] = 0.0

        # Sort by absolute correlation strength
        sorted_corrs = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("Top 5 Features correlated with Error Magnitude (Inspiratory Phase):")
        for name, val in sorted_corrs[:5]:
            print(f"{name}: {val:.4f}")
    else:
        print("No inspiratory phase data found for failure analysis.")

    # 9. Submission Logic
    THRESHOLD = 0.1619843989610672
    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        trainer.generate_submission(test_loader, test_ids)
    else:
        print(
            f"\nMetric {final_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
