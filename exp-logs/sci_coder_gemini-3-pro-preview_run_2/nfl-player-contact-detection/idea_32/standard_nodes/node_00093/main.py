import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_processor import DataProcessor
from library.dataset import ContactDataset
from library.trainer import Trainer


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline
    Config.EPOCHS = 5  # Reduce epochs for speed

    print("Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading
    processor = DataProcessor()

    # --- Load Training Data (Subsampled for Speed) ---
    print("\nLoading Training Data...")
    # Get processed arrays from DataProcessor
    (X_A, X_B, X_C, X_Vis), y_train, _ = processor.get_data(
        mode="train", load_cached_data=True
    )

    # Subsample to ensure fast training (limit to 500k samples)
    MAX_TRAIN_SAMPLES = 500000
    if len(y_train) > MAX_TRAIN_SAMPLES:
        print(
            f"  Subsampling training data from {len(y_train)} to {MAX_TRAIN_SAMPLES}..."
        )
        indices = np.random.choice(len(y_train), MAX_TRAIN_SAMPLES, replace=False)
        X_A = X_A[indices]
        X_B = X_B[indices]
        X_C = X_C[indices]
        X_Vis = X_Vis[indices]
        y_train = y_train[indices]

    train_dataset = ContactDataset((X_A, X_B, X_C, X_Vis), labels=y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # --- Load Validation Data (Full) ---
    print("Loading Validation Data...")
    (X_A_val, X_B_val, X_C_val, X_Vis_val), y_val, val_ids = processor.get_data(
        mode="validation", load_cached_data=True
    )

    val_dataset = ContactDataset((X_A_val, X_B_val, X_C_val, X_Vis_val), labels=y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    # 3. Training
    print("\nInitializing Trainer...")
    trainer = Trainer(device=Config.DEVICE)

    print("Starting Training...")
    # Explicitly pass epochs to override default argument if necessary
    best_threshold = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Final Validation Metric
    print("\nComputing Final Validation Metric...")
    _, val_y_true, val_probs = trainer.validate(val_loader)
    val_preds = (val_probs >= best_threshold).astype(int)
    final_mcc = compute_mcc(val_y_true, val_preds)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate error magnitude (absolute difference)
    errors = np.abs(val_y_true - val_probs)

    # Calculate correlations with mean feature values of each group
    # This helps identify if errors are driven by geometry, motion, dynamics, or visual noise
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "mean_geometry": np.mean(X_A_val, axis=1),
            "mean_motion": np.mean(X_B_val, axis=1),
            "mean_dynamics": np.mean(X_C_val, axis=1),
            "mean_visual": np.mean(X_Vis_val, axis=1),
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Feature Groups:")
    print(correlations)

    # 6. Submission
    TARGET_METRIC = 0.6634847318478787

    if final_mcc > TARGET_METRIC:
        print(f"\nMetric {final_mcc} > {TARGET_METRIC}. Generating Submission...")

        print("Loading Test Data...")
        (X_A_test, X_B_test, X_C_test, X_Vis_test), _, test_ids = processor.get_data(
            mode="test", load_cached_data=True
        )

        test_dataset = ContactDataset(
            (X_A_test, X_B_test, X_C_test, X_Vis_test), labels=None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        trainer.generate_submission(test_loader, test_ids, best_threshold)
    else:
        print(
            f"\nMetric {final_mcc} <= {TARGET_METRIC}. Skipping Submission generation."
        )


if __name__ == "__main__":
    main()
