import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config, seed_everything
from library.train import Trainer
from library.utils import load_checkpoint, kl_divergence
from library.data import get_dataloaders
from library.model import MultiResNetwork


def run():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    seed_everything(Config.SEED)

    # Patch Config for fast baseline execution
    # A100 GPU allows larger batch size. 3 Epochs is sufficient for a baseline.
    Config.EPOCHS = 3
    Config.BATCH_SIZE = 64

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Training
    # ==========================================
    print("\n=== Starting Training Phase ===")
    # Trainer handles data loading, caching, and training loop
    trainer = Trainer(debug=False)
    trainer.fit()

    # ==========================================
    # 3. Validation & Metric
    # ==========================================
    print("\n=== Starting Validation Phase ===")

    # Initialize model structure
    model = MultiResNetwork().to(device)

    # Load the best model saved by Trainer
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    # Get dataloaders (will use cached data from training phase)
    _, val_loader, test_loader = get_dataloaders(debug=False)

    all_preds = []
    all_targets = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            x_a, x_b = inputs
            x_a = x_a.to(device)
            x_b = x_b.to(device)

            # Forward pass
            outputs = model((x_a, x_b))

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Metric
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # Calculate KL Divergence per sample for analysis
    # KL = sum(p * log(p/q))
    # Handle case where p=0 safely
    terms = all_targets * np.log(all_targets / preds_clipped)
    terms[all_targets == 0] = 0.0
    kl_per_sample = np.sum(terms, axis=1)

    # Final scalar metric
    final_metric = np.mean(kl_per_sample)

    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n=== Failure Analysis ===")
    try:
        val_df = pd.read_csv(Config.VAL_CSV)

        # Ensure alignment
        if len(val_df) == len(kl_per_sample):
            val_df["error"] = kl_per_sample

            features_to_check = [
                "total_votes",
                "eeg_label_offset_seconds",
                "spectogram_label_offset_seconds",
            ]

            print("Correlation between Error (KL) and Metadata features:")
            for feat in features_to_check:
                if feat in val_df.columns:
                    # Drop NaNs for correlation calculation
                    subset = val_df[[feat, "error"]].dropna()
                    if len(subset) > 1:
                        # Calculate Pearson correlation
                        corr = subset[feat].corr(subset["error"])
                        print(f"  {feat}: {corr:.4f}")
        else:
            print(
                f"Warning: Validation DataFrame length ({len(val_df)}) "
                f"does not match predictions ({len(kl_per_sample)}). Skipping analysis."
            )
    except Exception as e:
        print(f"Error during failure analysis: {e}")

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.8169508603799445

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is below threshold ({threshold}). Generating submission..."
        )

        test_preds = []

        print("Running inference on test set...")
        with torch.no_grad():
            for inputs in test_loader:
                # Test loader returns (inputs) because targets are None
                # Inputs is tuple (x_a, x_b)
                x_a, x_b = inputs
                x_a = x_a.to(device)
                x_b = x_b.to(device)

                outputs = model((x_a, x_b))
                test_preds.append(outputs.cpu().numpy())

        if test_preds:
            test_preds = np.concatenate(test_preds)

            # Load test metadata to get eeg_ids
            test_df = pd.read_csv(Config.TEST_CSV)

            if len(test_df) != len(test_preds):
                print(
                    f"Error: Test predictions length ({len(test_preds)}) "
                    f"mismatches metadata ({len(test_df)})."
                )
            else:
                # Create submission DataFrame
                sub_df = pd.DataFrame(test_preds, columns=Config.CLASS_NAMES)
                sub_df["eeg_id"] = test_df["eeg_id"]

                # Reorder columns: eeg_id first
                cols = ["eeg_id"] + Config.CLASS_NAMES
                sub_df = sub_df[cols]

                # Save
                os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
                sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
                print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) >= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    run()
