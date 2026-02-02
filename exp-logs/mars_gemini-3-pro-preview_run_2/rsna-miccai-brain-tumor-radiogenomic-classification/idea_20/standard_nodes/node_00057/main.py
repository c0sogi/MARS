import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, get_device, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import run_training
from library.evaluate import generate_submission


def main():
    # 1. Setup and Configuration
    # Set seeds for reproducibility
    set_seed(Config.SEED)

    # Adjust Config for Fast Baseline execution
    # Reducing epochs to 10 ensures the run completes well within the time limit
    # while allowing sufficient convergence for this dataset size.
    Config.NUM_EPOCHS = 10

    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # 2. Training Pipeline
    print("\n=== Starting Training Pipeline ===")
    # run_training handles the full loop: Data loading, Model init, Training, Checkpointing
    run_training(load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation & Failure Analysis ===")
    device = get_device()

    # Load the best model saved during training
    model = AsymmetricEfficientNet().to(device)
    checkpoint_path = Config.MODEL_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading best model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle state dict loading
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Get Validation Data
    # We ignore train and test loaders here
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Inference on Validation Set
    all_targets = []
    all_preds = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Store results
            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    # Calculate Metric
    val_auc = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load validation metadata to link predictions with features
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure data alignment
    if len(val_df) != len(all_preds):
        print(
            f"Warning: Metadata size ({len(val_df)}) != Prediction size ({len(all_preds)}). Skipping detailed analysis."
        )
    else:
        # Add predictions and errors to dataframe
        val_df["pred"] = all_preds
        val_df["target"] = all_targets
        val_df["error"] = np.abs(val_df["target"] - val_df["pred"])

        # Extract Feature: FLAIR Slice Count
        # We count the number of files in the FLAIR directory for each subject
        # This is a proxy for brain volume/scan resolution
        print("Extracting 'FLAIR_slices' feature for correlation analysis...")
        flair_counts = []
        for idx, row in val_df.iterrows():
            flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
            try:
                # Fast file count
                count = len(
                    [name for name in os.listdir(flair_path) if name.endswith(".dcm")]
                )
            except Exception:
                count = 0
            flair_counts.append(count)

        val_df["flair_slices"] = flair_counts

        # Calculate Correlations
        # 1. Correlation between Error and Target (Class Bias)
        if val_df["error"].std() > 0 and val_df["target"].std() > 0:
            corr_target, _ = pearsonr(val_df["error"], val_df["target"])
            print(f"Correlation (Error vs Target Class): {corr_target:.4f}")
        else:
            print("Correlation (Error vs Target Class): Undefined (Zero Variance)")

        # 2. Correlation between Error and Slice Count (Structural Bias)
        if val_df["error"].std() > 0 and val_df["flair_slices"].std() > 0:
            corr_slices, _ = pearsonr(val_df["error"], val_df["flair_slices"])
            print(f"Correlation (Error vs FLAIR Slice Count): {corr_slices:.4f}")
        else:
            print("Correlation (Error vs FLAIR Slice Count): Undefined (Zero Variance)")

    # 5. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.6254545454545455

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
