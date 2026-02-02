import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.train import run_training
from library.predict import predict
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.utils import seed_everything, get_score


def main():
    # --- 1. Configuration Setup ---
    # Adjust Config for a fast baseline that fits within the time limit
    # while maintaining enough capacity to reach the target metric.
    Config.EPOCHS = 6  # Reduced epochs for speed (EfficientNet converges fast)
    Config.BATCH_SIZE = 8  # Ensure stability

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print("=== Configuration ===")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # --- 2. Training ---
    print("\n=== Starting Training Pipeline ===")
    # run_training handles data loading, model init, training loop, and saving best_model.pth
    run_training()

    # --- 3. Validation Assessment ---
    print("\n=== Starting Validation Assessment ===")
    device = Config.DEVICE

    # Load validation metadata
    if not os.path.exists(Config.VAL_METADATA):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA}"
        )

    df_val = pd.read_csv(Config.VAL_METADATA)
    print(f"Validation samples: {len(df_val)}")

    # Create Validation Dataset and Loader
    # We use 'train' mode to get labels, but 'valid' transforms
    val_dataset = CatheterDataset(
        df_val, transforms=get_transforms(data="valid"), mode="train"
    )

    # Double batch size for inference as no gradients are stored
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Load the best model trained in step 2
    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model = CatheterModel(pretrained=False)
    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run Inference
    all_preds = []
    all_targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Final Metric
    final_metric = get_score(all_targets, all_preds)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Calculate Mean Absolute Error per sample (averaged across all classes)
    # Shape: (N_samples,)
    sample_errors = np.mean(np.abs(all_targets - all_preds), axis=1)

    print("Correlation between Error Magnitude and Input Features:")

    # Feature 1: Label Complexity (Total number of catheters/lines present)
    label_counts = np.sum(all_targets, axis=1)
    corr_count, _ = pearsonr(sample_errors, label_counts)
    print(f"  Error vs. Total Active Labels: {corr_count:.6f}")

    # Feature 2: Presence of specific classes
    # We correlate the error with the binary presence of each class
    print("  Error vs. Specific Class Presence:")
    for i, col_name in enumerate(Config.TARGET_COLS):
        # Only calculate if there is variance in the target
        if np.std(all_targets[:, i]) > 0:
            corr, _ = pearsonr(sample_errors, all_targets[:, i])
            print(f"    {col_name}: {corr:.6f}")
        else:
            print(f"    {col_name}: N/A (No positive samples)")

    # --- 5. Submission Generation ---
    THRESHOLD = 0.9398508707740129

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict(
            batch_size=Config.BATCH_SIZE * 2,
            device=device,
            metadata_path=Config.TEST_METADATA,
            model_path=Config.BEST_MODEL_PATH,
            output_path=Config.SUBMISSION_FILE,
        )
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
