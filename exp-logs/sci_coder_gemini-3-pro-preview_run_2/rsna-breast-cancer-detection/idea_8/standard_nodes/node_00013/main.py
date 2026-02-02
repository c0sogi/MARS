import sys
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch.cuda.amp import autocast

# Import library modules
# Ensure the current directory is in the path to find the library package
sys.path.append(os.getcwd())

from library.config import Config
from library.trainer import Trainer, set_seed, calculate_pf1
from library.inference import generate_submission
from library.dataset import get_dataloaders
from library.model import MultiTaskEfficientNet


def main():
    # --- 1. Setup & Configuration ---
    # Override Config for a fast baseline run
    Config.NUM_EPOCHS = 1  # Run for 1 epoch to satisfy time constraints

    # Set the required submission path
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("=== Starting Fast Baseline Pipeline ===")
    print(f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}")

    # --- 2. Training ---
    # Trainer handles data loading, model init, and training loop
    trainer = Trainer()
    trainer.train()

    # --- 3. Validation & Evaluation ---
    print("\n=== Running Final Validation ===")

    # Load validation data
    # load_cached_data=True utilizes pre-processed parquet files for speed
    _, val_loader, _ = get_dataloaders(load_cached_data=True)
    if val_loader is None:
        print("Error: Validation loader is None. Check metadata generation.")
        sys.exit(1)

    # Load the best model saved by Trainer
    device = torch.device(Config.DEVICE)
    model = MultiTaskEfficientNet(pretrained=False)

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {Config.MODEL_PATH}")
    else:
        print("Error: Model artifact not found. Training may have failed.")
        sys.exit(1)

    model.to(device)
    model.eval()

    # Inference Loop for Validation
    all_preds = []
    all_targets = []
    meta_data_storage = []  # To store metadata for failure analysis

    print("Computing predictions on validation set...")
    with torch.no_grad():
        for images, meta_vec, targets in val_loader:
            images = images.to(device)
            meta_vec = meta_vec.to(device)
            t_cancer = targets["cancer"].to(device)

            # Optimize inference speed with mixed precision
            with autocast():
                outputs = model(images, meta_vec)
                # Primary head probability
                probs = torch.sigmoid(outputs["cancer"])

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(t_cancer.cpu().numpy().flatten())
            meta_data_storage.append(meta_vec.cpu().numpy())

    # Calculate Metric
    val_pf1 = calculate_pf1(all_preds, all_targets)
    # Print exact format required
    print(f"Final Validation Metric: {val_pf1}")

    # --- 4. Failure Analysis ---
    print("\n=== Failure Analysis ===")

    # Stack metadata: Shape (N, 4) -> [age_norm, implant, view_enc, machine_enc]
    if len(meta_data_storage) > 0:
        meta_array = np.vstack(meta_data_storage)

        # Calculate error magnitude
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        error_magnitude = np.abs(preds_arr - targets_arr)

        # Create DataFrame for analysis
        # Features from dataset.py: [age_norm, implant, view_enc, machine_enc]
        features = ["age_norm", "implant", "view_enc", "machine_enc"]

        analysis_df = pd.DataFrame(meta_array, columns=features)
        analysis_df["error"] = error_magnitude

        print(f"Analyzing correlations with error magnitude (N={len(analysis_df)})...")

        for feat in features:
            # Check if feature has variance to avoid warnings
            if analysis_df[feat].nunique() > 1:
                corr, p_val = pearsonr(analysis_df[feat], analysis_df["error"])
                print(
                    f"Feature: {feat:<12} | Correlation with Error: {corr:.6f} (p={p_val:.4f})"
                )
            else:
                print(
                    f"Feature: {feat:<12} | Correlation with Error: N/A (Constant value)"
                )
    else:
        print("No validation data available for failure analysis.")

    # --- 5. Submission ---
    THRESHOLD = 0.044888656586408615

    if val_pf1 > THRESHOLD:
        print(
            f"\nMetric ({val_pf1}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        generate_submission()
    else:
        print(
            f"\nMetric ({val_pf1}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
