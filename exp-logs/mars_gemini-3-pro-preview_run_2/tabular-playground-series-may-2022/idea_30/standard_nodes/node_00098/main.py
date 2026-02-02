import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library modules
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.model import RoPESwiGLURMSNet
from library.data import get_dataloaders
from library.train import run_training
from library.inference import predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    # Ensure the submission directory exists and update the path in Config
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    print("Starting execution of runfile.py...")
    print(f"Target Submission Path: {Config.SUBMISSION_PATH}")

    # --------------------------------------------------------------------------
    # 2. Model Training
    # --------------------------------------------------------------------------
    # We run with debug=False to use the full dataset and default epochs (40).
    # This is necessary to achieve the high validation metric threshold required.
    print("\n--- Phase 1: Training ---")
    run_training(debug=False, load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation and Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)

    # Load the validation data loader
    # We rely on get_dataloaders to return the consistent stratified split
    _, val_loader, _ = get_dataloaders(load_cached_data=True, debug=False)

    # Initialize model and load the best weights saved during training
    model = RoPESwiGLURMSNet()
    model.to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Containers for results
    all_preds = []
    all_targets = []
    all_cont_features = []

    # Optimized Inference Loop
    print("Running validation inference...")
    with torch.no_grad():
        for x_cat, x_cont, y in val_loader:
            # Move to GPU
            x_cat = x_cat.to(device, non_blocking=True)
            x_cont = x_cont.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass
            logits = model(x_cat, x_cont)
            preds = torch.sigmoid(logits)

            # Store results on CPU to save GPU memory
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
            all_cont_features.append(x_cont.cpu())

    # Concatenate all batches
    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_targets).numpy().flatten()
    X_cont = torch.cat(all_cont_features).numpy()

    # Compute and Print Final Metric
    final_auc = compute_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation of Error with Input Features
    print("Calculating error correlations...")
    errors = np.abs(y_true - y_pred)

    # Ensure dimensions match (handling potential drop_last issues, though val_loader usually preserves all)
    n_samples = min(len(errors), X_cont.shape[0])
    errors = errors[:n_samples]
    X_cont = X_cont[:n_samples]

    # Create a DataFrame for easy correlation computation
    feat_cols = [f"f_{i:02d}" for i in range(Config.NUM_CONT_FEATURES)]
    analysis_df = pd.DataFrame(X_cont, columns=feat_cols)
    analysis_df["error"] = errors

    # Compute correlations with the error column
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )

    print("\nTop 5 Features correlated with Error:")
    print(correlations.head(5))
    print("\nBottom 5 Features correlated with Error:")
    print(correlations.tail(5))

    # --------------------------------------------------------------------------
    # 4. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Phase 3: Submission ---")
    THRESHOLD = 0.9972336610045187

    if final_auc > THRESHOLD:
        print(
            f"Validation metric ({final_auc}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        # predict() uses Config.SUBMISSION_PATH which we updated at the start
        predict(load_cached_data=True, debug=False)
    else:
        print(
            f"Validation metric ({final_auc}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
