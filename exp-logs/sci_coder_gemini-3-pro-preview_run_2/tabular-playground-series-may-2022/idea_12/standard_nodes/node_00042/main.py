import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import GlobalContextTransformerResFunnel
from library.train import run_training, generate_submission


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Override Config for task requirements
    Config.SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set a fixed seed for reproducibility
    seed_everything(Config.SEED)

    print("Configuration configured. Starting pipeline...")

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print(f"Starting training for {Config.EPOCHS} epochs...")
    # run_training handles the loop, early stopping, and saving best_model.pth
    run_training(epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation & Failure Analysis ---")

    device = get_device()

    # Load the best model saved during training
    model = GlobalContextTransformerResFunnel().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get validation loader (utilizing cache)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Inference on Validation Set
    val_preds = []
    val_targets = []
    val_cont_features = []

    with torch.no_grad():
        for batch in val_loader:
            cont = batch["continuous"].to(device)
            seq = batch["sequence"].to(device)
            target = batch["target"].to(device)

            output = model(cont, seq)

            val_preds.append(output.cpu().numpy())
            val_targets.append(target.cpu().numpy())
            val_cont_features.append(cont.cpu().numpy())

    # Concatenate results
    val_preds = np.concatenate(val_preds).flatten()
    val_targets = np.concatenate(val_targets).flatten()
    val_cont_features = np.concatenate(val_cont_features, axis=0)

    # Compute and Print Final Metric
    final_auc = roc_auc_score(val_targets, val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    errors = np.abs(val_targets - val_preds)

    # Reconstruct feature names (f_00 to f_30 excluding f_27)
    feature_cols = [f"f_{i:02d}" for i in range(31)]
    cont_cols = [c for c in feature_cols if c != "f_27"]

    print("\nFailure Analysis: Correlation between Input Features and Error Magnitude")
    correlations = []
    for i, col_name in enumerate(cont_cols):
        # Calculate Pearson correlation
        feat_values = val_cont_features[:, i]
        if np.std(feat_values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, errors)[0, 1]
        correlations.append((col_name, corr))

    # Sort by absolute correlation strength
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<10} {'Correlation':<12}")
    print("-" * 25)
    for name, corr in correlations[:10]:
        print(f"{name:<10} {corr:.6f}")

    # --------------------------------------------------------------------------
    # 4. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9967793385748163

    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation AUC ({final_auc}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
