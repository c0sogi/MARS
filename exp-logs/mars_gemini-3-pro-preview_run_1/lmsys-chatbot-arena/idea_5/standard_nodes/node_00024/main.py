import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_score
from library.train import run_training
from library.inference import generate_predictions
from library.data_processing import get_dataloaders
from library.model import SiameseDeberta


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("========================================")
    print("      FAST BASELINE ORCHESTRATION       ")
    print("========================================")

    # Set seed for reproducibility
    seed_everything(Config.seed)

    # Override Config for a Fast Baseline
    Config.epochs = 1  # Limit epochs to ensure execution within 2 hours
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Limit maximum number of training samples as requested
    # We read the original metadata, subsample it, and save to working dir
    # This ensures we don't modify input/metadata but still train on less data
    print("Subsampling training data for fast execution...")
    try:
        df_train = pd.read_csv(Config.train_path)
        # Sample 25,000 rows (approx 60%) or full length if smaller
        # This balances speed with enough data to potentially pass the threshold
        n_samples = min(25000, len(df_train))
        df_train_sub = df_train.sample(n=n_samples, random_state=Config.seed)

        sub_train_path = os.path.join(Config.working_dir, "train_subsampled.csv")
        os.makedirs(Config.working_dir, exist_ok=True)
        df_train_sub.to_csv(sub_train_path, index=False)

        # Update Config to point to the subsampled data
        Config.train_path = sub_train_path
        print(f"Training data set to {sub_train_path} ({n_samples} rows).")
    except Exception as e:
        print(f"Warning: Could not subsample data ({e}). Using full dataset.")

    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    print("\n[Step 1] Executing Training Pipeline...")
    # This runs the training loop and saves the best model to Config.model_save_path
    run_training()

    # -------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 2] Performing Validation & Failure Analysis...")

    # Load Validation Data
    # We use get_dataloaders with load_cached_data=True to reuse processed arrays
    # Note: get_dataloaders returns (train, val, test) loaders
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the Trained Model
    model = SiameseDeberta()
    model.to(Config.device)

    if os.path.exists(Config.model_save_path):
        state_dict = torch.load(Config.model_save_path, map_location=Config.device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {Config.model_save_path}")
        return

    model.eval()

    # Containers for analysis
    all_losses = []
    all_features = []
    all_preds = []
    all_targets = []

    # Loss function for per-sample analysis
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids_a = batch["input_ids_a"].to(Config.device)
            attention_mask_a = batch["attention_mask_a"].to(Config.device)
            input_ids_b = batch["input_ids_b"].to(Config.device)
            attention_mask_b = batch["attention_mask_b"].to(Config.device)
            scalar_features = batch["scalar_features"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # Forward Pass
            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )

            # 1. Calculate Per-Sample Loss for Failure Analysis
            loss_per_sample = criterion_none(logits, labels)
            all_losses.append(loss_per_sample.cpu().numpy())

            # 2. Store Features for Correlation
            all_features.append(scalar_features.cpu().numpy())

            # 3. Store Predictions and Targets for Metric
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate results
    all_losses = np.concatenate(all_losses)
    all_features = np.concatenate(all_features, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Final Validation Metric
    final_metric = compute_score(all_targets, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error and Scalar Features
    feature_names = [
        "Char Diff",
        "Char Ratio",
        "Word Diff",
        "Word Ratio",
        "Newline Diff",
        "Newline Ratio",
    ]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    print(f"{'Feature':<20} | {'Correlation':<12}")
    print("-" * 35)

    for i, name in enumerate(feature_names):
        feat_values = all_features[:, i]
        # Compute Pearson Correlation using NumPy
        # Handle potential constant values or NaNs safely
        if np.std(feat_values) == 0 or np.std(all_losses) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, all_losses)[0, 1]

        print(f"{name:<20} | {corr:.4f}")

    # -------------------------------------------------------------------------
    # 4. Conditional Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 1.0115036312379488

    if final_metric < THRESHOLD:
        print(
            f"\n[Step 3] Metric ({final_metric:.6f}) is lower than threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        generate_predictions(load_cached_data=True)
    else:
        print(
            f"\n[Step 3] Metric ({final_metric:.6f}) is NOT lower than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
