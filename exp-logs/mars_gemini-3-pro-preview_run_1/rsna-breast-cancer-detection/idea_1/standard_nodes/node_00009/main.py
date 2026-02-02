import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.engine import run_training, generate_submission
from library.dataset import get_dataloaders
from library.utils import probabilistic_f1


def main():
    # Ensure reproducibility
    Config.set_seed(Config.SEED)

    print("==================================================")
    print("Breast Cancer Detection Pipeline")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Training
    # ---------------------------------------------------------
    # We limit the training sample size and epochs to ensure the
    # baseline executes quickly within the time constraints.
    # Optimization: Use full dataset and more epochs.
    TRAIN_SAMPLE_SIZE = None
    TRAIN_EPOCHS = 10

    print(
        f"\n[Step 1] Training Model (Samples={TRAIN_SAMPLE_SIZE}, Epochs={TRAIN_EPOCHS})..."
    )
    model = run_training(sample_size=TRAIN_SAMPLE_SIZE, epochs=TRAIN_EPOCHS)

    # ---------------------------------------------------------
    # 2. Validation & Metric
    # ---------------------------------------------------------
    print("\n[Step 2] Performing Validation on Full Validation Set...")

    # Load the full validation set for accurate metric calculation
    _, val_loader, _ = get_dataloaders(load_cached_data=True, sample_size=None)

    device = torch.device(Config.DEVICE)
    model.eval()

    all_targets = []
    all_preds = []

    # Inference Loop (No Gradient Calculation)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu()

            all_targets.append(targets)
            all_preds.append(probs)

    # Concatenate results
    y_true = torch.cat(all_targets).numpy().flatten()
    y_pred = torch.cat(all_preds).numpy().flatten()

    # Compute Probabilistic F1
    val_pf1 = probabilistic_f1(y_true, y_pred)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_pf1}")

    # ---------------------------------------------------------
    # 3. Failure Analysis
    # ---------------------------------------------------------
    print("\n[Step 3] Performing Failure Analysis...")

    # Calculate absolute prediction error
    errors = np.abs(y_true - y_pred)

    # Retrieve metadata from the dataset
    val_dataset = val_loader.dataset
    df_val = val_dataset.df.copy()

    # Align lengths if necessary (though they should match)
    if len(df_val) != len(errors):
        min_len = min(len(df_val), len(errors))
        df_val = df_val.iloc[:min_len]
        errors = errors[:min_len]

    df_val["error"] = errors

    # Features to analyze for correlation with error
    analysis_features = ["age", "implant"]

    # Process 'density' if available (Map A-D to 1-4)
    if "density" in df_val.columns:
        density_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        df_val["density_numeric"] = df_val["density"].map(density_map)
        analysis_features.append("density_numeric")

    print("Correlation between Error Magnitude and Features:")
    for feat in analysis_features:
        if feat in df_val.columns:
            # Drop NaNs for valid correlation
            subset = df_val[[feat, "error"]].dropna()
            if len(subset) > 0:
                corr = subset[feat].corr(subset["error"])
                print(f"  Feature '{feat}': {corr:.4f}")
            else:
                print(f"  Feature '{feat}': Insufficient data")
        else:
            print(f"  Feature '{feat}': Not found in metadata")

    # ---------------------------------------------------------
    # 4. Submission
    # ---------------------------------------------------------
    print("\n[Step 4] Generating Submission for Test Set...")

    # Generate predictions for the full test set (sample_size=None)
    # Only submit if the validation metric is better than the previous baseline.
    BASELINE_SCORE = 0.04436662048101425

    if val_pf1 > BASELINE_SCORE:
        print(
            f"Validation score ({val_pf1:.6f}) exceeds baseline ({BASELINE_SCORE:.6f}). Generating submission."
        )
        generate_submission(model=model, sample_size=None)
    else:
        print(
            f"Validation score ({val_pf1:.6f}) did not exceed baseline ({BASELINE_SCORE:.6f}). Skipping submission."
        )

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
