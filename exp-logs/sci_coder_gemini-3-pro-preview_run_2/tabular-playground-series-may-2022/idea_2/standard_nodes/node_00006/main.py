import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import os

# Import from provided library
from library.config import (
    seed_everything,
    SUBMISSION_SAVE_PATH,
    NUM_CONTINUOUS_FEATURES,
    MODEL_SAVE_PATH,
)
from library.data_loader import get_dataloaders
from library.trainer import Trainer
from library.utils import get_device


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # --------------------------------------------------------------------------
    # 2. Model Training
    # --------------------------------------------------------------------------
    print("\n--- Starting Training Phase ---")
    trainer = Trainer()

    # We limit epochs to 15 to ensure a fast baseline execution as requested.
    # The DCN-V2 architecture converges relatively quickly on this data.
    test_loader = trainer.fit(num_epochs=15, load_cached_data=True)

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n--- Starting Validation Phase ---")

    # Retrieve the validation loader.
    # trainer.fit() returns test_loader, so we fetch val_loader explicitly.
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    trainer.model.eval()

    all_targets = []
    all_preds = []
    all_cont_features = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            cat = batch["cat"].to(device)
            cont = batch["cont"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            logits = trainer.model(cat, cont)
            probs = torch.sigmoid(logits)

            # Store results (move to CPU for numpy operations)
            all_targets.append(target.cpu().numpy())
            all_preds.append(probs.cpu().numpy())
            all_cont_features.append(cont.cpu().numpy())

    # Concatenate batches
    y_true = np.concatenate(all_targets).flatten()
    y_pred = np.concatenate(all_preds).flatten()
    X_cont = np.vstack(all_cont_features)

    # Calculate Final Metric
    final_auc = roc_auc_score(y_true, y_pred)
    print(f"Final Validation Metric: {final_auc}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_true - y_pred)

    print("Calculating correlation between error magnitude and input features...")
    correlations = []

    # Iterate over continuous features to find correlation with error
    # Note: X_cont corresponds to features f_00 to f_30 (excluding f_27)
    # The indices in X_cont are sequential 0..29
    for i in range(X_cont.shape[1]):
        feature_vals = X_cont[:, i]

        # Pearson correlation
        # Handle potential constant features to avoid division by zero
        if np.std(feature_vals) > 1e-9 and np.std(errors) > 1e-9:
            corr = np.corrcoef(feature_vals, errors)[0, 1]
        else:
            corr = 0.0

        correlations.append((i, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 features correlated with prediction error:")
    for idx, corr in correlations[:5]:
        print(f"Feature Index {idx}: Correlation = {corr:.6f}")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.9948596381822921

    if final_auc > THRESHOLD:
        print(f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}).")
        trainer.generate_submission(test_loader, output_path=SUBMISSION_SAVE_PATH)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({THRESHOLD})."
        )
        print("Submission file will NOT be generated.")


if __name__ == "__main__":
    main()
