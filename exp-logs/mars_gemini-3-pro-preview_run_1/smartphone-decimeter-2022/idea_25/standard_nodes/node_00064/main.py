import os
import sys
import random
import numpy as np
import pandas as pd
import torch

# Import from provided library files
from library.config import Config
from library.trainer import Trainer
from library.dataset import get_feature_columns


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def perform_failure_analysis(trainer):
    print("\n--- Failure Analysis ---")
    trainer.model.eval()

    all_errors = []
    all_features = []

    feature_names = get_feature_columns()

    # We need to accumulate features and errors
    # To avoid OOM on large validation sets, we'll process in batches and maybe sample
    # However, for correlation, we need a decent amount of data.

    print("Computing correlations between features and error magnitude...")

    with torch.no_grad():
        for batch_idx, (X, targets, metadata) in enumerate(trainer.val_loader):
            X = X.to(trainer.device)

            # Forward pass
            preds = trainer.model(X)
            pred_high_res = preds[0]  # (B, 2, L)

            # Get targets
            if isinstance(targets, dict):
                gt_high_res = targets["scale_0"].to(trainer.device)
            else:
                gt_high_res = targets.to(trainer.device)

            # Calculate Euclidean error per point: (B, L)
            diff = pred_high_res - gt_high_res
            error = torch.sqrt(torch.sum(diff**2, dim=1))

            # Mask padding
            pad_mask = metadata["pad_mask"].to(trainer.device)  # (B, L)

            # Flatten and filter
            # Permute X to (B, L, C) for flattening
            X_perm = X.permute(0, 2, 1)  # (B, L, C)

            valid_mask = pad_mask.view(-1).bool()

            # Flatten
            error_flat = error.view(-1)[valid_mask]
            X_flat = X_perm.reshape(-1, X_perm.shape[-1])[valid_mask]

            # Move to CPU numpy
            all_errors.append(error_flat.cpu().numpy())
            all_features.append(X_flat.cpu().numpy())

            # Limit analysis size to avoid memory issues (e.g., first 100k points)
            if sum(len(e) for e in all_errors) > 100000:
                break

    if not all_errors:
        print("No validation data found for failure analysis.")
        return

    # Concatenate
    y_err = np.concatenate(all_errors)
    X_feat = np.concatenate(all_features)

    # Create DataFrame
    df_analysis = pd.DataFrame(X_feat, columns=feature_names)
    df_analysis["error_magnitude"] = y_err

    # Compute Correlation
    correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    top_corr = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 Features correlated with Error Magnitude:")
    for feat, corr_val in top_corr.items():
        # Get the sign from the original correlation series
        sign = correlations[feat]
        print(f"  {feat}: {sign:.4f}")


def main():
    # 1. Setup & Configuration
    set_seed(Config.SEED)

    # Override Config for Fast Baseline Execution
    print("Configuring for fast baseline run...")
    Config.EPOCHS = 5  # Limit epochs
    Config.TRAIN_SEQUENCE_LENGTH = 128  # Shorter sequences for faster iteration
    # Config.BATCH_SIZE is already 32, which is reasonable

    # 2. Initialize Trainer
    # load_cached_data=True will use existing parquet files if available,
    # or trigger preprocessing if not.
    trainer = Trainer(load_cached_data=True)

    # 3. Train Model
    trainer.run()

    # 4. Final Validation Assessment
    # Load the best model weights saved during training
    if os.path.exists(trainer.best_model_path):
        print(f"Loading best model from {trainer.best_model_path}")
        trainer.model.load_state_dict(
            torch.load(trainer.best_model_path, map_location=trainer.device)
        )
    else:
        print("Warning: No best model found. Using current model weights.")

    final_metric = trainer.validate()
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(trainer)

    # 6. Submission Generation
    THRESHOLD = 3.7864967500302016

    if final_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"\nValidation metric ({final_metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
