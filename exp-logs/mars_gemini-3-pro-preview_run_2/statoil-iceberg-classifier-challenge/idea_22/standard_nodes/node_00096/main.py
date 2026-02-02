import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_data, make_dataloaders, make_test_dataloader
from library.engine import train_fold, predict


def main():
    # 1. Setup and Configuration
    Config.setup()

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting Fast Baseline Execution...")

    # 2. Load Data
    # Use cached data if available for speed
    train_data, test_data = get_data(load_cached_data=True)

    # Containers for Out-Of-Fold (OOF) Evaluation
    oof_preds = []
    oof_targets = []
    oof_angles = []

    # Store models and their specific scalers for test inference
    fold_models = []

    # 3. Stratified K-Fold Cross Validation
    for fold in range(Config.NUM_FOLDS):
        # Create DataLoaders for this fold
        # scaler_stats are computed only on the training split of this fold
        train_loader, val_loader, scaler_stats = make_dataloaders(
            train_data, fold_idx=fold
        )

        # Train the model for this fold
        # train_fold handles the training loop, early stopping, and returns the best model
        model = train_fold(fold, train_loader, val_loader, device)

        # Generate OOF predictions for this fold
        val_probs = predict(model, val_loader, device)

        # Retrieve targets and metadata for validation set
        # Note: val_loader is not shuffled, so order matches predict output
        val_targets = val_loader.dataset.labels
        val_angles = val_loader.dataset.angles

        # Store results
        oof_preds.append(val_probs)
        oof_targets.append(val_targets)
        oof_angles.append(val_angles)

        # Store model and stats for ensemble
        fold_models.append((model, scaler_stats))

    # 4. Validation Assessment
    # Concatenate all fold results
    oof_preds = np.concatenate(oof_preds).flatten()
    oof_targets = np.concatenate(oof_targets).flatten()
    oof_angles = np.concatenate(oof_angles).flatten()

    # Calculate Log Loss
    # Clip predictions to prevent infinite loss
    epsilon = 1e-15
    oof_preds_clipped = np.clip(oof_preds, epsilon, 1 - epsilon)
    final_metric = log_loss(oof_targets, oof_preds_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\nFailure Analysis:")
    errors = np.abs(oof_targets - oof_preds)

    # Correlation with Incidence Angle
    corr_angle = np.corrcoef(errors, oof_angles)[0, 1]
    print(f"Correlation between Error and Incidence Angle: {corr_angle}")

    # Correlation with Target Class
    corr_target = np.corrcoef(errors, oof_targets)[0, 1]
    print(f"Correlation between Error and Target Class: {corr_target}")

    # 6. Submission Generation
    threshold = 0.16676861786296204

    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        test_preds_folds = []

        # Ensemble Inference
        for i, (model, stats) in enumerate(fold_models):
            print(f"Inference with Fold {i+1} model...")
            # Create test loader with the specific scaler stats used during training
            test_loader = make_test_dataloader(test_data, scaler_stats=stats)

            # Predict
            preds = predict(model, test_loader, device)
            test_preds_folds.append(preds)

        # Average predictions across all folds
        avg_preds = np.mean(test_preds_folds, axis=0).flatten()

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": test_data["ids"], "is_iceberg": avg_preds})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
