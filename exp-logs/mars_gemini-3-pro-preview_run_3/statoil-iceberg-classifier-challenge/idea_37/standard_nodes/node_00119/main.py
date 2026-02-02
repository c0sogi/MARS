import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import DPDB_NBA_CNN
from library.train import run_kfold_training, generate_submission


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Starting training pipeline...")

    # 2. Train the ensemble (5 Folds)
    # We use the default epochs (75) as the dataset is small and training is fast.
    run_kfold_training(max_samples=None, num_epochs=Config.NUM_EPOCHS)

    print("\nTraining complete. Starting OOF inference for validation...")

    # 3. Generate Out-Of-Fold (OOF) Predictions
    # Load full training data to reconstruct splits
    X_all, y_all, angles_all, _, _, _ = process_and_cache_data(load_cached_data=True)

    # storage for OOF results
    # We will store results mapped by original index
    oof_preds = np.zeros(len(y_all))
    oof_targets = np.zeros(len(y_all))
    oof_features = {}  # Dict to store features by index

    # Reconstruct Stratified K-Fold splits
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate through folds to generate predictions on unseen data
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        # Load Model for this fold
        model_path = Config.get_checkpoint_path(fold_idx)
        model = DPDB_NBA_CNN().to(device)
        try:
            load_checkpoint(model_path, model, device=device)
        except FileNotFoundError:
            print(f"Error: Model for fold {fold_idx} not found.")
            continue

        model.eval()

        # Prepare validation data for this fold
        X_val = X_all[val_idx]
        y_val = y_all[val_idx]
        angles_val = angles_all[val_idx]

        # Create dataset/loader manually to ensure we track indices correctly
        # We don't use get_cv_loaders here because we need direct control over indices for feature extraction
        val_dataset = IcebergDataset(X_val, angles_val, y_val, transform=None)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        fold_preds = []

        with torch.no_grad():
            batch_start = 0
            for inputs, angles, labels in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)

                # Inference
                logits = model(inputs, angles)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_preds.extend(probs)

                # Feature Extraction for Failure Analysis
                # Move to CPU numpy
                inputs_np = inputs.cpu().numpy()  # (B, 3, 75, 75)
                angles_np = angles.cpu().numpy()

                batch_size = inputs.size(0)

                for i in range(batch_size):
                    global_idx = val_idx[batch_start + i]

                    # Extract image stats
                    b1 = inputs_np[i, 0, :, :]
                    b2 = inputs_np[i, 1, :, :]

                    feat = {
                        "inc_angle": angles_np[i].item(),
                        "b1_mean": np.mean(b1),
                        "b1_std": np.std(b1),
                        "b2_mean": np.mean(b2),
                        "b2_std": np.std(b2),
                    }
                    oof_features[global_idx] = feat

                batch_start += batch_size

        # Store OOF predictions
        oof_preds[val_idx] = np.array(fold_preds)
        oof_targets[val_idx] = y_val

    # 4. Filter by Metadata Hold-out Set
    print("Loading metadata to identify hold-out validation set...")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Metadata file not found: {val_meta_path}")

    df_meta_val = pd.read_csv(val_meta_path)
    holdout_indices = df_meta_val["original_index"].values

    # Select predictions corresponding to the hold-out set
    # Note: Since we used OOF, these predictions were made by models that did NOT see these samples in training.
    holdout_preds = oof_preds[holdout_indices]
    holdout_targets = oof_targets[holdout_indices]

    # 5. Compute Metric
    final_metric = log_loss(holdout_targets, holdout_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Construct DataFrame for analysis
    analysis_data = []
    for i, idx in enumerate(holdout_indices):
        row = oof_features.get(idx, {})
        row["target"] = holdout_targets[i]
        row["pred"] = holdout_preds[i]
        row["error"] = np.abs(row["target"] - row["pred"])
        analysis_data.append(row)

    df_analysis = pd.DataFrame(analysis_data)

    # Compute correlations with error
    if not df_analysis.empty:
        correlations = (
            df_analysis.corr()["error"]
            .drop(["error", "target", "pred"])
            .sort_values(ascending=False)
        )
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Analysis dataframe empty.")

    # 7. Submission
    THRESHOLD = 0.1806015565870406
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
