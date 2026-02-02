import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from library.utils import set_seed, load_checkpoint
from library.dataset import get_data, IcebergDataset
from library.model import MPDPCNN, predict
from library.trainer import run_fold, generate_submission


def main():
    # Configuration
    SEED = 42
    EPOCHS = 50  # Fast baseline budget
    BATCH_SIZE = 32
    FOLDS = 5
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_50"
    THRESHOLD = 0.17174082291273365

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running 5-Fold MPDP-CNN Baseline on {device}...")

    # 1. Training Loop (5 Folds)
    # We use the provided run_fold which handles StratifiedKFold on Train+Val
    for fold in range(FOLDS):
        print(f"\n--- Training Fold {fold} ---")
        run_fold(
            fold_idx=fold,
            total_folds=FOLDS,
            epochs=EPOCHS,
            patience=10,  # Slightly reduced patience for speed
            batch_size=BATCH_SIZE,
            lr=1e-3,
            seed=SEED,
            input_dir=INPUT_DIR,
            metadata_dir=METADATA_DIR,
            working_dir=WORKING_DIR,
            load_cached_data=True,
        )

    # 2. Validation on Hold-Out Set
    print("\n--- Performing Validation on Hold-Out Set ---")

    # Calculate angle imputation value from train set (consistency)
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    angle_impute_val = df_train_meta["inc_angle"].median()

    # Load Validation Data
    X_val, ang_val, y_val, ids_val = get_data(
        "val",
        os.path.join(METADATA_DIR, "val.csv"),
        os.path.join(INPUT_DIR, "train.json"),
        WORKING_DIR,
        load_cached_data=True,
        angle_impute_val=angle_impute_val,
    )

    # Create DataLoader
    val_dataset = IcebergDataset(X_val, ang_val, ids=ids_val, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Ensemble Prediction
    ensemble_probs = np.zeros(len(y_val))
    valid_models = 0

    for fold in range(FOLDS):
        checkpoint_path = os.path.join(
            WORKING_DIR, "checkpoints", f"model_best_fold_{fold}.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found.")
            continue

        model = MPDPCNN().to(device)
        load_checkpoint(checkpoint_path, model)

        # Predict returns ids and probs
        # Since loader is shuffle=False, order is preserved, but we double check logic if needed.
        # predict() implementation extends lists.
        _, probs = predict(model, val_loader, device)
        ensemble_probs += np.array(probs)
        valid_models += 1

    if valid_models == 0:
        raise RuntimeError("No trained models found for validation.")

    ensemble_probs /= valid_models

    # 3. Metric Calculation
    final_metric = log_loss(y_val, ensemble_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    errors = np.abs(y_val - ensemble_probs)

    # Calculate Input Statistics
    # X_val shape: (N, 3, 75, 75). Channel 0 = Band 1, Channel 1 = Band 2
    b1_mean = np.mean(X_val[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X_val[:, 0, :, :], axis=(1, 2))
    b1_max = np.max(X_val[:, 0, :, :], axis=(1, 2))

    b2_mean = np.mean(X_val[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X_val[:, 1, :, :], axis=(1, 2))
    b2_max = np.max(X_val[:, 1, :, :], axis=(1, 2))

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": ang_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b1_max": b1_max,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
            "b2_max": b2_max,
        }
    )

    # Compute Correlations
    correlations = (
        analysis_df.corr()["error"].drop("error").sort_values(ascending=False)
    )
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission
    if final_metric < THRESHOLD:
        print("\nMetric condition met. Generating submission...")
        generate_submission(
            fold_indices=list(range(FOLDS)),
            batch_size=BATCH_SIZE,
            input_dir=INPUT_DIR,
            metadata_dir=METADATA_DIR,
            working_dir=WORKING_DIR,
            output_dir="./submission",
            load_cached_data=True,
        )
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
