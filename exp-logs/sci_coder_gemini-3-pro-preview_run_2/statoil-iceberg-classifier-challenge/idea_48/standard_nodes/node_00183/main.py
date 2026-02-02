import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library components
from library.config import (
    SEED,
    DEVICE,
    NUM_FOLDS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
)
from library.utils import seed_everything
from library.data_loader import load_and_process_data, get_data_loaders, get_test_loader
from library.train import run_fold, validate
from library.model import RDPWBN


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Load Data
    # load_and_process_data returns:
    # X_train, y_train, inc_train, train_ids, X_test, inc_test, test_ids
    print("Loading and processing data...")
    data = load_and_process_data(load_cached_data=True)
    X, y, inc, train_ids, X_test, inc_test, test_ids = data

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(y))
    model_paths = []

    # Iterate through folds
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold_idx + 1}/{NUM_FOLDS} {'='*20}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        inc_train_fold, inc_val_fold = inc[train_idx], inc[val_idx]

        # Create Loaders
        train_loader, val_loader = get_data_loaders(
            X_train_fold,
            y_train_fold,
            inc_train_fold,
            X_val_fold,
            y_val_fold,
            inc_val_fold,
            BATCH_SIZE,
            num_workers=2,
        )

        # Train Model
        # run_fold handles model instantiation, training, early stopping, and returns best state
        best_state = run_fold(
            fold_idx,
            train_loader,
            val_loader,
            DEVICE,
            LEARNING_RATE,
            NUM_EPOCHS,
            PATIENCE,
        )

        # Save Best Model
        save_path = os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")
        torch.save(best_state, save_path)
        model_paths.append(save_path)

        # Generate OOF Predictions for this fold
        model = RDPWBN().to(DEVICE)
        model.load_state_dict(best_state)

        # validate returns: val_loss, val_acc, preds, targets
        # preds are probabilities for class 1
        _, _, preds, _ = validate(
            model, val_loader, torch.nn.CrossEntropyLoss(), DEVICE
        )

        oof_preds[val_idx] = preds

    # 4. Global Validation Metric
    final_log_loss = log_loss(y, oof_preds)
    print(f"Final Validation Metric: {final_log_loss}")

    # 5. Failure Analysis
    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS")
    print("=" * 30)

    # Calculate absolute error
    errors = np.abs(oof_preds - y)

    # Extract features for correlation analysis
    # X is (N, 3, 75, 75). Channel 0: Band 1, Channel 1: Band 2
    # We compute mean and std for each image
    b1_flat = X[:, 0, :, :].reshape(len(X), -1)
    b2_flat = X[:, 1, :, :].reshape(len(X), -1)

    b1_mean = b1_flat.mean(axis=1)
    b1_std = b1_flat.std(axis=1)
    b2_mean = b2_flat.mean(axis=1)
    b2_std = b2_flat.std(axis=1)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute correlation
    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 6. Submission Logic
    THRESHOLD = 0.14772333549413377

    if final_log_loss < THRESHOLD:
        print(
            f"\nValidation Metric ({final_log_loss}) is better than threshold ({THRESHOLD})."
        )
        print("Generating submission...")

        # Create Test Loader
        test_loader = get_test_loader(X_test, inc_test, BATCH_SIZE, num_workers=2)

        # Ensemble Inference
        test_preds_accum = np.zeros(len(X_test))

        for i, path in enumerate(model_paths):
            print(f"Inference with model fold {i+1}...")
            model = RDPWBN().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, inc_angles in test_loader:
                    images = images.to(DEVICE)
                    inc_angles = inc_angles.to(DEVICE)

                    outputs = model(images, inc_angles)
                    # Get probability of class 1
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    fold_preds.extend(probs.cpu().numpy())

            test_preds_accum += np.array(fold_preds)

        # Average predictions
        avg_preds = test_preds_accum / NUM_FOLDS

        # Create Submission DataFrame
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Metric ({final_log_loss}) is NOT better than threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
