import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import random

# Import from provided library
from library.config import Config
from library.model_architecture import DualPathWideBodyNet
from library.training_engine import train_fold
from library.data_processing import load_data, IcebergDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    # Override Config for fast baseline execution
    Config.EPOCHS = 25
    Config.setup()

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Training Loop (5 Folds)
    print("Starting 5-Fold Cross-Validation Training...")
    for fold_idx in range(Config.NUM_FOLDS):
        train_fold(fold_idx)

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")

    # Load pre-processed data
    data_dict = load_data(load_cached_data=True)

    # Cite debug_lesson_7: Prevent Data Leakage via Out-Of-Fold Evaluation
    # Use the explicit hold-out validation set
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    inc_val = data_dict["inc_val"]
    min_vals = data_dict["min_vals"]
    max_vals = data_dict["max_vals"]

    # Create Validation Dataset and Loader
    val_ds = IcebergDataset(X_val, inc_val, y_val, min_vals, max_vals, transform=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Ensemble Inference on Validation Set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []

    # Load all 5 models
    for fold_idx in range(Config.NUM_FOLDS):
        model = DualPathWideBodyNet()
        model.to(device)
        checkpoint_path = Config.get_model_path(fold_idx)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        models.append(model)

    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, inc_angles, labels in val_loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)

            # Ensemble prediction
            batch_preds = []
            for model in models:
                logits = model(images, inc_angles)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average predictions across models
            avg_preds = np.mean(batch_preds, axis=0)
            val_preds.extend(avg_preds)
            val_targets.extend(labels.numpy())

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()

    # Calculate Metric
    final_metric = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis:")
    # Calculate error magnitude
    errors = np.abs(val_preds - val_targets)

    # Extract features for correlation
    # X_val is (N, 75, 75, 3)
    # Band 1 is channel 0, Band 2 is channel 1
    b1_mean = np.mean(X_val[..., 0], axis=(1, 2))
    b1_std = np.std(X_val[..., 0], axis=(1, 2))
    b2_mean = np.mean(X_val[..., 1], axis=(1, 2))
    b2_std = np.std(X_val[..., 1], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": inc_val,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("error"))

    # 4. Submission
    THRESHOLD = 0.15744295919935183

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        X_test = data_dict["X_test"]
        inc_test = data_dict["inc_test"]
        ids_test = data_dict["ids_test"]

        test_ds = IcebergDataset(
            X_test,
            inc_test,
            labels=None,
            min_vals=min_vals,
            max_vals=max_vals,
            transform=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_preds = []

        with torch.no_grad():
            for images, inc_angles in test_loader:
                images = images.to(device)
                inc_angles = inc_angles.to(device)

                batch_preds = []
                for model in models:
                    logits = model(images, inc_angles)
                    probs = torch.sigmoid(logits)
                    batch_preds.append(probs.cpu().numpy())

                avg_preds = np.mean(batch_preds, axis=0)
                test_preds.extend(avg_preds)

        test_preds = np.array(test_preds).flatten()

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds})

        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
