import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import seed_everything
from library.train import run_training
from library.inference import generate_submission
from library.model import FractureModel
from library.dataset import FractureDataset


def weighted_log_loss_metric(y_true, y_pred):
    """
    Calculates the weighted multi-label logarithmic loss.
    Weights: 1 for C1-C7, 7 for patient_overall.
    """
    # Define weights
    # Columns are C1-C7 (indices 0-6) and patient_overall (index 7)
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 7])

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate binary log loss for each label
    # L_ij = -w_j * [y * log(p) + (1-y) * log(1-p)]
    loss_per_label = -weights * (
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    # Sum losses per row (exam)
    loss_per_row = np.sum(loss_per_label, axis=1)

    # Average across all rows
    final_metric = np.mean(loss_per_row)

    return final_metric, loss_per_row


def evaluate_validation(device):
    """
    Runs inference on validation set and computes the competition metric.
    """
    print("Loading validation data...")
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    val_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    val_dataset = FractureDataset(
        val_df, transforms=val_transforms, mode="val", load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = FractureModel(
        backbone_name=Config.BACKBONE, pretrained=False, num_classes=Config.NUM_CLASSES
    )
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_targets = []

    print("Running validation inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    # Concatenate
    # Shape: (N, 7) for C1-C7
    y_pred_c = np.concatenate(all_probs, axis=0)

    # Shape: (N, 8) for C1-C7 + Overall
    y_true_raw = np.concatenate(all_targets, axis=0)
    # Slice to get only C1-C7 for the metric calculation logic below
    y_true_c = y_true_raw[:, :7]

    # Derive patient_overall
    # Pred: max(C1...C7)
    y_pred_overall = np.max(y_pred_c, axis=1, keepdims=True)
    # True: patient_overall column from dataframe (or from the loaded labels)
    y_true_overall = val_df[Config.OVERALL_COL].values.reshape(-1, 1)

    # Combine into (N, 8) arrays
    y_pred_full = np.hstack([y_pred_c, y_pred_overall])
    y_true_full = np.hstack([y_true_c, y_true_overall])

    # Calculate Metric
    metric, loss_per_row = weighted_log_loss_metric(y_true_full, y_pred_full)

    return metric, loss_per_row, val_df


def test_submission_format():
    """
    Ensures submission format is valid.
    """
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df.shape}")
        print(df.head())


def analyze_failures(loss_per_row, val_df):
    """
    Analyzes correlation between error magnitude and metadata.
    """
    val_df = val_df.copy()
    val_df["error_magnitude"] = loss_per_row

    print("\n=== Failure Analysis ===")
    print("Correlation between Error Magnitude and Target Variables:")

    # Correlate with targets
    target_cols = Config.TARGET_COLS + [Config.OVERALL_COL]
    correlations = val_df[target_cols + ["error_magnitude"]].corr()["error_magnitude"]

    print(correlations.drop("error_magnitude").sort_values(ascending=False))


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Adjust Config for baseline speed
    # We use a small number of epochs to ensure completion within 2 hours
    # The dataset is small (~160 train samples), so 5-10 epochs is very fast.
    Config.EPOCHS = 10

    # 2. Training
    print("\n=== Starting Training ===")
    run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False)

    # 3. Validation & Metric
    print("\n=== Starting Validation ===")
    metric, loss_per_row, val_df = evaluate_validation(device)
    print(f"Final Validation Metric: {metric}")

    # 4. Failure Analysis
    analyze_failures(loss_per_row, val_df)

    # 5. Submission
    # Only generate submission if metric is reasonable (better than baseline ~9.7)
    if metric < 9.5:
        print("\n=== Generating Submission ===")
        generate_submission(
            batch_size=Config.BATCH_SIZE, device=Config.DEVICE, debug=False
        )
        test_submission_format()
        print("Done.")
    else:
        print(f"\nMetric {metric:.4f} is too high. Skipping submission generation.")


if __name__ == "__main__":
    main()
