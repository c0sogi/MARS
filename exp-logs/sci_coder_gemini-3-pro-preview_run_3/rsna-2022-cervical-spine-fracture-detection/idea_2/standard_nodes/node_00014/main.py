import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from library.config import Config, seed_everything
from library.train import train_model
from library.data import get_dataloaders, get_test_dataloader
from library.model import ResNetMIL
from library.utils import competition_log_loss


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Train Model
    # train_model handles the training loop, validation monitoring, and saving the best checkpoint.
    print("\n=== Starting Training ===")
    best_model_path = train_model(epochs=Config.EPOCHS)
    print(f"Best model saved at: {best_model_path}")

    # 3. Validation & Failure Analysis
    print("\n=== Validation & Failure Analysis ===")

    # Load the best model
    model = ResNetMIL(
        backbone_name=Config.BACKBONE, pretrained=False, num_classes=Config.N_CLASSES
    )
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Re-create validation loader to access data and metadata
    _, val_loader = get_dataloaders(
        Config.TRAIN_METADATA,
        Config.VAL_METADATA,
        Config.TRAIN_IMAGES_DIR,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []

    # Inference on Validation Set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate Metric
    metric = competition_log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis: Correlation between Error and Input Feature (Num Slices)
    # We hypothesize that scan depth (number of slices) might affect model performance.

    # Calculate weighted log loss per sample
    epsilon = 1e-15
    y_pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)
    weights = np.array([1 / 7] * 7 + [1.0])  # Weights for C1-C7 and patient_overall

    # Loss per element: -[y log p + (1-y) log (1-p)]
    loss_matrix = -(
        all_targets * np.log(y_pred_clipped)
        + (1 - all_targets) * np.log(1 - y_pred_clipped)
    )

    # Weighted loss per patient (scalar error magnitude)
    weighted_loss_matrix = loss_matrix * weights
    patient_errors = np.mean(weighted_loss_matrix, axis=1)

    # Extract Num Slices for validation samples
    val_df = val_loader.dataset.df
    slice_counts = []

    for _, row in val_df.iterrows():
        # Construct full path to image directory
        # image_path in metadata is relative (e.g., "train_images/UID")
        full_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        try:
            # Count files in the directory
            count = len(glob.glob(os.path.join(full_path, "*")))
        except Exception:
            count = 0
        slice_counts.append(count)

    # Calculate Pearson Correlation
    if len(slice_counts) > 1:
        correlation = np.corrcoef(patient_errors, slice_counts)[0, 1]
        print(
            f"Correlation between Error Magnitude and Number of Slices: {correlation}"
        )
    else:
        print("Not enough samples for correlation analysis.")

    # 4. Submission Generation
    threshold = 0.1307335607

    if metric < threshold:
        print("\n=== Generating Submission ===")

        # Load Test Data
        test_loader, test_df = get_test_dataloader(
            Config.TEST_METADATA,
            Config.TEST_IMAGES_DIR,
            Config.BATCH_SIZE,
            Config.NUM_WORKERS,
        )

        test_preds = []

        # Inference on Test Set
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.sigmoid(logits)
                test_preds.append(probs.cpu().numpy())

        if len(test_preds) > 0:
            test_preds = np.concatenate(test_preds, axis=0)
        else:
            # Fallback for empty test set (unlikely)
            test_preds = np.zeros((len(test_df), 8))

        # Format Submission
        # We need to expand the predictions (one row per study) into the submission format (8 rows per study)
        # Columns correspond to: C1, C2, C3, C4, C5, C6, C7, patient_overall
        target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        submission_rows = []

        for idx, row in test_df.iterrows():
            study_uid = row["StudyInstanceUID"]
            probs = test_preds[idx]

            for col_idx, col_name in enumerate(target_cols):
                row_id = f"{study_uid}_{col_name}"
                probability = probs[col_idx]
                submission_rows.append([row_id, probability])

        submission_df = pd.DataFrame(submission_rows, columns=["row_id", "fractured"])
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {metric} is not lower than threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
