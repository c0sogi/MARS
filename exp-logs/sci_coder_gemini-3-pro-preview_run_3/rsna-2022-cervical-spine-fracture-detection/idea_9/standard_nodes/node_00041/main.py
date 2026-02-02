import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.trainer import run_training
from library.inference import predict_test_set
from library.dataset import CervicalSpineDataset
from library.model import CervicalSpineMIL


def main():
    # 1. Setup
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)

    # 2. Training
    # We use a small number of epochs (5) for the baseline to ensure it runs quickly (< 2 hours).
    # The dataset is small (161 samples), so this will be very fast.
    print("=== Starting Training Phase ===")
    run_training(num_epochs=5, batch_size=Config.BATCH_SIZE, load_cached_data=True)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation Phase ===")

    # Load Model
    model = CervicalSpineMIL(pretrained=False)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model file not found.")
        return

    model.to(device)
    model.eval()

    # Load Validation Data
    val_dataset = CervicalSpineDataset(
        Config.VAL_METADATA_PATH, phase="val", load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    all_preds = []
    all_targets = []
    study_ids = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            # Targets
            labels_vert = batch["labels"]["vertebrae"]  # (B, 7)
            labels_patient = batch["labels"]["patient_overall"].unsqueeze(1)  # (B, 1)

            # Forward
            outputs = model(images)
            pred_vert = torch.sigmoid(outputs["vertebrae_logits"]).cpu()
            pred_patient = torch.sigmoid(outputs["patient_logit"]).cpu()

            # Combine C1-C7 and Patient into (B, 8)
            preds_batch = torch.cat([pred_vert, pred_patient], dim=1)
            targets_batch = torch.cat([labels_vert, labels_patient], dim=1)

            all_preds.append(preds_batch)
            all_targets.append(targets_batch)
            study_ids.extend(batch["study_id"])

    # Concatenate
    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    all_targets_np = torch.cat(all_targets, dim=0).numpy()

    # Compute Metric: Weighted Log Loss
    # Clip to avoid log(0)
    epsilon = 1e-15
    all_preds_clipped = np.clip(all_preds_np, epsilon, 1 - epsilon)

    # Weights: 1 for C1-C7, 7 for patient_overall
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 7])

    # Calculate element-wise Log Loss
    loss_matrix = -(
        all_targets_np * np.log(all_preds_clipped)
        + (1 - all_targets_np) * np.log(1 - all_preds_clipped)
    )

    # Apply weights and average across all rows
    val_metric = np.mean(loss_matrix * weights)

    print(f"Final Validation Metric: {val_metric}")

    # 4. Failure Analysis
    print("\n=== Starting Failure Analysis ===")

    # Calculate error per study (mean log loss over the 8 targets)
    study_errors = []
    for i in range(len(all_preds_np)):
        # Log loss for this single sample (scalar)
        loss = log_loss(all_targets_np[i], all_preds_clipped[i], labels=[0, 1])
        study_errors.append(loss)

    # Create a map for easy lookup
    error_map = dict(zip(study_ids, study_errors))

    # Get Input Feature: Volume Depth (Number of Slices)
    # We read the validation metadata to get paths
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    depths = []
    errors = []

    for idx, row in val_df.iterrows():
        sid = row["StudyInstanceUID"]
        if sid in error_map:
            # Get path
            rel_path = row["image_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Count slices
            try:
                if os.path.exists(full_path):
                    n_slices = len([name for name in os.listdir(full_path)])
                else:
                    n_slices = 0
            except Exception:
                n_slices = 0

            depths.append(n_slices)
            errors.append(error_map[sid])

    # Calculate Correlation
    if len(depths) > 1:
        # np.corrcoef returns a matrix [[1, r], [r, 1]]
        correlation = np.corrcoef(depths, errors)[0, 1]
        print(f"Correlation between Error and Volume Depth (Slices): {correlation}")
    else:
        print("Not enough data for correlation analysis.")

    # 5. Submission
    print("\n=== Submission Check ===")
    THRESHOLD = 0.12231192492082398

    if val_metric < THRESHOLD:
        print(f"Validation metric {val_metric} is better than threshold {THRESHOLD}.")
        print("Generating submission...")
        predict_test_set(load_cached_data=True)
    else:
        print(f"Validation metric {val_metric} is worse than threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
