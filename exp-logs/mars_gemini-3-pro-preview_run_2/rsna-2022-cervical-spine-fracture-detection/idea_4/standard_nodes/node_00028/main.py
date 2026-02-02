import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config, seed_everything
from library.dataset import CervicalSpineDataset
from library.model import AnatomicallyAwareModel
from library.train import run_training
from library.utils import get_logger


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(Config.SEED)
    logger = get_logger("Runfile")

    # Override Config for optimized execution
    Config.NUM_WORKERS = 12  # Utilize all available vCPUs
    Config.EPOCHS = 5  # Sufficient for the small dataset (161 samples) provided

    # Ensure submission directory exists
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    logger.info("Configuration configured. Starting pipeline...")

    # --- 2. Training ---
    # Execute the training loop provided in the library
    # This saves the best model to Config.CHECKPOINT_PATH
    run_training()

    # --- 3. Validation Inference ---
    logger.info("Starting Validation Inference...")
    device = torch.device(Config.DEVICE)

    # Load the best model
    model = AnatomicallyAwareModel().to(device)
    if not os.path.exists(Config.CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found at {Config.CHECKPOINT_PATH}")
        sys.exit(1)

    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Prepare Validation Loader
    val_dataset = CervicalSpineDataset(Config.VAL_METADATA_PATH, phase="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_targets = []

    # Inference Loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # --- 4. Metric Calculation ---
    # Metric: Weighted Multi-Label Logarithmic Loss
    # Weights: 1/7 for C1-C7, 1.0 for patient_overall
    # Config.TARGET_COLS order: ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    weights = np.array([1.0 / 7.0] * 7 + [1.0])

    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    preds_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # Calculate weighted log loss element-wise
    # L_ij = -w_j * [y_ij * log(p_ij) + (1 - y_ij) * log(1 - p_ij)]
    loss_matrix = -weights * (
        all_targets * np.log(preds_clipped)
        + (1 - all_targets) * np.log(1 - preds_clipped)
    )

    # Average across all rows (Total Loss / Total Elements)
    # Note: The task says "loss is averaged across all rows". Each patient contributes 8 rows.
    final_metric = np.sum(loss_matrix) / (loss_matrix.shape[0] * loss_matrix.shape[1])

    print(f"Final Validation Metric: {final_metric}")

    # --- 5. Failure Analysis ---
    logger.info("Performing Failure Analysis...")

    # Calculate mean loss per patient to correlate with metadata
    patient_losses = np.mean(loss_matrix, axis=1)

    # Load validation metadata
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Ensure alignment (DataLoader with shuffle=False preserves order)
    if len(val_meta) == len(patient_losses):
        val_meta["error"] = patient_losses
        val_meta["fracture_count"] = val_meta[
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        ].sum(axis=1)

        # Calculate correlations
        corr_overall = val_meta["error"].corr(val_meta["patient_overall"])
        corr_count = val_meta["error"].corr(val_meta["fracture_count"])

        print(f"Correlation (Error vs Patient Overall): {corr_overall}")
        print(f"Correlation (Error vs Fracture Count): {corr_count}")
    else:
        logger.warning(
            "Mismatch between validation predictions and metadata length. Skipping detailed correlation analysis."
        )

    # --- 6. Submission ---
    THRESHOLD = 0.38122559812935913

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        # Prepare Test Loader
        test_dataset = CervicalSpineDataset(Config.TEST_METADATA_PATH, phase="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        submission_rows = []

        with torch.no_grad():
            for images, study_uids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                # Format predictions
                for i, uid in enumerate(study_uids):
                    p = probs[i]  # Shape (8,)
                    # Target Columns: C1..C7, patient_overall
                    # Row IDs: [UID]_C1, [UID]_C2, ..., [UID]_patient_overall

                    # C1-C7
                    for j in range(7):
                        submission_rows.append(
                            {"row_id": f"{uid}_C{j+1}", "fractured": p[j]}
                        )

                    # Patient Overall
                    submission_rows.append(
                        {"row_id": f"{uid}_patient_overall", "fractured": p[7]}
                    )

        # Create DataFrame and Save
        sub_df = pd.DataFrame(submission_rows)
        sub_df.to_csv(SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {SUBMISSION_FILE} with {len(sub_df)} rows.")

    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
