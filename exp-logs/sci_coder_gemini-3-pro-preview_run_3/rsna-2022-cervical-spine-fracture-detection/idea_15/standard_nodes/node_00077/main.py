import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device, get_logger
from library.preprocessor import Preprocessor
from library.trainer import Trainer
from library.inference import predict_test_set
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import ConvNeXtMIL


def main():
    # Initialize Logger and Seeds
    logger = get_logger()
    seed_everything(Config.SEED)
    logger.info("Starting End-to-End Pipeline (Fast Baseline)...")

    # --- 2. Preprocessing ---
    logger.info("=== Step 1: Preprocessing ===")
    preprocessor = Preprocessor()
    preprocessor.run()

    # --- 3. Training ---
    logger.info("=== Step 2: Training ===")
    trainer = Trainer()
    trainer.fit()

    # --- 4. Validation & Failure Analysis ---
    logger.info("=== Step 3: Validation & Failure Analysis ===")

    device = get_device()

    # Load the best model saved by Trainer
    model = ConvNeXtMIL(pretrained=False)
    weights_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        logger.warning("Best model not found. Using random weights for validation.")

    model.to(device)
    model.eval()

    # Setup Validation Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    if Config.DEBUG_DATA_SIZE:
        val_df = val_df.iloc[: Config.DEBUG_DATA_SIZE]

    val_dataset = CervicalSpineDataset(
        val_df,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms(split="val"),
        split="val",
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Metric Calculation Components
    # We use BCEWithLogitsLoss with reduction='none' to compute loss per sample manually
    bce_func = torch.nn.BCEWithLogitsLoss(reduction="none")

    all_sample_losses = []
    all_patient_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["targets"].to(device, dtype=torch.float32)

            # Forward Pass
            logits = model(images)

            # Calculate Loss Matrix (Batch, 8)
            loss_matrix = bce_func(logits, targets)

            # Separate Vertebrae (0-6) and Patient (7) losses
            c_losses = loss_matrix[:, :7]
            p_losses = loss_matrix[:, 7]

            # Compute Weighted Metric per sample: Mean(Vertebrae) + Patient
            # This aligns with "Patient label weighted more highly" (1 vs 1/7)
            # Cite solution_lesson_node_00043: Normalize by 2 to match weighted average definition.
            sample_losses = (c_losses.mean(dim=1) + p_losses) / 2.0

            all_sample_losses.extend(sample_losses.cpu().numpy())
            all_patient_targets.extend(targets[:, 7].cpu().numpy())

    # Final Metric: Average across all rows (samples)
    final_metric = np.mean(all_sample_losses)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Fracture Presence
    # (Input Feature: Patient Overall Label)
    if len(all_sample_losses) > 1:
        corr, _ = pearsonr(all_sample_losses, all_patient_targets)
        print(f"Failure Analysis - Correlation (Loss vs Fracture Presence): {corr:.4f}")
    else:
        print("Failure Analysis - Not enough samples for correlation.")

    # --- 5. Submission Logic ---
    logger.info("=== Step 4: Submission Check ===")
    threshold = 0.06429807151236185

    if final_metric < threshold:
        logger.info(f"Metric {final_metric} < {threshold}. Generating submission...")
        # Run inference on test set
        predict_test_set(debug_size=Config.DEBUG_DATA_SIZE)
    else:
        logger.info(
            f"Metric {final_metric} >= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
