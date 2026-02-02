import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, setup_logger, get_weighted_log_loss
from library.data import create_dataloaders
from library.model import Calibrated25DModel
from library.engine import fit, predict_and_submit


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = setup_logger()
    device = torch.device(Config.DEVICE)

    # --- OPTIMIZATION FOR TIME LIMIT ---
    # The test set is large (1817 studies). Processing 96 slices/study takes too long.
    # We reduce sequence length and epochs to ensure completion within 56 mins.
    Config.EPOCHS = 3
    Config.SEQ_LEN = 48  # Reduced from 96 to speed up I/O and inference

    logger.info(
        f"Configuration: Epochs={Config.EPOCHS}, Seq_Len={Config.SEQ_LEN}, Device={device}"
    )

    # 2. Data Loading
    logger.info("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = Calibrated25DModel()
    model = model.to(device)

    # 4. Training
    logger.info("Starting training loop...")
    best_loss = fit(model, train_loader, val_loader, device)

    # 5. Validation Reporting
    print(f"Final Validation Metric: {best_loss}")

    # 6. Failure Analysis
    logger.info("Performing failure analysis...")
    model.eval()

    # We need to compute loss per sample to correlate with features
    val_losses = []
    val_slice_counts = []
    val_fracture_counts = []

    # Access the dataset's path map to get slice counts
    val_dataset = val_loader.dataset

    # Iterate through validation set (non-shuffled)
    # Note: We must iterate the loader to get predictions, but we need to match
    # them to metadata. val_loader is shuffle=False, so order matches val_df.

    batch_start_idx = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)

            # Calculate weighted loss per sample
            # Weights shape: (8,)
            weights = Config.LOSS_WEIGHTS.to(device)

            # BCE per element: (B, 8)
            bce_none = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction="none"
            )

            # Weighted BCE: (B, 8)
            weighted_bce = bce_none * weights

            # Mean over classes: (B,) - this represents the sample's contribution to the metric
            sample_losses = weighted_bce.mean(dim=1).cpu().numpy()
            val_losses.extend(sample_losses)

            # Get metadata for this batch
            batch_size = images.size(0)
            for i in range(batch_size):
                global_idx = batch_start_idx + i
                if global_idx < len(val_df):
                    uid = val_df.iloc[global_idx]["StudyInstanceUID"]

                    # Get slice count
                    if uid in val_dataset.paths_map:
                        val_slice_counts.append(len(val_dataset.paths_map[uid]))
                    else:
                        val_slice_counts.append(0)

                    # Get fracture count (sum of C1-C7 columns in df)
                    # Assuming columns C1..C7 exist
                    f_count = val_df.iloc[global_idx][
                        ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                    ].sum()
                    val_fracture_counts.append(f_count)

            batch_start_idx += batch_size

    # Correlations
    if len(val_losses) == len(val_slice_counts):
        corr_slices, _ = pearsonr(val_losses, val_slice_counts)
        corr_fractures, _ = pearsonr(val_losses, val_fracture_counts)

        print(f"Correlation (Error vs Slice Count): {corr_slices:.4f}")
        print(f"Correlation (Error vs Fracture Count): {corr_fractures:.4f}")
    else:
        logger.warning("Mismatch in failure analysis lengths. Skipping correlations.")

    # 7. Submission
    # Threshold defined in task
    SUBMISSION_THRESHOLD = 0.1241588886

    if best_loss < SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation metric {best_loss} < {SUBMISSION_THRESHOLD}. Generating submission..."
        )

        # Ensure model is in best state
        if os.path.exists(Config.MODEL_SAVE_PATH):
            model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=device)
            )

        predict_and_submit(model, test_loader, test_df, device)
    else:
        logger.info(
            f"Validation metric {best_loss} did not meet threshold {SUBMISSION_THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
