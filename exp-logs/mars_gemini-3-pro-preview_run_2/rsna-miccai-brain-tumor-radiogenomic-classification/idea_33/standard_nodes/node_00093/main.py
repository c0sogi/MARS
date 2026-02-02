import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import create_dataloaders
from library.model import AsymmetricGroupedEfficientNet
from library.engine import run_training, validate, generate_submission


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("runfile")
    logger.info("Initializing pipeline...")

    # 2. Data Loading
    # load_cached_data=True ensures we use the cache if available, or generate it if not.
    # The cache generation uses the Logical-Consensus strategy defined in roi_selection.py
    dataloaders = create_dataloaders(load_cached_data=True)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders.get("test")

    # 3. Model Initialization
    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    model = AsymmetricGroupedEfficientNet().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Training
    # run_training handles the loop and saves the best model to Config.MODEL_SAVE_PATH
    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=Config.EPOCHS,
        device=device,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 6. Final Validation
    logger.info("Loading best model for validation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    criterion = torch.nn.BCEWithLogitsLoss()
    val_metrics = validate(model, val_loader, criterion, device)
    final_auc = val_metrics["auc"]

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc:.16f}")

    # 7. Failure Analysis
    logger.info("Performing failure analysis...")

    # We need to align predictions with metadata to analyze errors.
    # The val_loader is sequential, and val_loader.dataset.metadata contains the exact rows used.
    val_df = val_loader.dataset.metadata.copy()

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    val_df["prob"] = all_probs
    val_df["target"] = all_targets
    val_df["error"] = (val_df["target"] - val_df["prob"]).abs()

    # Feature extraction for correlation: Slice Count (Proxy for scan quality/volume)
    # We extract this from the file system based on the paths in metadata
    slice_counts = []
    for _, row in val_df.iterrows():
        # Construct full path to FLAIR directory
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        try:
            # Count files in directory
            if os.path.exists(flair_path):
                count = len(
                    [
                        name
                        for name in os.listdir(flair_path)
                        if os.path.isfile(os.path.join(flair_path, name))
                    ]
                )
            else:
                count = 0
        except Exception:
            count = 0
        slice_counts.append(count)

    val_df["slice_count"] = slice_counts

    # Calculate correlations
    # Handle cases where std dev is 0 (constant input) to avoid warnings
    if val_df["slice_count"].std() > 0:
        corr_slices, _ = pearsonr(val_df["error"], val_df["slice_count"])
    else:
        corr_slices = 0.0

    if val_df["target"].std() > 0:
        corr_target, _ = pearsonr(val_df["error"], val_df["target"])
    else:
        corr_target = 0.0

    print(f"Correlation between Error and Slice Count: {corr_slices:.4f}")
    print(f"Correlation between Error and Target Class: {corr_target:.4f}")

    # 8. Submission
    threshold = 0.6321818181818182
    if final_auc > threshold:
        if test_loader is not None:
            logger.info(
                f"Validation metric ({final_auc:.4f}) > threshold ({threshold:.4f}). Generating submission..."
            )
            generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
        else:
            logger.warning("Test loader is None. Cannot generate submission.")
    else:
        logger.warning(
            f"Validation metric ({final_auc:.4f}) <= threshold ({threshold:.4f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
