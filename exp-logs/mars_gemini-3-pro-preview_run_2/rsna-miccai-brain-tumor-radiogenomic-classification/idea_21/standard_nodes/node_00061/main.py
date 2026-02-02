import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.engine import run_training, generate_submission
from library.dataset import BraTSDataset
from library.model import AsymmetricEfficientNet


def perform_failure_analysis(model, device, logger):
    """
    Evaluates the model on the validation set, calculates the final metric,
    and performs failure analysis (correlation of error with slice count).
    """
    # Load validation metadata
    df_val = pd.read_csv(Config.VAL_CSV)

    # Create dataset and loader
    val_dataset = BraTSDataset(df_val, phase="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    all_targets = []
    all_preds = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            # Forward pass
            outputs = model(images)
            # Sigmoid for probability
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()

            all_targets.extend(targets.numpy().flatten())
            all_preds.extend(preds)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 1. Calculate and Print Final Metric
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Fallback for edge cases with single class in batch/set
        auc = 0.5

    # STRICT FORMAT REQUIREMENT
    print(f"Final Validation Metric: {auc}")

    # 2. Failure Analysis
    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Extract feature: FLAIR Slice Count
    # We iterate through the dataframe to count files in the FLAIR directory
    # This serves as a proxy for "Scan Volume/Depth"
    slice_counts = []
    for _, row in df_val.iterrows():
        flair_path_rel = row["path_FLAIR"]
        flair_path_full = os.path.join(Config.INPUT_DIR, flair_path_rel)

        # Count files safely
        if os.path.exists(flair_path_full):
            # Simple count of files
            count = len([f for f in os.listdir(flair_path_full) if f.endswith(".dcm")])
        else:
            count = 0
        slice_counts.append(count)

    slice_counts = np.array(slice_counts)

    # Calculate correlation
    if len(errors) > 1 and np.std(slice_counts) > 0 and np.std(errors) > 0:
        correlation = np.corrcoef(slice_counts, errors)[0, 1]
    else:
        correlation = 0.0

    print(
        f"Correlation between Error Magnitude and FLAIR Slice Count: {correlation:.6f}"
    )

    return auc


def main():
    # Setup
    seed_everything(Config.SEED)
    logger = get_logger()

    # --------------------------------------------------------------------------
    # 1. Training
    # --------------------------------------------------------------------------
    # run_training executes the training loop and saves the best model to Config.BEST_MODEL_PATH
    logger.info("Starting training pipeline...")
    run_training(logger)

    # --------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    logger.info("Loading best model for analysis...")
    device = torch.device(Config.DEVICE)
    model = AsymmetricEfficientNet()

    if os.path.exists(Config.BEST_MODEL_PATH):
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.error("Best model not found. Training may have failed.")
        return

    model = model.to(device)

    # Perform analysis
    final_auc = perform_failure_analysis(model, device, logger)

    # --------------------------------------------------------------------------
    # 3. Submission
    # --------------------------------------------------------------------------
    # Threshold defined in task
    THRESHOLD = 0.6303636363636363

    if final_auc > THRESHOLD:
        logger.info(
            f"AUC {final_auc:.6f} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(logger)
    else:
        logger.info(
            f"AUC {final_auc:.6f} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
