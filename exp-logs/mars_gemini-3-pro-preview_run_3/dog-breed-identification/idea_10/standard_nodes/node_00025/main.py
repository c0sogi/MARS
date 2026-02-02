import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr
from PIL import Image
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint
from library.dataset import (
    get_dataloaders,
    get_class_mapping,
    get_transforms,
    DogDataset,
)
from library.engine import train_fold
from library.model import get_model
from library.inference import predict_with_tta, run_inference


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    logger = get_logger("runfile")

    logger.info("Starting Fast Baseline Run...")

    # Modify Config for speed constraints to ensure completion within 2 hours
    # 5 folds * 4 epochs = 20 epochs total. ~40 mins training time on A100.
    Config.FINE_TUNE_EPOCHS = 4
    Config.WARMUP_EPOCHS = 1

    # Ensure class mapping is computed on the full dataset first
    # This guarantees consistent indices across all folds
    get_class_mapping(load_cached_data=False)

    # 2. Prepare Data Splitting
    # We use the provided training metadata to create 5 stratified folds
    df_train_full = pd.read_csv(Config.TRAIN_CSV)

    # StratifiedKFold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Assign folds
    df_train_full["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(df_train_full, df_train_full["breed"])
    ):
        df_train_full.loc[val_idx, "fold"] = fold_idx

    # Store original paths to restore later
    original_train_csv = Config.TRAIN_CSV
    original_val_csv = Config.VAL_CSV

    # 3. Training Loop
    for fold in range(Config.N_FOLDS):
        logger.info(f"--- Processing Fold {fold}/{Config.N_FOLDS - 1} ---")

        # Prepare temporary CSVs for this fold
        train_df = df_train_full[df_train_full["fold"] != fold].reset_index(drop=True)
        val_df = df_train_full[df_train_full["fold"] == fold].reset_index(drop=True)

        temp_train_path = os.path.join(Config.WORKING_DIR, f"train_fold_{fold}.csv")
        temp_val_path = os.path.join(Config.WORKING_DIR, f"val_fold_{fold}.csv")

        train_df.to_csv(temp_train_path, index=False)
        val_df.to_csv(temp_val_path, index=False)

        # Monkey-patch Config to use these temporary files
        Config.TRAIN_CSV = temp_train_path
        Config.VAL_CSV = temp_val_path

        # Get DataLoaders (will use the patched CSVs)
        # Note: get_dataloaders calls get_class_mapping with load_cached_data=True,
        # so it uses the mapping we generated at step 1.
        dataloaders = get_dataloaders()

        # Initialize Model
        model = get_model(device=Config.DEVICE, pretrained=True)

        # Train
        train_fold(fold, model, dataloaders["train"], dataloaders["val"], Config.DEVICE)

        # Cleanup
        del model, dataloaders
        torch.cuda.empty_cache()

    # Restore Config
    Config.TRAIN_CSV = original_train_csv
    Config.VAL_CSV = original_val_csv

    # 4. Final Validation on Hold-out Set
    logger.info("--- Performing Final Validation ---")

    # Load the official validation set (not the fold validation sets)
    # This corresponds to metadata/val.csv
    class_to_idx, classes = get_class_mapping(load_cached_data=True)
    val_transform = get_transforms("val")

    # We use the original VAL_CSV
    val_dataset = DogDataset(Config.VAL_CSV, class_to_idx, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Collect True Labels
    y_true = []
    for _, labels, _ in val_loader:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)

    # Ensemble Prediction
    accumulated_probs = None

    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.WORKING_DIR, f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            logger.warning(f"Model for fold {fold} missing. Skipping.")
            continue

        model = get_model(device=Config.DEVICE, pretrained=False)
        load_checkpoint(model_path, model, device=Config.DEVICE)

        probs, _ = predict_with_tta(model, val_loader, Config.DEVICE)

        if accumulated_probs is None:
            accumulated_probs = probs
        else:
            accumulated_probs += probs

        del model
        torch.cuda.empty_cache()

    if accumulated_probs is None:
        logger.error("No predictions generated.")
        return

    avg_probs = accumulated_probs / Config.N_FOLDS

    # Compute Metric
    # Ensure we account for all classes in log_loss
    metric = log_loss(y_true, avg_probs, labels=list(range(len(classes))))
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    logger.info("--- Performing Failure Analysis ---")

    # Calculate per-sample loss
    # Clip probabilities to avoid log(0)
    epsilon = 1e-15
    avg_probs_clipped = np.clip(avg_probs, epsilon, 1 - epsilon)

    # Select probability of the true class
    # advanced indexing: [row_indices, col_indices]
    true_class_probs = avg_probs_clipped[np.arange(len(y_true)), y_true]
    sample_losses = -np.log(true_class_probs)

    # Load metadata to correlate
    df_val = pd.read_csv(Config.VAL_CSV)

    # Extract image stats
    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []

    for idx, row in df_val.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # File Size
        try:
            f_size = os.path.getsize(full_path)
        except:
            f_size = 0
        file_sizes.append(f_size)

        # Dimensions
        try:
            with Image.open(full_path) as img:
                w, h = img.size
        except:
            w, h = 0, 0

        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h if h > 0 else 0)

    # Compute Correlations
    corr_size, _ = pearsonr(sample_losses, file_sizes)
    corr_width, _ = pearsonr(sample_losses, widths)
    corr_height, _ = pearsonr(sample_losses, heights)
    corr_ar, _ = pearsonr(sample_losses, aspect_ratios)

    print(f"Correlation (Loss vs File Size): {corr_size}")
    print(f"Correlation (Loss vs Width): {corr_width}")
    print(f"Correlation (Loss vs Height): {corr_height}")
    print(f"Correlation (Loss vs Aspect Ratio): {corr_ar}")

    # 6. Submission
    threshold = 0.14004325100369866
    if metric < threshold:
        logger.info(
            f"Metric {metric} meets threshold {threshold}. Generating submission..."
        )
        # run_inference uses Config.TEST_CSV and Config.SUBMISSION_PATH
        # It loads models from Config.WORKING_DIR
        run_inference()
    else:
        logger.info(
            f"Metric {metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
