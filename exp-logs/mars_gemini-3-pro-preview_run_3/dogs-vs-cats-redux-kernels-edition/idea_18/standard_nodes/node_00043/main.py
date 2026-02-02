import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import log_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library components
from library.config import (
    ENSEMBLE_CONFIGS,
    DEVICE,
    SEED,
    INPUT_DIR,
    TRAIN_META_PATH,
    VAL_META_PATH,
)
from library.utils import seed_everything, get_logger, get_checkpoint_path
from library.dataset import load_data_and_create_folds, get_dataloaders
from library.models import get_model
from library.engine import fit, validate_one_epoch
from library.inference import generate_ensemble_predictions


def main():
    # 1. Setup
    seed_everything(SEED)
    logger = get_logger("main")
    logger.info("Starting Fast Baseline Run...")

    # 2. Load Data & Folds
    # load_cached_data=True allows using pre-computed folds if available
    df_folds = load_data_and_create_folds(load_cached_data=True)

    # 3. Configure Fast Baseline
    # We modify the config objects in-place to limit runtime as per instructions.
    # Constraints: 1 Epoch, 1 Fold (Fold 0), Subsampled Training Data.
    TRAIN_SUBSET_RATIO = 0.1  # Use 10% of training data for speed

    for config in ENSEMBLE_CONFIGS:
        config.epochs = 1
        config.num_folds = 1  # Only run Fold 0

    # Store validation predictions for ensemble averaging
    # Key: index in df_folds, Value: list of probabilities
    val_preds_accumulator = {}
    val_indices_all = []
    val_labels_all = []

    # 4. Training Loop
    for config in ENSEMBLE_CONFIGS:
        logger.info(f"Processing Architecture: {config.model_name}")

        for fold in range(config.num_folds):
            logger.info(f"Training Fold {fold}...")

            # --- Data Subsampling for Fast Training ---
            # We want to limit training samples but keep validation full for valid metrics.
            train_mask = df_folds["fold"] != fold
            val_mask = df_folds["fold"] == fold

            # Sample training data
            train_indices = (
                df_folds[train_mask]
                .sample(frac=TRAIN_SUBSET_RATIO, random_state=SEED)
                .index
            )
            val_indices = df_folds[val_mask].index

            # Create a subset dataframe containing only the selected rows
            subset_indices = np.concatenate([train_indices, val_indices])
            df_subset = df_folds.loc[subset_indices].copy()

            # Get DataLoaders using the subset
            train_loader, val_loader = get_dataloaders(
                df_subset, fold, config.img_size, config.batch_size
            )

            # --- Model Setup ---
            model = get_model(config.model_name, pretrained=True, num_classes=1)
            model.to(DEVICE)

            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

            # --- Train ---
            model = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=DEVICE,
                epochs=config.epochs,
                model_name=config.name,
                fold=fold,
            )

            # --- Collect Validation Predictions for Ensemble ---
            # We run a validation pass to get probabilities for the ensemble metric
            # Note: fit() returns the model with best weights loaded
            model.eval()
            fold_preds = []
            fold_labels = []
            fold_indices = []  # We need to track indices to align with metadata

            # We iterate val_loader again.
            # Note: val_loader from get_dataloaders is sequential (shuffle=False).
            # The indices in df_subset[val_mask] correspond to the order in val_loader.
            current_val_indices = df_subset[val_mask].index.values
            idx_pointer = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

                    fold_preds.extend(probs)
                    fold_labels.extend(labels.numpy().flatten())

                    batch_size = images.size(0)
                    fold_indices.extend(
                        current_val_indices[idx_pointer : idx_pointer + batch_size]
                    )
                    idx_pointer += batch_size

            # Accumulate
            for i, idx in enumerate(fold_indices):
                if idx not in val_preds_accumulator:
                    val_preds_accumulator[idx] = []
                val_preds_accumulator[idx].append(fold_preds[i])

            # Store labels once (they are constant)
            if not val_labels_all:
                val_labels_all = fold_labels
                val_indices_all = fold_indices

            # Cleanup
            del model, optimizer, scheduler, train_loader, val_loader
            torch.cuda.empty_cache()

    # 5. Evaluation (Ensemble)
    logger.info("Computing Ensemble Metrics...")

    # Average predictions for each validation sample
    final_preds = []
    final_labels = []  # Re-align based on indices to be safe

    # We iterate over val_indices_all to ensure order
    for idx in val_indices_all:
        preds = val_preds_accumulator[idx]
        avg_prob = sum(preds) / len(preds)
        final_preds.append(avg_prob)

        # Get label from dataframe
        final_labels.append(df_folds.loc[idx, "label"])

    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)

    # Compute Metric
    val_log_loss = log_loss(final_labels, final_preds, labels=[0, 1])

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_log_loss:.16f}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate Error
    errors = np.abs(final_preds - final_labels)

    # Extract Metadata Features for the validation set
    # We need to read file stats.
    widths = []
    heights = []
    file_sizes = []
    aspect_ratios = []

    # We iterate indices to get filepaths
    for idx in val_indices_all:
        rel_path = df_folds.loc[idx, "filepath"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # File Size
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))

            # Image Dims (Read header only if possible, but cv2 reads all)
            # For failure analysis on ~2000 images, reading is acceptable.
            img = cv2.imread(full_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Compute Correlations
    features = {
        "width": widths,
        "height": heights,
        "aspect_ratio": aspect_ratios,
        "file_size": file_sizes,
    }

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    for name, values in features.items():
        if len(set(values)) > 1:  # Avoid constant input
            corr, p_val = pearsonr(errors, values)
            print(f"  {name}: Correlation = {corr:.4f} (p={p_val:.4f})")
        else:
            print(f"  {name}: Correlation = N/A (Constant values)")

    # 7. Submission
    # Threshold from instructions
    THRESHOLD = 0.009074434935821756

    if val_log_loss < THRESHOLD:
        logger.info(
            f"Validation metric {val_log_loss} < {THRESHOLD}. Generating submission..."
        )
        # generate_ensemble_predictions uses the global ENSEMBLE_CONFIGS which we modified in-place.
        # It will run inference for the models we trained (Fold 0, Epoch 1).
        generate_ensemble_predictions(use_tta=True)
    else:
        logger.info(
            f"Validation metric {val_log_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
