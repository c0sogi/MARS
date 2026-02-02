import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import get_logger, calculate_roc_auc, seed_everything
from library.data import load_data_to_memory, get_fold_dataloaders
from library.training import Trainer
from library.inference import run_inference


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for Optimized Execution
    # Prioritize convergence on a single fold over short training on multiple folds.
    # Cite solution_lesson_node_00014
    Config.epochs = 15
    Config.n_folds = 5
    Config.load_in_memory = True
    Config.debug = False

    logger = get_logger("runfile")
    Config.setup()

    logger.info(f"Starting execution with {Config.epochs} epochs per fold.")

    # --- 2. Data Loading ---
    # Load all data into RAM (Cached)
    train_images, train_labels, test_images, test_ids = load_data_to_memory(
        load_cached_data=True
    )

    # --- 3. Training & Cross-Validation ---
    # We will collect Out-Of-Fold (OOF) predictions to evaluate the model globally.
    # Initialize array to store predictions for all training samples.
    oof_preds = np.full(len(train_labels), -1.0, dtype=np.float32)

    # Re-create the StratifiedKFold splitter to map fold indices back to global indices
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    splits = list(skf.split(train_images, train_labels))

    # Run only a single fold to ensure full convergence within time limits
    # Cite solution_lesson_node_00014
    for fold in range(1):
        logger.info(f"=== Running Fold {fold}/{Config.n_folds - 1} ===")

        # Get DataLoaders for this fold
        train_loader, val_loader = get_fold_dataloaders(
            fold, train_images, train_labels
        )

        # Train the model
        trainer = Trainer(fold, train_loader, val_loader)
        trainer.fit()

        # --- Generate OOF Predictions for this fold ---
        # Load the best model checkpoint for this fold
        best_model_path = os.path.join(
            Config.checkpoints_dir, f"best_model_fold_{fold}.pth"
        )

        # Re-initialize model and load weights
        model = trainer.model
        checkpoint = torch.load(best_model_path, map_location=Config.device)
        model.load_state_dict(checkpoint)
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(Config.device)

                # Inference (Model returns averaged logits in eval mode)
                logits = model(images)
                probs = torch.sigmoid(logits)

                fold_probs.append(probs.cpu().numpy())

        fold_probs = np.concatenate(fold_probs).ravel()

        # Map predictions back to global array
        _, val_indices = splits[fold]

        # Safety check for lengths
        if len(fold_probs) == len(val_indices):
            oof_preds[val_indices] = fold_probs
        else:
            logger.error(
                f"Shape mismatch in Fold {fold}: Preds {len(fold_probs)} vs Indices {len(val_indices)}"
            )

    # --- 4. Validation Metric ---
    # Calculate AUC on the full OOF set
    valid_mask = oof_preds != -1.0
    final_preds = oof_preds[valid_mask]
    final_targets = train_labels[valid_mask]

    final_auc = calculate_roc_auc(final_targets, final_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # --- 5. Failure Analysis ---
    logger.info("Performing Failure Analysis on Validation Set...")

    # Calculate absolute error
    errors = np.abs(final_preds - final_targets)

    # Extract features for the validation images
    val_imgs = train_images[valid_mask]

    # Normalize images to 0-1 for statistics
    val_imgs_norm = val_imgs.astype(np.float32) / 255.0

    # Compute simple statistics
    brightness = val_imgs_norm.mean(axis=(1, 2, 3))
    contrast = val_imgs_norm.std(axis=(1, 2, 3))
    red_mean = val_imgs_norm[..., 0].mean(axis=(1, 2))
    green_mean = val_imgs_norm[..., 1].mean(axis=(1, 2))
    blue_mean = val_imgs_norm[..., 2].mean(axis=(1, 2))

    features = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Red Mean": red_mean,
        "Green Mean": green_mean,
        "Blue Mean": blue_mean,
    }

    print("Failure Analysis - Correlation with Error Magnitude:")
    for name, feat_values in features.items():
        corr, _ = pearsonr(errors, feat_values)
        print(f"  {name}: {corr:.4f}")

    # --- 6. Submission ---
    threshold = 0.9889066475479729

    if final_auc > threshold:
        logger.info("Validation metric exceeds threshold. Generating submission...")
        run_inference()
    else:
        logger.warning(
            f"Validation metric {final_auc} did not exceed threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
