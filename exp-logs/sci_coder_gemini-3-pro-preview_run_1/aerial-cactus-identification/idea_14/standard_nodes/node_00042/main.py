import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import library modules
from library.config import Config
from library import train, inference, data, utils


def main():
    """
    Orchestration script for the Cactus Identification Task.
    Executes Training, Inference, Validation, and Failure Analysis.
    """
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Monkey-patch Config for a Fast Baseline execution
    # Reducing epochs and folds to ensure completion within the 2-hour limit
    print("Configuring Fast Baseline Pipeline...")
    Config.EPOCHS = 5  # Reduced from 30
    Config.N_FOLDS = 2  # Reduced from 5 (Minimum for CV)
    Config.USE_SWA = False  # Disable SWA to save training time

    # Set seeds for reproducibility
    utils.seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Training Phase
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STARTING TRAINING PHASE ")
    print("=" * 40)
    # Run training with full data (debug=False) but reduced epochs/folds
    train.run_training(debug=False)

    # -------------------------------------------------------------------------
    # 3. Inference Phase (Submission Generation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STARTING INFERENCE PHASE ")
    print("=" * 40)
    # This generates the ./submission/submission.csv file and prints OOF metrics
    inference.run_inference(debug=False)

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" VALIDATION & FAILURE ANALYSIS ")
    print("=" * 40)

    # Load the specific hold-out validation set defined in metadata
    print(f"Loading validation set from {Config.VAL_METADATA_PATH}...")
    val_imgs, val_labels, val_fs, _ = data.load_and_cache_split(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_IMGS,
        Config.CACHE_VAL_LABELS,
        Config.CACHE_VAL_FILESIZES,
        Config.INPUT_DIR,
        load_cached=True,
    )

    # Normalize file sizes using training stats (loaded from cache generated during training)
    if os.path.exists(Config.CACHE_TRAIN_FILESIZES):
        train_fs = np.load(Config.CACHE_TRAIN_FILESIZES)
        fs_mean = np.mean(train_fs)
        fs_std = np.std(train_fs) + 1e-8
    else:
        # Fallback to local stats if training cache is missing
        fs_mean = np.mean(val_fs)
        fs_std = np.std(val_fs) + 1e-8

    val_fs_norm = (val_fs - fs_mean) / fs_std

    # Create DataLoader for Validation Set
    val_dataset = data.CactusDataset(val_imgs, val_labels, val_fs_norm, transform=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device(Config.DEVICE)

    # Collect predictions from all trained base models (Ensemble)
    all_model_preds = []

    print("Generating predictions on validation set using trained ensemble...")
    for arch in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            # Construct checkpoint path (Using 'best' model)
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"{arch}_fold{fold}_best.pth"
            )

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found: {ckpt_path}")
                continue

            try:
                # Load Model
                model = inference.get_model_instance(arch, device)
                inference.load_checkpoint_weights(model, ckpt_path, device)

                # Switch RepVGG to deploy mode (fuse layers)
                if hasattr(model, "switch_to_deploy"):
                    model.switch_to_deploy()

                # Predict (uses 4-view TTA inside predict_loader)
                preds = inference.predict_loader(model, val_loader, device)
                all_model_preds.append(preds)

                # Cleanup to save memory
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error predicting with {arch} fold {fold}: {e}")

    if not all_model_preds:
        print("Critical Error: No predictions generated for validation.")
        sys.exit(1)

    # Average predictions (Simple Averaging Ensemble)
    avg_preds = np.mean(all_model_preds, axis=0)

    # Compute Final Metric (Area Under ROC Curve)
    final_metric = roc_auc_score(val_labels, avg_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error magnitude
    errors = np.abs(val_labels - avg_preds)

    # Calculate correlation between Error and File Size
    # We use numpy's correlation coefficient matrix
    corr_matrix = np.corrcoef(errors, val_fs)
    corr = corr_matrix[0, 1]

    print(f"Failure Analysis - Correlation between Error and File Size: {corr:.6f}")

    # -------------------------------------------------------------------------
    # 5. Submission Verification
    # -------------------------------------------------------------------------
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"Submission successfully generated at: {submission_path}")
    else:
        print("Error: Submission file was not generated.")


if __name__ == "__main__":
    main()
