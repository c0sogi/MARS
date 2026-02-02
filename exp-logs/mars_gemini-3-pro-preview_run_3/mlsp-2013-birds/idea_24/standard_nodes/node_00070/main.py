import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# Try importing IterativeStratification, fallback if necessary
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_SKMULTILEARN = True
except ImportError:
    HAS_SKMULTILEARN = False

# Import library functions
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import load_data, BirdDataset, get_transforms
from library.models import get_bird_model
from library.optimization import get_optimizer_with_llrd
from library.training import train_one_epoch, validate_one_epoch
from library.inference import (
    predict_with_tta,
    save_submission,
    load_and_average_checkpoints,
)

# Constants
BATCH_SIZE = 32
NUM_EPOCHS = 12  # Fast baseline configuration
FOLDS = 5
MODELS = ["resnet18", "efficientnet_b0", "densenet121"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.9479806884980326
CHECKPOINT_DIR = "./working/checkpoints"
SUBMISSION_PATH = "./submission/submission.csv"


def main():
    # 1. Setup
    seed_everything(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Data Loading
    # Load train and validation metadata and combine for CV
    t_img, t_lbl, t_id = load_data(
        "./metadata/train.csv", split_name="train", load_cached_data=True
    )
    v_img, v_lbl, v_id = load_data(
        "./metadata/val.csv", split_name="val", load_cached_data=True
    )

    # Combine datasets
    X_all = np.concatenate([t_img, v_img], axis=0)
    y_all = np.concatenate([t_lbl, v_lbl], axis=0)
    id_all = np.concatenate([t_id, v_id], axis=0)

    # Load Test Data
    test_img, test_lbl, test_id = load_data(
        "./metadata/test.csv", split_name="test", load_cached_data=True
    )

    # 3. Cross-Validation Splitting
    # Dummy X for stratifier (it only needs shape[0] and y)
    dummy_X = np.zeros((len(y_all), 1))

    splits = []
    if HAS_SKMULTILEARN:
        try:
            stratifier = IterativeStratification(n_splits=FOLDS, order=1)
            splits = list(stratifier.split(dummy_X, y_all))
        except Exception as e:
            print(f"IterativeStratification failed: {e}. Falling back to KFold.")

    if not splits:
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
        splits = list(kf.split(dummy_X, y_all))

    # 4. Training Loop
    # Accumulator for OOF predictions (N_samples, N_classes)
    # We will average the predictions from all architectures for each sample
    ensemble_oof_preds = np.zeros_like(y_all)

    # To handle potential overlapping indices if splitting goes wrong (unlikely),
    # or just to be safe, we can track counts, but with standard folds, indices are unique.

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n=== Fold {fold_idx}/{FOLDS} ===")

        # Create Datasets and Loaders
        train_ds = BirdDataset(
            X_all[train_idx],
            y_all[train_idx],
            id_all[train_idx],
            transforms=get_transforms("train"),
        )
        val_ds = BirdDataset(
            X_all[val_idx],
            y_all[val_idx],
            id_all[val_idx],
            transforms=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Store predictions from each architecture for this fold
        fold_arch_preds = []

        for model_name in MODELS:
            print(f"Training {model_name}...")

            # Initialize Model
            model = get_bird_model(model_name, num_classes=19).to(DEVICE)

            # Initialize Optimizer (LLRD)
            optimizer = get_optimizer_with_llrd(
                model, model_name, lr=1e-3, weight_decay=1e-2, layer_decay=0.9
            )

            # Initialize Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS
            )

            # Track Top-3 Checkpoints
            top3_checkpoints = []  # List of (auc, path)

            for epoch in range(1, NUM_EPOCHS + 1):
                # Train
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, DEVICE, epoch
                )

                # Validate
                val_loss, val_auc = validate_one_epoch(model, val_loader, DEVICE)

                # Step Scheduler
                scheduler.step()

                # Save Checkpoint
                ckpt_path = os.path.join(
                    CHECKPOINT_DIR, f"{model_name}_fold{fold_idx}_ep{epoch}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)

                # Update Top-3
                top3_checkpoints.append((val_auc, ckpt_path))
                top3_checkpoints.sort(key=lambda x: x[0], reverse=True)

                # Prune if > 3
                if len(top3_checkpoints) > 3:
                    to_remove = top3_checkpoints.pop()
                    if os.path.exists(to_remove[1]):
                        os.remove(to_remove[1])

            # Generate OOF predictions for this architecture using Top-3 Ensemble
            print(f"Generating OOF for {model_name} using Top-3 snapshots...")
            arch_oof_preds = np.zeros((len(val_idx), 19))

            for _, ckpt_path in top3_checkpoints:
                # Load weights
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                # Predict with TTA
                probs, _ = predict_with_tta(model, val_loader, DEVICE)
                arch_oof_preds += probs

            # Average over snapshots
            arch_oof_preds /= len(top3_checkpoints)
            fold_arch_preds.append(arch_oof_preds)

            # Cleanup
            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Average across architectures for this fold
        avg_fold_preds = np.mean(fold_arch_preds, axis=0)

        # Store in global OOF matrix
        ensemble_oof_preds[val_idx] = avg_fold_preds

    # 5. Global Validation Assessment
    final_auc = calculate_roc_auc(y_all, ensemble_oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate per-sample error (Mean Absolute Error across classes)
    per_sample_error = np.mean(np.abs(y_all - ensemble_oof_preds), axis=1)

    # Calculate Image Statistics (Brightness and Contrast)
    # X_all is (N, 224, 224, 3) uint8
    imgs_float = X_all.astype(np.float32) / 255.0
    # Mean across spatial dimensions and channels
    img_means = np.mean(imgs_float, axis=(1, 2, 3))
    # Std across spatial dimensions and channels (proxy for contrast)
    img_stds = np.std(imgs_float, axis=(1, 2, 3))

    # Correlations
    corr_brightness = np.corrcoef(per_sample_error, img_means)[0, 1]
    corr_contrast = np.corrcoef(per_sample_error, img_stds)[0, 1]

    print(f"Correlation (Error vs Brightness): {corr_brightness}")
    print(f"Correlation (Error vs Contrast): {corr_contrast}")

    # 7. Submission Generation
    if final_auc > THRESHOLD:
        print("\nThreshold met. Generating submission...")

        # Prepare Test Loader
        test_ds = BirdDataset(
            test_img, test_lbl, test_id, transforms=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        final_test_probs = np.zeros((len(test_img), 19))

        # Iterate over architectures
        for model_name in MODELS:
            # Gather all checkpoints for this model (all folds, all top-3 snapshots)
            # Pattern: {model_name}_fold*_ep*.pth
            pattern = os.path.join(CHECKPOINT_DIR, f"{model_name}_fold*_ep*.pth")
            checkpoints = glob.glob(pattern)

            if not checkpoints:
                print(f"Warning: No checkpoints found for {model_name}")
                continue

            # Use library function to average predictions for this architecture
            # This function handles model instantiation, loading, TTA, and averaging
            probs, rec_ids = load_and_average_checkpoints(
                model_name, checkpoints, test_loader, DEVICE
            )

            final_test_probs += probs

        # Average across architectures
        final_test_probs /= len(MODELS)

        # Save submission
        save_submission(rec_ids, final_test_probs, SUBMISSION_PATH)

    else:
        print(f"\nThreshold {THRESHOLD} not met. Skipping submission.")


if __name__ == "__main__":
    main()
