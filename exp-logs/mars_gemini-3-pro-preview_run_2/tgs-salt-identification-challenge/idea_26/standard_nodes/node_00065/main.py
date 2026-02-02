import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import SaltDataset
from library.model import ResNet34WideLinkNet
from library.engine import train_model, validate, predict_and_submit
from library.utils import unpad_image, calc_iou

# -------------------------------------------------------------------------
# Helper Classes & Functions
# -------------------------------------------------------------------------


class SafeSaltDataset(SaltDataset):
    """
    Subclass of SaltDataset that handles missing depths (NaNs) in the test set
    by filling them with the mean of valid depths. This prevents NaN propagation
    during Multi-Task Loss calculation in Stage 3.
    """

    def _load_depths_from_disk(self):
        # Load raw depths
        vals = self.df["z"].values.astype(np.float32)

        # Check for NaNs
        mask = np.isnan(vals)
        if mask.any():
            # Calculate mean of valid values
            valid_mean = np.nanmean(vals)
            if np.isnan(valid_mean):
                valid_mean = 0.0

            # Fill NaNs
            vals[mask] = valid_mean

        return vals


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Override Config for runtime constraints (Fast Baseline)
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 32
    PATIENCE = 5

    # Ensure output directories exist
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: Multi-Task Ensemble Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STAGE 1: Multi-Task Ensemble Training")
    print("=" * 40)

    # Load full training set
    full_train_ds = SaltDataset(mode="train", load_cached_data=True)

    # K-Fold Split
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    ensemble_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_ds)):
        print(f"\n--- Training Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Create Subsets
        train_sub = Subset(full_train_ds, train_idx)
        val_sub = Subset(full_train_ds, val_idx)

        # Dataloaders
        train_loader = DataLoader(
            train_sub,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = ResNet34WideLinkNet().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train
        save_path = os.path.join(Config.CHECKPOINTS_DIR, f"fold_{fold}_best.pth")
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            epochs=Config.EPOCHS,
            patience=PATIENCE,
            save_path=save_path,
        )

        ensemble_models.append(save_path)

        # Clean up
        del model, optimizer, train_loader, val_loader, train_sub, val_sub
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # STAGE 2: Soft Pseudo-Label Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STAGE 2: Soft Pseudo-Label Generation")
    print("=" * 40)

    test_ds = SaltDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate probabilities {id: accumulated_prob_sum}
    accumulated_probs = {}

    for fold_idx, model_path in enumerate(ensemble_models):
        print(f"Predicting with model fold {fold_idx + 1}...")
        model = ResNet34WideLinkNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                ids = batch["id"]

                # Simple Inference (No TTA for speed in this stage)
                preds = model(images)["mask"]
                probs = torch.sigmoid(preds).cpu().numpy()

                for i, img_id in enumerate(ids):
                    p = probs[i].squeeze()  # 128x128
                    # Unpad to original 101x101
                    p_unpad = unpad_image(p, (Config.ORIG_HEIGHT, Config.ORIG_WIDTH))

                    if img_id not in accumulated_probs:
                        accumulated_probs[img_id] = np.zeros(
                            (Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.float32
                        )

                    accumulated_probs[img_id] += p_unpad

        del model
        torch.cuda.empty_cache()

    # Average probabilities
    for img_id in accumulated_probs:
        accumulated_probs[img_id] /= Config.N_FOLDS

    print("Pseudo-labels generated.")

    # -------------------------------------------------------------------------
    # STAGE 3: Student Distillation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" STAGE 3: Student Distillation")
    print("=" * 40)

    # Prepare datasets
    train_ds_labeled = SaltDataset(mode="train", load_cached_data=True)

    # Use SafeSaltDataset for pseudo data to handle missing depths in test set
    # Pass mean/std from labeled set to ensure consistent standardization
    pseudo_ds = SafeSaltDataset(
        mode="pseudo",
        soft_masks=accumulated_probs,
        depth_mean=train_ds_labeled.depth_mean,
        depth_std=train_ds_labeled.depth_std,
        load_cached_data=True,
    )

    combined_ds = ConcatDataset([train_ds_labeled, pseudo_ds])

    # Hold-out Validation Set (The official val.csv)
    holdout_val_ds = SaltDataset(mode="val", load_cached_data=True)

    # Loaders
    student_train_loader = DataLoader(
        combined_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    holdout_val_loader = DataLoader(
        holdout_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Student Model
    student_model = ResNet34WideLinkNet().to(device)
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    student_save_path = os.path.join(Config.CHECKPOINTS_DIR, "student_best.pth")

    # Train Student
    best_threshold = train_model(
        student_model,
        student_train_loader,
        holdout_val_loader,
        optimizer,
        device,
        epochs=Config.EPOCHS,
        patience=PATIENCE,
        save_path=student_save_path,
    )

    # -------------------------------------------------------------------------
    # Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Validation & Failure Analysis")
    print("=" * 40)

    # Load best student
    student_model.load_state_dict(torch.load(student_save_path, map_location=device))
    student_model.eval()

    # Get predictions on validation set
    val_loss, val_map, val_probs, val_masks = validate(
        student_model, holdout_val_loader, device, return_probs=True
    )

    print(f"Final Validation Metric: {val_map:.10f}")

    # Failure Analysis: Correlation between IoU and Features
    val_df = holdout_val_ds.df
    ious = []
    depths = val_df["z"].values
    coverages = val_df["salt_coverage"].values

    for prob, mask in zip(val_probs, val_masks):
        pred_bin = (prob > best_threshold).astype(np.uint8)
        iou = calc_iou(pred_bin, mask)
        ious.append(iou)

    ious = np.array(ious)

    # Handle potential NaNs in correlation calculation
    if len(ious) > 1:
        corr_depth, _ = pearsonr(depths, ious)
        corr_cov, _ = pearsonr(coverages, ious)
    else:
        corr_depth, corr_cov = 0.0, 0.0

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (IoU vs Depth): {corr_depth:.4f}")
    print(f"Correlation (IoU vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    if val_map > 0.7985:
        print("\n=== Generating Submission ===")

        # Load Final Test Set
        test_ds_final = SaltDataset(mode="test", load_cached_data=True)
        test_loader_final = DataLoader(
            test_ds_final,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict_and_submit(
            student_model,
            test_loader_final,
            device,
            best_threshold,
            Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation metric {val_map:.4f} did not meet threshold 0.7985. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
