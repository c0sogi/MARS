import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from sklearn.model_selection import KFold
import copy
import cv2

from library.config import Config, seed_everything
from library.utils import unpad_image, pad_image, rle_encode
from library.dataset import process_data, SaltDataset, get_transforms, get_loaders
from library.models import TeacherLinkNet, StudentLinkNet
from library.training import train_model, generate_submission
from library.losses import StudentLoss, LovaszHingeLoss

# Ensure reproducibility
seed_everything(Config.SEED)


def run_cv_training(debug=False):
    """
    Runs 5-Fold Cross Validation using the TeacherLinkNet (Image+Depth).
    Optimizes for the best single-stage model performance.
    """
    print("=" * 50)
    print("Running 5-Fold Cross Validation (Image + Depth)")
    print("=" * 50)

    if debug:
        print("DEBUG Mode: Reducing epochs")
        Config.NUM_EPOCHS = 2

    # 1. Prepare Data
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Process data (loads from cache if available)
    t_imgs, t_masks, t_depths, t_ids = process_data(train_df, "train", Config.CACHE_DIR)
    v_imgs, v_masks, v_depths, v_ids = process_data(val_df, "val", Config.CACHE_DIR)

    # Combine for CV
    all_images = np.concatenate([t_imgs, v_imgs], axis=0)
    all_masks = np.concatenate([t_masks, v_masks], axis=0)
    all_depths = np.concatenate([t_depths, v_depths], axis=0)
    all_ids = np.concatenate([t_ids, v_ids], axis=0)

    # Normalize Depths (Global Stats)
    d_mean = np.mean(all_depths)
    d_std = np.std(all_depths) + 1e-8
    all_depths_norm = (all_depths - d_mean) / d_std

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    fold_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Create Datasets
        train_ds = SaltDataset(
            all_images[train_idx],
            all_depths_norm[train_idx],
            all_ids[train_idx],
            all_masks[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = SaltDataset(
            all_images[val_idx],
            all_depths_norm[val_idx],
            all_ids[val_idx],
            all_masks[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model (TeacherLinkNet - Image+Depth)
        # Cite solution_lesson_node_00032: Explicit Injection is superior to distillation
        model = TeacherLinkNet(num_classes=1).to(Config.DEVICE)

        # Train
        # Note: train_model handles saving "best_model.pth" in cache.
        # We need to rename it after training to avoid overwriting.
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=Config.DEVICE,
            config=Config,
            teacher_model=None,
            is_student=False,  # Uses Standard Lovasz+BCE
        )

        # Validate final performance
        trainer_temp = from_model_trainer(model)
        val_map, _ = trainer_temp.validate(val_loader)
        print(f"Fold {fold+1} Best mAP: {val_map:.6f}")

        # Save Fold Model
        save_name = f"fold_{fold}_model.pth"
        save_path = os.path.join(Config.CACHE_DIR, save_name)
        torch.save(model.state_dict(), save_path)
        fold_model_paths.append(save_path)
        print(f"Fold {fold+1} model saved to {save_path}")

        if debug:
            break

    return fold_model_paths


# Helper to instantiate a Trainer just for validation reuse
def from_model_trainer(model):
    from library.training import Trainer

    return Trainer(model, Config.DEVICE)
