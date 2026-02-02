import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from timm.data import Mixup
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, save_checkpoint, load_checkpoint
from library.data import load_metadata, get_transforms, CassavaDataset
from library.model import CassavaModel, ModelEMA
from library.losses import CassavaLoss
from library.engine import train_one_epoch, validate
from library.inference import predict_test_set


def run_failure_analysis(val_df, val_loader, model, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and file size (proxy for complexity).
    """
    model.eval()
    all_targets = []
    all_probs = []

    # Get predictions
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate error magnitude (1 - probability of the correct class)
    rows = np.arange(len(all_targets))
    true_class_probs = all_probs[rows, all_targets]
    error_magnitudes = 1.0 - true_class_probs

    # Add to DataFrame
    val_df = val_df.copy()
    val_df["error_magnitude"] = error_magnitudes

    # Get file sizes as a feature
    file_sizes = []
    for path in val_df["file_path"]:
        full_path = os.path.join(Config.INPUT_DIR, path)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    val_df["file_size"] = file_sizes

    # Calculate correlation
    corr_size = val_df["error_magnitude"].corr(val_df["file_size"])

    print(f"\nFailure Analysis:")
    print(f"Correlation between Error Magnitude and File Size: {corr_size:.10f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger()
    device = Config.DEVICE

    logger.info(f"Starting execution. Device: {device}")

    # 2. Data Loading
    df_full = load_metadata("train")

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # For fast baseline, we only train Fold 0
    target_fold = 0
    best_val_metric = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_full, df_full["label"])):
        if fold != target_fold:
            continue

        logger.info(f"=== Training Fold {fold} ===")

        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        # 3. Model & Optimizer
        model = CassavaModel(pretrained=True)
        model.to(device)

        # Initialize EMA
        model_ema = None
        if Config.USE_EMA:
            model_ema = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler for total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.TOTAL_EPOCHS, eta_min=Config.MIN_LR
        )

        criterion = CassavaLoss(smoothing=Config.PHASE_2_LABEL_SMOOTHING)

        best_acc = 0.0

        # ====================================================
        # Phase 1: Coarse Feature Learning
        # ====================================================
        logger.info("--- Starting Phase 1: Coarse Feature Learning ---")

        train_tf_p1 = get_transforms("train", Config.PHASE_1_IMG_SIZE)
        train_ds_p1 = CassavaDataset(df_train, transforms=train_tf_p1)
        train_loader_p1 = DataLoader(
            train_ds_p1,
            batch_size=Config.PHASE_1_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # MixUp for Phase 1
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=Config.PHASE_1_MIXUP_PROB,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=Config.PHASE_1_LABEL_SMOOTHING,
            num_classes=Config.NUM_CLASSES,
        )

        for epoch in range(Config.PHASE_1_EPOCHS):
            train_one_epoch(
                epoch=epoch,
                model=model,
                train_loader=train_loader_p1,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                accumulation_steps=Config.PHASE_1_ACCUM_STEPS,
            )
            scheduler.step()

        # ====================================================
        # Phase Transition
        # ====================================================
        logger.info("--- Transitioning to Phase 2: Fine-Grained Refinement ---")

        # Reset EMA weights to match current model
        if model_ema is not None:
            model_ema.reset_weights(model)

        # ====================================================
        # Phase 2: Fine-Grained Refinement
        # ====================================================

        # Transforms & Loader (Higher Resolution)
        train_tf_p2 = get_transforms("train", Config.PHASE_2_IMG_SIZE)
        train_ds_p2 = CassavaDataset(df_train, transforms=train_tf_p2)
        train_loader_p2 = DataLoader(
            train_ds_p2,
            batch_size=Config.PHASE_2_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # Validation Loader (High Resolution)
        val_tf = get_transforms("val", Config.PHASE_2_IMG_SIZE)
        val_ds = CassavaDataset(df_val, transforms=val_tf)
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.PHASE_2_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        for epoch in range(Config.PHASE_1_EPOCHS, Config.TOTAL_EPOCHS):
            train_one_epoch(
                epoch=epoch,
                model=model,
                train_loader=train_loader_p2,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_ema=model_ema,
                mixup_fn=None,  # Disabled
                accumulation_steps=Config.PHASE_2_ACCUM_STEPS,
            )
            scheduler.step()

            # Validate
            val_model = model_ema.module if model_ema else model
            acc, val_loss = validate(val_model, val_loader, criterion, device)

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": val_model.state_dict(),
                        "best_acc": best_acc,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=True,
                    fold=fold,
                )

        logger.info(f"Fold {fold} finished. Best Accuracy: {best_acc:.2f}%")

        # Load best model for analysis
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth"
        )
        load_checkpoint(best_model_path, model, device=device)

        # Failure Analysis
        run_failure_analysis(df_val, val_loader, model, device)

        best_val_metric = best_acc / 100.0
        print(f"Final Validation Metric: {best_val_metric:.10f}")

    # 4. Submission
    if best_val_metric > 0.9076:
        logger.info("Validation metric threshold met. Generating submission...")
        predict_test_set()
    else:
        logger.info(
            f"Validation metric {best_val_metric:.4f} did not meet threshold 0.9076. Skipping submission."
        )


if __name__ == "__main__":
    main()
