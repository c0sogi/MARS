import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from provided libraries
from library.config import Config
from library.data import load_dataset_arrays, PathologyDataset, get_transforms
from library.models import get_model
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.engine import train_one_epoch, validate, predict


def main():
    # --- 1. Configuration & Setup ---
    seed_everything(Config.SEED)
    Config.setup()

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("Loading data...")
    # Load Train Data
    train_images, train_labels, train_ids = load_dataset_arrays(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )

    # Load Validation Data
    val_images, val_labels, val_ids = load_dataset_arrays(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    # Load Test Data
    test_images, test_labels, test_ids = load_dataset_arrays(
        Config.TEST_METADATA_PATH, "test", load_cached_data=True
    )

    # --- 3. Training Loop (Single Model) ---
    # Prioritize convergence over ensembling (Cite solution_lesson_node_00014)

    # Create Datasets
    train_dataset = PathologyDataset(
        train_images,
        train_labels,
        train_ids,
        transforms=get_transforms("train"),
    )
    val_dataset = PathologyDataset(
        val_images,
        val_labels,
        val_ids,
        transforms=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model_name = Config.MODEL_NAMES[0]
    print(f"\nTraining {model_name}...")

    model = get_model(model_name, pretrained=True).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    best_auc = 0.0
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{model_name}.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        # Removed EMA to avoid initialization bias in short/medium runs (Cite solution_lesson_node_00024)
        train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        auc = validate(model, val_loader, device)

        scheduler.step()

        if auc > best_auc:
            best_auc = auc
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                auc,
                best_ckpt_path,
            )
            print(f"  New Best AUC: {best_auc:.5f}")

    # Cleanup
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # --- 4. Final Validation on Hold-out Set ---
    print("\nRunning Final Validation on Hold-out Set...")

    print(f"Inference with {os.path.basename(best_ckpt_path)}...")
    model = get_model(model_name, pretrained=False)
    load_checkpoint(best_ckpt_path, model, device=device)
    model.to(device)

    # predict() uses 8-view TTA (Cite solution_lesson_node_00006)
    ids, preds = predict(model, val_loader, device)
    val_preds = np.array(preds)

    final_val_auc = roc_auc_score(val_labels, val_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # --- 5. Failure Analysis ---
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(val_labels - val_preds)

    # Calculate simple image stats for validation set (normalized 0-1)
    val_imgs_norm = val_images.astype(np.float32) / 255.0

    # Brightness: Mean of all channels
    brightness = val_imgs_norm.mean(axis=(1, 2, 3))
    # Contrast: Std of all channels
    contrast = val_imgs_norm.std(axis=(1, 2, 3))
    # Red Mean: Mean of Red channel (index 0)
    red_mean = val_imgs_norm[:, :, :, 0].mean(axis=(1, 2))

    # Correlations
    corr_brightness = np.corrcoef(errors, brightness)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast)[0, 1]
    corr_red = np.corrcoef(errors, red_mean)[0, 1]

    print(f"Correlation (Error vs Brightness): {corr_brightness:.4f}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast:.4f}")
    print(f"Correlation (Error vs Red Mean):   {corr_red:.4f}")

    # --- 6. Test Inference & Submission ---
    threshold = 0.9889066475479729
    if final_val_auc > threshold:
        print("\nValidation metric meets threshold. Generating submission...")

        test_dataset = PathologyDataset(
            test_images, test_labels, test_ids, transforms=get_transforms("test")
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Test Inference with {os.path.basename(best_ckpt_path)}...")
        # Model is already loaded from validation step
        ids, preds = predict(model, test_loader, device)
        test_preds = np.array(preds)

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": ids, "label": test_preds})

        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_val_auc} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
