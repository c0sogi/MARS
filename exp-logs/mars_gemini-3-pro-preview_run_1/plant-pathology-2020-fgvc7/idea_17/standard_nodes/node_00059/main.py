import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_class_weights,
    calculate_roc_auc,
    save_checkpoint,
)
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34, train_one_epoch, validate


def analyze_failures(df, targets, preds):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\n==== Failure Analysis ====")

    # Calculate Error Magnitude
    # For multi-class, we define error as 1.0 - probability assigned to the true class
    true_indices = np.argmax(targets, axis=1)
    n_samples = len(targets)
    probs_at_true = preds[np.arange(n_samples), true_indices]
    error_magnitude = 1.0 - probs_at_true

    # Extract Meta-Features
    widths = []
    heights = []
    intensities = []

    # We need to read images to get stats.
    # Since we have the dataframe, we can iterate.
    print("Extracting image meta-features for correlation analysis...")
    for idx, row in df.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for missing images (though unlikely given checks)
            widths.append(0)
            heights.append(0)
            intensities.append(0)
            continue

        h, w, c = img.shape
        # Calculate mean intensity (normalized)
        mean_intensity = img.mean() / 255.0

        widths.append(w)
        heights.append(h)
        intensities.append(mean_intensity)

    widths = np.array(widths)
    heights = np.array(heights)
    intensities = np.array(intensities)

    # Calculate Correlations
    # Handle constant arrays (std=0) to avoid NaNs in correlation
    if np.std(widths) > 0:
        corr_w, _ = pearsonr(error_magnitude, widths)
    else:
        corr_w = 0.0

    if np.std(heights) > 0:
        corr_h, _ = pearsonr(error_magnitude, heights)
    else:
        corr_h = 0.0

    if np.std(intensities) > 0:
        corr_i, _ = pearsonr(error_magnitude, intensities)
    else:
        corr_i = 0.0

    print(f"Correlation (Error vs Width):     {corr_w:.6f}")
    print(f"Correlation (Error vs Height):    {corr_h:.6f}")
    print(f"Correlation (Error vs Intensity): {corr_i:.6f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Preparation
    # Load metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine for 10-Fold Splitting
    full_df = pd.concat([train_meta, val_meta]).reset_index(drop=True)

    # Ensure stratify label exists
    if "stratify_label" not in full_df.columns:
        full_df["stratify_label"] = full_df[Config.CLASSES].idxmax(axis=1)

    # Class Weights
    class_weights = get_class_weights(Config.TRAIN_METADATA_PATH)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # OOF Storage
    oof_preds_single = np.zeros((len(full_df), Config.NUM_CLASSES))
    oof_preds_tta = np.zeros((len(full_df), Config.NUM_CLASSES))
    oof_targets = np.zeros((len(full_df), Config.NUM_CLASSES))

    print(f"Starting {Config.N_FOLDS}-Fold Training...")

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["stratify_label"])
    ):
        print(f"\n=== Fold {fold} ===")

        # Split
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # Loaders
        train_ds = AppleDataset(train_df, transforms=get_transforms("train"))
        val_ds = AppleDataset(val_df, transforms=get_transforms("valid"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        model = AppleResNet34(
            num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        model.to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config.EPOCHS, T_mult=Config.T_MULT, eta_min=Config.ETA_MIN
        )

        # Training
        best_auc = 0.0
        best_model_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")

        for epoch in range(Config.EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc, _, _ = validate(
                model, val_loader, criterion, device, use_tta=False
            )

            scheduler.step()

            # Save Checkpoint
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, best_model_path, best_auc)

        # Load Best Model for OOF
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate OOF (Single)
        _, _, preds_single, targets = validate(
            model, val_loader, criterion, device, use_tta=False
        )
        oof_preds_single[val_idx] = preds_single
        oof_targets[val_idx] = targets

        # Generate OOF (TTA)
        _, _, preds_tta, _ = validate(
            model, val_loader, criterion, device, use_tta=True
        )
        oof_preds_tta[val_idx] = preds_tta

    # 4. Validation Assessment
    auc_single = calculate_roc_auc(oof_targets, oof_preds_single)
    auc_tta = calculate_roc_auc(oof_targets, oof_preds_tta)

    final_metric = max(auc_single, auc_tta)
    use_tta = auc_tta > auc_single

    # Required Output
    print(f"Final Validation Metric: {final_metric}")
    print(
        f"Strategy Selected: {'TTA' if use_tta else 'Single View'} (Single: {auc_single:.6f}, TTA: {auc_tta:.6f})"
    )

    # 5. Failure Analysis
    # Use the predictions from the selected strategy
    best_preds = oof_preds_tta if use_tta else oof_preds_single
    analyze_failures(full_df, oof_targets, best_preds)

    # 6. Submission
    THRESHOLD = 0.9901680711448418
    if final_metric > THRESHOLD:
        print("\nMetric threshold passed. Generating submission...")

        test_df = pd.read_csv(Config.TEST_METADATA_PATH)
        test_ds = AppleDataset(
            test_df, transforms=get_transforms("test"), output_label=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        final_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

        # Ensemble Inference
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(Config.MODELS_DIR, f"resnet34_fold_{fold}.pth")
            model = AppleResNet34(num_classes=Config.NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)

                    if use_tta:
                        # Horizontal Flip
                        inputs_h = torch.flip(inputs, dims=[3])
                        outputs_h = model(inputs_h)
                        probs_h = torch.softmax(outputs_h, dim=1)

                        # Vertical Flip
                        inputs_v = torch.flip(inputs, dims=[2])
                        outputs_v = model(inputs_v)
                        probs_v = torch.softmax(outputs_v, dim=1)

                        # Average
                        probs = (probs + probs_h + probs_v) / 3.0

                    fold_preds.append(probs.cpu().numpy())

            final_preds += np.concatenate(fold_preds)

        final_preds /= Config.N_FOLDS

        # Save
        submission_df = pd.DataFrame(final_preds, columns=Config.CLASSES)
        submission_df.insert(0, "image_id", test_df["image_id"])
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric {final_metric} did not exceed threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
