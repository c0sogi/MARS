import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV2
from scipy.stats import pearsonr
import cv2
import glob

# Import provided library modules
from library.config import Config
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.engine import train_one_epoch, validate
from library.utils import seed_everything, get_logger, get_score

# --- Constants for Fast Baseline ---
NUM_TRAIN_SAMPLES = 15000  # Subsample for speed
NUM_EPOCHS = 6  # Reduced epochs for fast baseline


def main():
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger(name="main")
    device = Config.device
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Loading metadata...")
    df_train = pd.read_csv(Config.train_metadata)

    # Apply Fast Baseline Constraint: Subsample training data
    if len(df_train) > NUM_TRAIN_SAMPLES:
        logger.info(
            f"Subsampling training data from {len(df_train)} to {NUM_TRAIN_SAMPLES} for fast baseline."
        )
        df_train = df_train.sample(
            n=NUM_TRAIN_SAMPLES, random_state=Config.seed
        ).reset_index(drop=True)

    # Save temporary subsampled metadata for the Dataset class to read
    temp_train_meta = os.path.join(Config.output_dir, "train_subsampled.csv")
    df_train.to_csv(temp_train_meta, index=False)

    # Initialize Datasets
    train_dataset = CatheterDataset(
        metadata_path=temp_train_meta,
        transform=get_transforms(data_type="train"),
        is_test=False,
    )

    val_dataset = CatheterDataset(
        metadata_path=Config.val_metadata,
        transform=get_transforms(data_type="val"),
        is_test=False,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    logger.info("Initializing model...")
    model = CatheterModel()
    model.to(device)

    # Exponential Moving Average
    ema_model = None
    if Config.use_ema:
        ema_model = ModelEmaV2(model, decay=Config.ema_decay)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # 4. Training Loop
    best_score = 0.0
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")

    logger.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # Train
        avg_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch, ema_model
        )

        # Validate (using EMA if available)
        eval_model = ema_model.module if ema_model else model
        val_loss, val_score = validate(eval_model, val_loader, device)

        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(eval_model.state_dict(), best_model_path)
            logger.info(f"New best model saved with AUC: {best_score:.4f}")

    # 5. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get predictions on validation set for metrics and failure analysis
    # We need to run this manually to get raw preds and targets aligned
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    final_metric = get_score(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    logger.info("Performing Failure Analysis...")

    # Calculate Mean Absolute Error per sample
    # Shape: (N_samples, N_classes) -> (N_samples,)
    errors = np.mean(np.abs(y_true - y_pred), axis=1)

    # Get metadata for correlation
    # We need to read image dimensions from the validation files
    val_df = pd.read_csv(Config.val_metadata)
    widths = []
    heights = []
    aspect_ratios = []

    # Limit failure analysis metadata reading to a subset if it's too slow,
    # but 5000 files is fast enough.
    for idx, row in val_df.iterrows():
        img_path = os.path.join(Config.input_dir, row["file_path"])
        # Read header only if possible, but cv2 reads full.
        # For speed, we assume standard reading is acceptable or use a quick PIL check.
        # Given the constraints, we'll read with cv2 as it's already imported.
        try:
            img = cv2.imread(img_path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            else:
                # Fallback for missing images (shouldn't happen based on metadata check)
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate correlations
    # Filter out invalid reads
    valid_mask = widths > 0

    if np.sum(valid_mask) > 0:
        corr_width, _ = pearsonr(errors[valid_mask], widths[valid_mask])
        corr_height, _ = pearsonr(errors[valid_mask], heights[valid_mask])
        corr_ar, _ = pearsonr(errors[valid_mask], aspect_ratios[valid_mask])

        print(f"Correlation Error vs Width: {corr_width:.4f}")
        print(f"Correlation Error vs Height: {corr_height:.4f}")
        print(f"Correlation Error vs Aspect Ratio: {corr_ar:.4f}")
    else:
        print("Could not perform failure analysis due to image reading errors.")

    # 6. Submission
    THRESHOLD = 0.9563622421530574

    if final_metric > THRESHOLD:
        logger.info(
            f"Validation metric {final_metric} > {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_dataset = CatheterDataset(
            metadata_path=Config.test_metadata,
            transform=get_transforms(data_type="test"),
            is_test=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        test_preds = []
        test_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)

                test_preds.append(preds.cpu().numpy())
                test_ids.extend(ids)

        test_preds = np.concatenate(test_preds, axis=0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
        submission_df.insert(0, "StudyInstanceUID", test_ids)

        # Save
        submission_df.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")

    else:
        logger.info(
            f"Validation metric {final_metric} <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
