import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import cv2

# Import from provided library files
from library.config import Config
from library.dataset import load_data, AppleDataset, get_transforms
from library.model import AppleConvNeXt
from library.engine import train_one_epoch, valid_one_epoch, inference
from library.utils import (
    seed_everything,
    calculate_f1_score,
    save_checkpoint,
    get_logger,
)


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for fast baseline execution as per requirements
    Config.EPOCHS = 6  # Reduced from 15 to ensure completion within time limit
    Config.DEBUG = False  # Use full dataset to achieve the required score

    # Create directories
    Config.create_dirs()

    # Setup Logger and Seeds
    logger = get_logger(Config.LOG_PATH)
    seed_everything(Config.SEED)

    logger.info("Starting Fast Baseline Run...")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Epochs: {Config.EPOCHS}")

    # --- 2. Data Loading ---
    logger.info("Loading Data...")
    # Load metadata with caching enabled
    df_train = load_data(
        Config.TRAIN_CSV, "train", debug=Config.DEBUG, load_cached_data=True
    )
    df_val = load_data(Config.VAL_CSV, "val", debug=Config.DEBUG, load_cached_data=True)

    # Create Datasets
    train_dataset = AppleDataset(df_train, transforms=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transforms=get_transforms("valid"))

    # Create DataLoaders
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

    # --- 3. Model & Optimization ---
    logger.info("Initializing Model...")
    model = AppleConvNeXt(pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=Config.USE_AMP)

    # --- 4. Training Loop ---
    best_f1 = 0.0

    for epoch in range(Config.EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(
            epoch,
            model,
            optimizer,
            scheduler,
            criterion,
            train_loader,
            Config.DEVICE,
            scaler,
            logger,
        )

        # Validate
        val_loss, val_f1 = valid_one_epoch(
            epoch, model, criterion, val_loader, Config.DEVICE, logger
        )

        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            logger.info(f"New Best F1: {best_f1:.6f}. Saving model...")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                },
                Config.MODEL_PATH,
            )

    # --- 5. Final Validation & Metric Calculation ---
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_preds = []
    val_targets = []

    # Optimized inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(Config.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(images)

            # Store probabilities and targets
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Calculate Final Metric
    final_metric = calculate_f1_score(
        val_targets, val_preds, threshold=Config.THRESHOLD
    )
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    logger.info("\nRunning Failure Analysis...")

    # Calculate Error Magnitude (Mean Absolute Error per sample)
    # Shape: (N_samples,)
    errors = np.mean(np.abs(val_preds - val_targets), axis=1)

    # Extract Input Features (Width, Height, Aspect Ratio)
    widths = []
    heights = []

    # Efficiently read image dimensions
    for _, row in df_val.iterrows():
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        # Use OpenCV to read header/image.
        # Note: imread loads the whole image, but it's fast enough for ~3000 images on this hardware.
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
        else:
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)

    # Handle division by zero for aspect ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratios = np.true_divide(widths, heights)
        aspect_ratios[~np.isfinite(aspect_ratios)] = 0

    # Calculate Correlations
    def get_correlation(x, y):
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    corr_w = get_correlation(errors, widths)
    corr_h = get_correlation(errors, heights)
    corr_ar = get_correlation(errors, aspect_ratios)

    print("Failure Analysis - Error Correlation with Input Features:")
    print(f"Width Correlation: {corr_w}")
    print(f"Height Correlation: {corr_h}")
    print(f"Aspect Ratio Correlation: {corr_ar}")

    # --- 7. Submission Generation ---
    THRESHOLD_SCORE = 0.9187550291577454

    if final_metric > THRESHOLD_SCORE:
        logger.info(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD_SCORE}). Generating Submission..."
        )

        # Load Test Data
        df_test = load_data(
            Config.TEST_CSV, "test", debug=Config.DEBUG, load_cached_data=True
        )
        test_dataset = AppleDataset(
            df_test, transforms=get_transforms("test"), output_label=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_preds = inference(model, test_loader, Config.DEVICE)

        # Format Submission
        submission_rows = []
        for i, row in df_test.iterrows():
            probs = test_preds[i]
            active_indices = np.where(probs > Config.THRESHOLD)[0]

            if len(active_indices) == 0:
                # Fallback to max probability class if no threshold crossed
                top_idx = np.argmax(probs)
                labels_str = Config.CLASSES[top_idx]
            else:
                labels = [Config.CLASSES[idx] for idx in active_indices]
                labels_str = " ".join(labels)

            submission_rows.append({"image": row["image"], "labels": labels_str})

        df_sub = pd.DataFrame(submission_rows)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"\nMetric ({final_metric}) did not pass threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
