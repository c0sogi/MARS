import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config
from library.dataset import load_metadata_df, get_transforms, ArtworkDataset
from library.model import ArtworkModel
from library.train import train_one_epoch, validate, find_best_threshold
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    ModelEMA,
    calculate_f1,
)


def run():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Override Config for Fast Baseline
    Config.EPOCHS = 5  # Reduced from 18 to ensure < 2 hours runtime

    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading metadata...")
    train_df = load_metadata_df("train", load_cached_data=True)
    val_df = load_metadata_df("val", load_cached_data=True)

    # Datasets
    train_dataset = ArtworkDataset(
        train_df, mode="train", transforms=get_transforms("train", Config.IMG_SIZE)
    )
    val_dataset = ArtworkDataset(
        val_df, mode="val", transforms=get_transforms("val", Config.IMG_SIZE)
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = ArtworkModel(pretrained=True)
    model.to(device)

    # EMA
    ema_model = None
    if Config.USE_EMA:
        print("Initializing ModelEMA...")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)
    scaler = GradScaler(enabled=Config.USE_AMP)

    # Loss
    pos_weight = torch.ones([Config.NUM_CLASSES], device=device) * Config.POS_WEIGHT
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_f1 = 0.0
    best_threshold = 0.5
    checkpoint_path = "best_model_runfile.pth"

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ema_model
        )

        # Scheduler
        scheduler.step()

        # Validate (use EMA if available)
        val_model = ema_model.ema if ema_model else model
        val_loss, val_preds, val_targets = validate(
            val_model, val_loader, criterion, device
        )

        # Threshold Search
        current_f1, current_thresh = find_best_threshold(val_preds, val_targets)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch} | Time: {elapsed:.1f}s | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | F1: {current_f1:.4f} @ {current_thresh:.2f}"
        )

        # Save Best
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = current_thresh
            save_checkpoint(
                val_model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                filename=checkpoint_path,
            )

    # -------------------------------------------------------------------------
    # 5. Final Validation & Metric
    # -------------------------------------------------------------------------
    print("\nLoading best model for final evaluation...")
    # Load weights into the main model structure for inference
    best_model = ArtworkModel(pretrained=False)
    best_model.to(device)
    load_checkpoint(
        os.path.join(Config.WORKING_DIR, checkpoint_path), best_model, device=device
    )

    # Final Validation Pass
    val_loss, val_preds, val_targets = validate(
        best_model, val_loader, criterion, device
    )

    # Recalculate F1 with the best threshold found
    binary_preds = (val_preds > best_threshold).astype(int)
    final_metric = calculate_f1(binary_preds, val_targets)

    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")

    # Calculate per-sample F1 (or use Hamming loss, here we use 1 - F1)
    # Note: f1_score with average=None returns per-class F1, not per-sample.
    # We need to loop or use vectorized operations.
    # Simple per-sample accuracy: Intersection / Union (Jaccard) or F1

    # Vectorized per-sample F1 calculation
    # TP = sum(pred * target)
    # FP = sum(pred * (1-target))
    # FN = sum((1-pred) * target)
    # F1 = 2TP / (2TP + FP + FN)

    tp = np.sum(binary_preds * val_targets, axis=1)
    fp = np.sum(binary_preds * (1 - val_targets), axis=1)
    fn = np.sum((1 - binary_preds) * val_targets, axis=1)

    epsilon = 1e-7
    sample_f1s = (2 * tp) / (2 * tp + fp + fn + epsilon)
    error_magnitude = 1.0 - sample_f1s

    # Feature 1: Label Cardinality (Number of Ground Truth Labels)
    label_counts = np.sum(val_targets, axis=1)

    # Feature 2: Image Dimensions (Sampled for speed)
    # We will sample 1000 images from validation to check correlation
    sample_indices = np.random.choice(
        len(val_df), size=min(1000, len(val_df)), replace=False
    )

    widths = []
    heights = []
    ratios = []
    sampled_errors = []

    print(f"Sampling {len(sample_indices)} images for dimension analysis...")
    for idx in sample_indices:
        rel_path = val_df.iloc[idx]["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path)
        if img is not None:
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            ratios.append(w / h)
            sampled_errors.append(error_magnitude[idx])

    # Correlations
    corr_cardinality, _ = pearsonr(label_counts, error_magnitude)
    print(f"Correlation (Error vs Label Count): {corr_cardinality:.4f}")

    if len(widths) > 0:
        corr_width, _ = pearsonr(widths, sampled_errors)
        corr_height, _ = pearsonr(heights, sampled_errors)
        corr_ratio, _ = pearsonr(ratios, sampled_errors)
        print(f"Correlation (Error vs Width): {corr_width:.4f}")
        print(f"Correlation (Error vs Height): {corr_height:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ratio:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    TARGET_METRIC = 0.6106623748931248

    if final_metric > TARGET_METRIC:
        print(f"\nMetric {final_metric} > {TARGET_METRIC}. Generating submission...")

        # Load Test Data
        test_df = load_metadata_df("test", load_cached_data=True)
        test_dataset = ArtworkDataset(
            test_df, mode="test", transforms=get_transforms("val", Config.IMG_SIZE)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=False,
        )

        # Inference
        best_model.eval()
        all_ids = []
        all_pred_strings = []

        with torch.no_grad():
            for images, img_ids in test_loader:
                images = images.to(device)
                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    logits = best_model(images)

                probs = torch.sigmoid(logits).cpu().numpy()
                binary = (probs > best_threshold).astype(int)

                for i in range(len(img_ids)):
                    # Get indices where binary is 1
                    indices = np.where(binary[i] == 1)[0]
                    # Join with space
                    pred_str = " ".join(map(str, indices))
                    all_ids.append(img_ids[i])
                    all_pred_strings.append(pred_str)

        # Create DataFrame
        submission_df = pd.DataFrame({"id": all_ids, "attribute_ids": all_pred_strings})

        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(f"\nMetric {final_metric} <= {TARGET_METRIC}. Skipping submission.")


if __name__ == "__main__":
    run()
