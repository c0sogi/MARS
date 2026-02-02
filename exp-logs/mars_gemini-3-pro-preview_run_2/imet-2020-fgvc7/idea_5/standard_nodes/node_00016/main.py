import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import cv2

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_device, MetricMonitor
from library.dataset import load_dataset, ArtworkDataset
from library.models import get_model, ModelEMA
from library.engine import (
    train_one_epoch,
    validate,
    generate_soft_labels,
    inference,
    find_best_threshold,
    create_submission,
)

# --- Runtime Constraints ---
# Limiting sample size and epochs to ensure execution within 57 minutes
# 15,000 samples * 5 epochs * 3 models fits comfortably within the time budget on A100.
MAX_TRAIN_SAMPLES = 15000
TRAIN_EPOCHS = 5


def get_optimizer_scheduler(model, num_train_steps):
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        total_steps=num_train_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    return optimizer, scheduler


def train_model(model_name, save_path, train_loader, val_loader, device, use_ema=False):
    print(f"\n=== Training {model_name} ===")
    model = get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    ema_model = None
    if use_ema:
        ema_model = ModelEMA(model)

    num_train_steps = len(train_loader) * TRAIN_EPOCHS
    optimizer, scheduler = get_optimizer_scheduler(model, num_train_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_loss = float("inf")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            use_ema=use_ema,
            ema_model=ema_model,
        )

        # Validate
        # Use EMA model for validation if available
        eval_model = ema_model.ema if ema_model else model
        val_loss, _, _ = validate(eval_model, val_loader, device)

        # Save best model based on Minimum Validation Loss (Calibration focus)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(eval_model.state_dict(), save_path)
            print(f"  Saved Best Model (Loss: {best_val_loss:.4f})")

    # Load best state
    model.load_state_dict(torch.load(save_path))
    return model


def perform_failure_analysis(val_df, preds, targets, threshold, input_dir):
    print("\n=== Failure Analysis ===")

    # Calculate per-sample F1
    preds_bin = (preds > threshold).astype(int)

    errors = []
    widths = []
    heights = []
    file_sizes = []

    # Analyze a subset to save time
    subset_indices = np.random.choice(
        len(val_df), size=min(len(val_df), 2000), replace=False
    )

    print(f"Analyzing {len(subset_indices)} validation samples...")

    for idx in subset_indices:
        # Error magnitude: 1.0 - F1 score for this sample
        # Handle zero division if both target and pred are empty (perfect match)
        t = targets[idx]
        p = preds_bin[idx]

        # Simple F1 calculation for single sample
        tp = np.sum((t == 1) & (p == 1))
        fp = np.sum((t == 0) & (p == 1))
        fn = np.sum((t == 1) & (p == 0))

        if tp + fp + fn == 0:
            f1 = 1.0
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)

        errors.append(1.0 - f1)

        # Get Metadata
        row = val_df.iloc[idx]
        path = os.path.join(input_dir, row["file_path"])

        try:
            size = os.path.getsize(path)
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0
        except Exception:
            size, h, w = 0, 0, 0

        widths.append(w)
        heights.append(h)
        file_sizes.append(size)

    # Correlations
    df_analysis = pd.DataFrame(
        {"error": errors, "width": widths, "height": heights, "file_size": file_sizes}
    )

    # Drop failures
    df_analysis = df_analysis[df_analysis["width"] > 0]

    if len(df_analysis) > 10:
        corr_w, _ = pearsonr(df_analysis["error"], df_analysis["width"])
        corr_h, _ = pearsonr(df_analysis["error"], df_analysis["height"])
        corr_s, _ = pearsonr(df_analysis["error"], df_analysis["file_size"])

        print("Correlation between Error Magnitude (1-F1) and Input Features:")
        print(f"  Width: {corr_w:.4f}")
        print(f"  Height: {corr_h:.4f}")
        print(f"  File Size: {corr_s:.4f}")
    else:
        print("Insufficient data for correlation analysis.")


def main():
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 1. Data Loading
    # We manually slice the dataframe to limit training time
    print("Loading datasets...")
    train_dataset_full = load_dataset("train")

    # Slice training data
    if len(train_dataset_full.df) > MAX_TRAIN_SAMPLES:
        print(f"Limiting training data to {MAX_TRAIN_SAMPLES} samples for speed.")
        train_dataset_full.df = train_dataset_full.df.iloc[
            :MAX_TRAIN_SAMPLES
        ].reset_index(drop=True)

    train_loader = DataLoader(
        train_dataset_full,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation set (use full set for accurate metrics)
    val_dataset = load_dataset("val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Train Teachers
    # Teacher 1: ConvNeXt Small
    train_model(
        Config.TEACHER_MODEL_1,
        Config.TEACHER_1_CHECKPOINT,
        train_loader,
        val_loader,
        device,
    )

    # Teacher 2: Swin Base (Use EMA for stability)
    train_model(
        Config.TEACHER_MODEL_2,
        Config.TEACHER_2_CHECKPOINT,
        train_loader,
        val_loader,
        device,
        use_ema=True,
    )

    # 3. Generate Soft Labels
    print("\n=== Generating Soft Labels ===")
    teacher1 = get_model(Config.TEACHER_MODEL_1, num_classes=Config.NUM_CLASSES)
    teacher1.load_state_dict(torch.load(Config.TEACHER_1_CHECKPOINT))

    teacher2 = get_model(Config.TEACHER_MODEL_2, num_classes=Config.NUM_CLASSES)
    teacher2.load_state_dict(torch.load(Config.TEACHER_2_CHECKPOINT))

    # Create a deterministic loader for soft label generation (no shuffle, no aug)
    # We must use the same subset as training
    soft_gen_dataset = load_dataset(
        "train", transform=None
    )  # Uses default train transform, need Val transform for consistency?
    # Actually, for distillation, predicting on augmented views (train transform) is often better (consistency regularization).
    # But to align with the code in `generate_soft_labels` which iterates, we need to ensure order.
    # The `load_dataset` creates a new instance. We need to apply the same slicing.
    if len(soft_gen_dataset.df) > MAX_TRAIN_SAMPLES:
        soft_gen_dataset.df = soft_gen_dataset.df.iloc[:MAX_TRAIN_SAMPLES].reset_index(
            drop=True
        )

    # Use deterministic transform for label generation to reduce noise?
    # Idea 5 says "Soft-Target Generation... serve as ground truth".
    # Usually we want clean predictions. Let's use the validation transform (deterministic).
    from library.dataset import get_transforms

    soft_gen_dataset.transform = get_transforms(mode="val")

    soft_gen_loader = DataLoader(
        soft_gen_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    generate_soft_labels(
        [teacher1, teacher2],
        soft_gen_loader,
        device,
        Config.TEACHER_PREDS_PATH,
        load_cached_data=False,  # Force regeneration
    )

    # Free memory
    del teacher1, teacher2, soft_gen_loader, soft_gen_dataset
    torch.cuda.empty_cache()

    # 4. Train Student
    print("\n=== Training Student (Distillation) ===")
    # Reload dataset with soft labels
    student_dataset = load_dataset("train", use_soft_labels=True)
    if len(student_dataset.df) > MAX_TRAIN_SAMPLES:
        student_dataset.df = student_dataset.df.iloc[:MAX_TRAIN_SAMPLES].reset_index(
            drop=True
        )
        # Soft labels are loaded inside dataset init, but if we slice DF, we must slice soft labels too
        # The dataset class handles slicing if we pass debug=True, but here we do manual slicing.
        # We need to manually slice the soft_labels in the dataset object.
        if student_dataset.soft_labels is not None:
            student_dataset.soft_labels = student_dataset.soft_labels[
                :MAX_TRAIN_SAMPLES
            ]

    student_loader = DataLoader(
        student_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,  # ConvNeXt Large needs smaller batch maybe? Config has 24.
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    student_model = train_model(
        Config.STUDENT_MODEL,
        Config.STUDENT_CHECKPOINT,
        student_loader,
        val_loader,
        device,
    )

    # 5. Final Validation
    print("\n=== Final Validation ===")
    # Load best student
    student_model.load_state_dict(torch.load(Config.STUDENT_CHECKPOINT))
    student_model.eval()

    val_loss, val_preds, val_targets = validate(student_model, val_loader, device)

    # Optimize Threshold
    best_threshold = find_best_threshold(val_targets, val_preds)

    # Calculate Final Metric
    val_preds_bin = (val_preds > best_threshold).astype(int)
    final_metric = f1_score(val_targets, val_preds_bin, average="micro")

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    perform_failure_analysis(
        val_dataset.df, val_preds, val_targets, best_threshold, Config.INPUT_DIR
    )

    # 7. Submission
    TARGET_METRIC = 0.6566335249339754

    if final_metric > TARGET_METRIC:
        print("\nMetric threshold met. Generating submission...")
        test_dataset = load_dataset("test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        ids, probs = inference(
            student_model, test_loader, device, use_tta=Config.TTA_FLIP
        )
        create_submission(ids, probs, best_threshold, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric {final_metric} did not meet target {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()
