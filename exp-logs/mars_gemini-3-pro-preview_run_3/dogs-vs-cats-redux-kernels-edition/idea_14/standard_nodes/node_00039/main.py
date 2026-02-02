import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from scipy.stats import pointbiserialr, pearsonr

# Import from library
from library.config import (
    DEVICE,
    SEED,
    WORKING_DIR,
    MODEL_RESNET,
    MODEL_CONVNEXT,
    MODEL_MAXVIT,
    IMG_SIZE_TEACHER,
    IMG_SIZE_STUDENT,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_METADATA_PATH,
    INPUT_DIR,
)
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.dataset import CatDogDataset, DualResolutionDataset, load_metadata
from library.models import create_model
from library.engine import train_one_epoch, train_distill_one_epoch, validate
from library.inference import inference_fn, predict_with_tta

# Override EPOCHS for fast baseline execution
EPOCHS = 5


def train_teacher(model_name, checkpoint_name, img_size):
    print(f"\n=== Training Teacher: {model_name} ===")

    # 1. Model
    model = create_model(model_name, pretrained=True, num_classes=1)
    model.to(DEVICE)

    # 2. Data
    train_dataset = CatDogDataset(split="train", img_size=img_size)
    val_dataset = CatDogDataset(split="val", img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")

    # 4. Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            filename=f"{checkpoint_name}_checkpoint.pth",
            best_filename=f"{checkpoint_name}_best.pth",
        )
        print(f"Epoch {epoch} finished. Best Val Loss: {best_loss:.6f}")

    return best_loss


def train_student_distillation():
    print(f"\n=== Training Student (Distillation): {MODEL_MAXVIT} ===")

    # 1. Load Teachers
    print("Loading Teachers...")
    teacher_resnet = create_model(MODEL_RESNET, pretrained=False, num_classes=1)
    load_checkpoint(teacher_resnet, "resnet_best.pth")
    teacher_resnet.to(DEVICE)
    teacher_resnet.eval()
    for p in teacher_resnet.parameters():
        p.requires_grad = False

    teacher_convnext = create_model(MODEL_CONVNEXT, pretrained=False, num_classes=1)
    load_checkpoint(teacher_convnext, "convnext_best.pth")
    teacher_convnext.to(DEVICE)
    teacher_convnext.eval()
    for p in teacher_convnext.parameters():
        p.requires_grad = False

    teachers = [teacher_resnet, teacher_convnext]

    # 2. Student Model
    student = create_model(MODEL_MAXVIT, pretrained=True, num_classes=1)
    student.to(DEVICE)

    # 3. Data
    # Training uses DualResolution for distillation
    train_dataset = DualResolutionDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation uses standard dataset at student resolution
    val_dataset = CatDogDataset(split="val", img_size=IMG_SIZE_STUDENT)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")

    # 5. Loop
    for epoch in range(1, EPOCHS + 1):
        # Train with distillation
        train_distill_one_epoch(
            student, teachers, train_loader, optimizer, DEVICE, epoch
        )

        # Validate normally
        val_loss = validate(student, val_loader, DEVICE)
        scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            filename="maxvit_checkpoint.pth",
            best_filename="maxvit_best.pth",
        )
        print(f"Epoch {epoch} finished. Best Val Loss: {best_loss:.6f}")


def validate_ensemble():
    print("\n=== Validating Ensemble ===")

    # 1. Load Models
    model_resnet = create_model(MODEL_RESNET, pretrained=False, num_classes=1)
    load_checkpoint(model_resnet, "resnet_best.pth")
    model_resnet.to(DEVICE)
    model_resnet.eval()

    model_convnext = create_model(MODEL_CONVNEXT, pretrained=False, num_classes=1)
    load_checkpoint(model_convnext, "convnext_best.pth")
    model_convnext.to(DEVICE)
    model_convnext.eval()

    model_maxvit = create_model(MODEL_MAXVIT, pretrained=False, num_classes=1)
    load_checkpoint(model_maxvit, "maxvit_best.pth")
    model_maxvit.to(DEVICE)
    model_maxvit.eval()

    # 2. Data Loaders
    # Teacher Loader (256)
    ds_teacher = CatDogDataset(split="val", img_size=IMG_SIZE_TEACHER)
    dl_teacher = DataLoader(
        ds_teacher, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Student Loader (224)
    ds_student = CatDogDataset(split="val", img_size=IMG_SIZE_STUDENT)
    dl_student = DataLoader(
        ds_student, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    all_preds = []
    all_targets = []

    # 3. Inference Loop
    with torch.no_grad():
        for (imgs_t, targets), (imgs_s, _) in zip(dl_teacher, dl_student):
            imgs_t = imgs_t.to(DEVICE)
            imgs_s = imgs_s.to(DEVICE)

            # TTA Prediction
            p_res = predict_with_tta(model_resnet, imgs_t)
            p_conv = predict_with_tta(model_convnext, imgs_t)
            p_max = predict_with_tta(model_maxvit, imgs_s)

            # Ensemble Average
            avg_prob = (p_res + p_conv + p_max) / 3.0

            all_preds.extend(avg_prob.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    # 4. Metric
    final_metric = log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    return final_metric, np.array(all_preds), np.array(all_targets)


def failure_analysis(preds, targets):
    print("\n=== Failure Analysis ===")

    # Load metadata to get features
    df = load_metadata("val", load_cached_data=True)

    # Ensure alignment (loaders are not shuffled, metadata is static)
    if len(df) != len(preds):
        print("Warning: Metadata length mismatch. Skipping detailed feature analysis.")
        return

    # Calculate absolute error
    errors = np.abs(preds - targets)
    df["error"] = errors

    # Extract features
    # We need to read image dims. Since this is "fast", we might skip reading all images again
    # if not cached. But let's try to get file size at least.

    file_sizes = []
    widths = []
    heights = []

    # Use a sample for speed if dataset is huge, but 4500 is fine.
    print("Extracting metadata features for correlation analysis...")
    import cv2

    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_DIR, row["filepath"])
        try:
            sz = os.path.getsize(full_path)
            # Only read dims for a subset to save time if needed, but let's do all
            # img = cv2.imread(full_path) # Too slow to read all for analysis in restricted time?
            # Let's just use file_size which is fast stat call
            file_sizes.append(sz)
        except:
            file_sizes.append(0)

    df["file_size"] = file_sizes

    # Correlation
    corr_size, p_size = pearsonr(df["file_size"], df["error"])
    print(f"Correlation (Error vs File Size): {corr_size:.4f} (p={p_size:.4f})")

    # Check if we have width/height in metadata from previous steps? No.
    # We will just report file size correlation as a proxy for complexity/resolution.


def main():
    set_seed(SEED)

    # 1. Train Teachers
    train_teacher(MODEL_RESNET, "resnet", IMG_SIZE_TEACHER)
    train_teacher(MODEL_CONVNEXT, "convnext", IMG_SIZE_TEACHER)

    # 2. Train Student (Distillation)
    train_student_distillation()

    # 3. Validate Ensemble
    metric, preds, targets = validate_ensemble()

    # 4. Failure Analysis
    failure_analysis(preds, targets)

    # 5. Submission
    THRESHOLD = 0.009241249605204765
    if metric < THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")
        inference_fn(
            resnet_checkpoint="resnet_best.pth",
            convnext_checkpoint="convnext_best.pth",
            maxvit_checkpoint="maxvit_best.pth",
        )
    else:
        print(
            f"\nValidation metric {metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
