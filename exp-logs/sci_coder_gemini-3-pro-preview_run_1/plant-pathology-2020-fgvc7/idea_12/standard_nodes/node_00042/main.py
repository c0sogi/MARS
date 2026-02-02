import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import (
    seed_everything,
    get_class_weights,
    calculate_roc_auc,
    AverageMeter,
)
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import DistillationLoss
from library.engine import train_one_epoch, validate, inference, generate_submission


def run_phase_1_teacher_generation(cfg, df_train, device):
    """
    Phase 1: Stratified 5-Fold CV to determine E_opt and generate OOF logits.
    """
    print("==== Phase 1: Teacher Generation (5-Fold CV) ====")

    # Prepare for Stratified Split
    targets = df_train[cfg.target_cols].values
    y = targets.argmax(axis=1)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    # Storage for results
    fold_scores = np.zeros((cfg.n_folds, cfg.epochs))
    oof_logits_dict = {}

    # Directory for temporary checkpoints
    ckpt_dir = os.path.join(cfg.working_dir, "phase1_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pre-calculate class weights
    class_weights = get_class_weights(
        df_train, cfg.target_cols, cache_dir=cfg.working_dir
    ).to(device)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y)):
        print(f"-- Fold {fold + 1}/{cfg.n_folds} --")

        df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = AppleDataset(
            df_fold_train, cfg, transform=get_transforms("train", cfg), mode="standard"
        )
        val_ds = AppleDataset(
            df_fold_val, cfg, transform=get_transforms("valid", cfg), mode="standard"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        # Model
        model = get_model(cfg, pretrained=True)

        # Optimizer & Scheduler
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
        )

        # Loss (Standard Cross Entropy for Teacher)
        loss_fn = DistillationLoss(class_weights=class_weights, alpha=1.0)

        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, loss_fn, scheduler
            )
            val_loss, val_score, val_logits, _ = validate(
                model, val_loader, device, loss_fn
            )

            fold_scores[fold, epoch - 1] = val_score
            scheduler.step()

            # Save checkpoint for this epoch to retrieve later for OOF
            ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{epoch}.pth")
            torch.save(
                {"logits": val_logits, "ids": df_fold_val["image_id"].values}, ckpt_path
            )

    # Calculate E_opt
    mean_scores = fold_scores.mean(axis=0)
    e_opt = int(np.argmax(mean_scores) + 1)
    best_score = mean_scores.max()

    print(f"Global Optimal Epoch (E_opt): {e_opt} with Mean AUC: {best_score:.4f}")

    # Collect OOF Logits corresponding to E_opt
    print("Collecting OOF logits from E_opt...")
    for fold in range(cfg.n_folds):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_epoch_{e_opt}.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        ids = checkpoint["ids"]
        logits = checkpoint["logits"]

        for img_id, logit in zip(ids, logits):
            oof_logits_dict[img_id] = logit

    return e_opt, oof_logits_dict


def run_phase_2_student_training(cfg, df_train, e_opt, teacher_logits, device):
    """
    Phase 2: Train Student on full data using Distillation.
    """
    print(f"==== Phase 2: Student Training (Full Data, {e_opt} epochs) ====")

    # Full Dataset with Distillation Mode
    train_ds = AppleDataset(
        df_train,
        cfg,
        transform=get_transforms("train", cfg),
        mode="distillation",
        teacher_logits=teacher_logits,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model
    model = get_model(cfg, pretrained=True)

    # Optimizer & Scheduler - T_max synchronized with E_opt
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=e_opt, eta_min=cfg.min_lr
    )

    # Loss: Distillation
    class_weights = get_class_weights(
        df_train, cfg.target_cols, cache_dir=cfg.working_dir
    ).to(device)
    loss_fn = DistillationLoss(
        class_weights=class_weights,
        alpha=cfg.distillation_alpha,
        temperature=cfg.temperature,
    )

    for epoch in range(1, e_opt + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scheduler
        )
        scheduler.step()
        print(f"Epoch {epoch}/{e_opt} Train Loss: {train_loss:.4f}")

    return model


def failure_analysis(model, df_val, cfg, device):
    """
    Performs failure analysis on the validation set.
    """
    print("\n==== Failure Analysis ====")

    ds = AppleDataset(
        df_val, cfg, transform=get_transforms("valid", cfg), mode="standard"
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    all_losses = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            losses = loss_fn(logits, labels)

            all_losses.extend(losses.cpu().numpy())

    df_val = df_val.copy()
    df_val["error_magnitude"] = all_losses

    # Extract meta features for correlation
    meta_stats = []
    for idx, row in df_val.iterrows():
        path = cfg.get_image_path(row["file_path"])
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                h, w, c = img.shape
                # Simple intensity calculation
                intensity = img.mean() / 255.0
                ar = w / h
                meta_stats.append(
                    {
                        "width": w,
                        "height": h,
                        "aspect_ratio": ar,
                        "intensity": intensity,
                        "error": row["error_magnitude"],
                    }
                )

    if meta_stats:
        df_meta = pd.DataFrame(meta_stats)
        correlations = df_meta.corr()["error"].drop("error")
        print("Correlation between Error Magnitude and Input Features:")
        print(correlations)
    else:
        print("Could not extract meta features for analysis.")


def main():
    # 1. Setup
    cfg = Config()
    seed_everything(cfg.seed)
    device = cfg.device

    print(f"Device: {device}")

    # 2. Load Metadata
    df_train = pd.read_csv(cfg.train_metadata_path)
    df_val = pd.read_csv(cfg.val_metadata_path)
    df_test = pd.read_csv(cfg.test_metadata_path)

    # 3. Phase 1: Teacher Generation
    e_opt, teacher_logits = run_phase_1_teacher_generation(cfg, df_train, device)

    # 4. Phase 2: Student Training
    student_model = run_phase_2_student_training(
        cfg, df_train, e_opt, teacher_logits, device
    )

    # 5. Validation Assessment
    print("\n==== Validation Assessment ====")
    val_ds = AppleDataset(
        df_val, cfg, transform=get_transforms("valid", cfg), mode="standard"
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # Use standard CE for validation reporting (loss_fn=None defaults to CE in validate logic if needed,
    # but here we just need the score)
    _, val_score, _, _ = validate(student_model, val_loader, device, loss_fn=None)

    # REQUIRED FORMAT
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    failure_analysis(student_model, df_val, cfg, device)

    # 7. Submission
    threshold = 0.9871488489626378
    if val_score > threshold:
        print(
            f"\nValidation score ({val_score}) exceeds threshold ({threshold}). Generating submission..."
        )
        test_ds = AppleDataset(
            df_test, cfg, transform=get_transforms("valid", cfg), mode="test"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        generate_submission(
            student_model, test_loader, device, cfg.submission_path, cfg.target_cols
        )
        print(f"Submission saved to {cfg.submission_path}")
    else:
        print(
            f"\nValidation score ({val_score}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
