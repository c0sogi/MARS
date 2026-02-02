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
    save_checkpoint,
    load_checkpoint,
)
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.engine import train_one_epoch, validate, inference, generate_submission


def run_training(cfg, df_train, df_val, device):
    """
    Standard training loop with Validation Checkpointing.
    Cite solution_lesson_node_00031: Checkpointing based on validation metric.
    Cite solution_lesson_node_00001: Inverse Class Frequency Weighting.
    Cite solution_lesson_node_00003: ResNet34 + Strong Augmentation.
    """
    print(f"==== Training (ResNet34, {cfg.epochs} epochs) ====")

    # Datasets
    train_ds = AppleDataset(
        df_train, cfg, transform=get_transforms("train", cfg), mode="standard"
    )
    val_ds = AppleDataset(
        df_val, cfg, transform=get_transforms("valid", cfg), mode="standard"
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
    # Cite solution_lesson_node_00015: Synchronize T_max with epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
    )

    # Loss: Weighted Cross Entropy
    class_weights = get_class_weights(
        df_train, cfg.target_cols, cache_dir=cfg.working_dir
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_score = -float("inf")
    best_model_path = os.path.join(cfg.models_dir, "best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scheduler
        )

        val_loss, val_score, _, _ = validate(model, val_loader, device, loss_fn)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val AUC: {val_score:.4f}"
        )

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            save_checkpoint(model, optimizer, epoch, val_score, best_model_path)
            print(f"  -> New Best Score! Saved to {best_model_path}")

    # Load Best Model
    print(f"Loading best model with score {best_score:.4f}...")
    model, _, _, _ = load_checkpoint(model, best_model_path, device=device)

    return model, best_score


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

    # 3. Training
    student_model, val_score = run_training(cfg, df_train, df_val, device)

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
