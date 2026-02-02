import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Import library modules
import importlib
import library.config
import library.utils
import library.dataset
import library.model
import library.engine

# Reload modules to handle persistent runtime caching (Cite debug_lesson_3)
importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.engine)

from library.config import CFG
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_loaders
from library.model import CassavaConvNeXt
from library.engine import train_one_epoch, valid_one_epoch, inference_fn


def main():
    # --- 1. Setup ---
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    # --- 2. Stage 1: Warmup (Frozen Backbone) ---
    # Loaders for base resolution (384x384)
    train_loader, val_loader, test_loader = get_loaders(
        CFG.img_size_base, load_cached_data=True
    )

    # Initialize Model
    model = CassavaConvNeXt(model_name=CFG.model_name, pretrained=True)
    model.to(device)

    # Freeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Optimizer for Head only
    optimizer = optim.AdamW(
        model.head.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Train Warmup
    for epoch in range(CFG.epochs_warmup):
        train_one_epoch(
            epoch, model, train_loader, optimizer, device, accum_iter=CFG.accum_iter
        )
        valid_one_epoch(epoch, model, val_loader, device)

    # --- 3. Stage 2: Base Training (Unfrozen) ---
    # Unfreeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Re-initialize Optimizer for full model
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )

    # Scheduler: T_max is total steps because engine steps scheduler per batch
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * CFG.epochs_base, eta_min=CFG.min_lr
    )

    best_acc = 0.0

    for epoch in range(CFG.epochs_base):
        train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
            accum_iter=CFG.accum_iter,
        )
        _, val_acc = valid_one_epoch(epoch, model, val_loader, device)

        # Save Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
            },
            is_best,
            CFG.output_dir,
            filename="checkpoint_stage2.pth",
        )

    # --- 4. Stage 3: Fine-Tuning (High Resolution) ---
    # Load Best Model from Stage 2
    best_model_path = os.path.join(CFG.output_dir, "best_model.pth")
    load_checkpoint(model, best_model_path, device)

    # Get High-Res Loaders (512x512)
    train_loader, val_loader, test_loader = get_loaders(
        CFG.img_size_finetune, load_cached_data=True
    )

    # Optimizer with lower LR
    optimizer = optim.AdamW(
        model.parameters(), lr=CFG.learning_rate / 10, weight_decay=CFG.weight_decay
    )

    # Scheduler for fine-tuning
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * CFG.epochs_finetune, eta_min=CFG.min_lr
    )

    for epoch in range(CFG.epochs_finetune):
        train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            device,
            scheduler=scheduler,
            accum_iter=CFG.accum_iter,
        )
        _, val_acc = valid_one_epoch(epoch, model, val_loader, device)

        # Save Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
            },
            is_best,
            CFG.output_dir,
            filename="checkpoint_stage3.pth",
        )

    # --- 5. Final Evaluation & Failure Analysis ---
    # Load Absolute Best Model
    load_checkpoint(model, best_model_path, device)

    # Compute Final Metric on Val Set (Standard inference, no TTA)
    _, final_acc = valid_one_epoch(999, model, val_loader, device)
    print(f"Final Validation Metric: {final_acc}")

    # Failure Analysis
    model.eval()

    # Collect predictions and targets
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets.numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    errors = (val_preds != val_targets).astype(int)

    # Load Metadata for Features
    val_df = val_loader.dataset.df.copy()

    # Calculate Features (File Size)
    file_sizes = []

    for idx, row in val_df.iterrows():
        fpath = os.path.join(CFG.input_root, row["file_path"])
        if os.path.exists(fpath):
            file_sizes.append(os.path.getsize(fpath))
        else:
            file_sizes.append(0)

    val_df["file_size"] = file_sizes
    val_df["error"] = errors

    # Correlation
    corr_size = val_df["error"].corr(val_df["file_size"])
    print(f"Correlation between Error and File Size: {corr_size}")

    # --- 6. Submission ---
    THRESHOLD = 0.9025367158754805
    if final_acc > THRESHOLD:
        # Inference with TTA
        test_preds = inference_fn(model, test_loader, device, tta_steps=CFG.tta_steps)

        # Save
        sub_df = pd.read_csv(CFG.test_csv)
        sub_df["label"] = test_preds
        sub_df[["image_id", "label"]].to_csv(CFG.submission_path, index=False)
        print(f"Submission saved to {CFG.submission_path}")
    else:
        print(f"Metric {final_acc} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
