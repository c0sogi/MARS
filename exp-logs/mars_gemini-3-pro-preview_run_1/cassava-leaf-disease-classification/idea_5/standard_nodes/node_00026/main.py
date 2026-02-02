import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    average_weights,
    update_bn,
)
from library.dataset import get_dataloaders, Mixup
from library.model import CassavaModel
from library.engine import train_one_epoch, validate, generate_submission


def run():
    # 1. Setup and Configuration
    # We use debug=False to run the full pipeline, but we will adjust epochs manually
    # to ensure it fits strictly within the "fast baseline" time constraints while
    # still performing the multi-stage training.
    config = Config(debug=False)

    # Optimize schedule for execution speed on A100
    config.epochs_stage1 = 6
    config.epochs_stage2 = 4
    config.epochs_swa = 3

    seed_everything(config.seed)
    device = config.device

    print(f"Running on device: {device}")
    print(
        f"Config: Stage1={config.epochs_stage1}ep, Stage2={config.epochs_stage2}ep, SWA={config.epochs_swa}ep"
    )

    # ==========================================
    # Stage 1: Training at 384x384
    # ==========================================
    print("\n=== Stage 1: Training at 384x384 ===")
    train_loader, val_loader, _ = get_dataloaders(config, stage=1)

    model = CassavaModel(config, pretrained=True)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.lr_stage1, weight_decay=config.weight_decay
    )

    # Scheduler: Linear Warmup -> Cosine Annealing
    if config.warmup_epochs_stage1 > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_epochs_stage1,
        )
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs_stage1 - config.warmup_epochs_stage1,
            eta_min=config.min_lr_stage1,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, main_sched],
            milestones=[config.warmup_epochs_stage1],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.epochs_stage1, eta_min=config.min_lr_stage1
        )

    mixup_fn = Mixup(config)

    best_acc = 0.0
    best_model_path_stage1 = os.path.join(config.checkpoint_dir, "stage1_best.pth")

    for epoch in range(config.epochs_stage1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config, mixup_fn, scheduler=None
        )
        scheduler.step()

        acc, val_loss = validate(model, val_loader, device, config)
        print(
            f"Stage 1 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": acc,
                    "epoch": epoch,
                },
                best_model_path_stage1,
            )

    # Cleanup to save memory
    del train_loader, val_loader, optimizer, scheduler
    torch.cuda.empty_cache()

    # ==========================================
    # Stage 2: Fine-tuning at 512x512
    # ==========================================
    print("\n=== Stage 2: Fine-tuning at 512x512 ===")
    train_loader, val_loader, test_loader = get_dataloaders(config, stage=2)

    # Load best weights from Stage 1
    checkpoint = torch.load(best_model_path_stage1, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = AdamW(
        model.parameters(), lr=config.lr_stage2, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs_stage2, eta_min=config.min_lr_stage2
    )

    best_acc_stage2 = 0.0
    best_model_path_stage2 = os.path.join(config.checkpoint_dir, "stage2_best.pth")

    for epoch in range(config.epochs_stage2):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config, mixup_fn, scheduler=None
        )
        scheduler.step()

        acc, val_loss = validate(model, val_loader, device, config)
        print(
            f"Stage 2 Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={acc:.4f}"
        )

        if acc > best_acc_stage2:
            best_acc_stage2 = acc
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "accuracy": acc,
                    "epoch": epoch,
                },
                best_model_path_stage2,
            )

    # ==========================================
    # Stage 3: SWA Training
    # ==========================================
    print("\n=== Stage 3: SWA Training ===")
    # Load best model from Stage 2
    if os.path.exists(best_model_path_stage2):
        checkpoint = torch.load(best_model_path_stage2, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Use constant learning rate for SWA
    optimizer = AdamW(
        model.parameters(), lr=config.swa_lr, weight_decay=config.weight_decay
    )

    swa_checkpoints = []

    for epoch in range(config.epochs_swa):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config, mixup_fn, scheduler=None
        )

        # Save snapshot at end of every epoch
        ckpt_path = os.path.join(config.checkpoint_dir, f"swa_snapshot_{epoch}.pth")
        save_checkpoint({"model_state_dict": model.state_dict()}, ckpt_path)
        swa_checkpoints.append(ckpt_path)

        acc, val_loss = validate(model, val_loader, device, config)
        print(
            f"SWA Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={acc:.4f} (Snapshot saved)"
        )

    # ==========================================
    # SWA Averaging & Finalization
    # ==========================================
    print("\n=== Performing SWA Averaging ===")
    avg_state_dict = average_weights(swa_checkpoints)
    model.load_state_dict(avg_state_dict)

    print("Updating Batch Norm statistics...")
    update_bn(train_loader, model, device)

    final_model_path = os.path.join(config.output_dir, "final_swa_model.pth")
    save_checkpoint({"model_state_dict": model.state_dict()}, final_model_path)

    # ==========================================
    # Final Validation & Failure Analysis
    # ==========================================
    print("\n=== Final Validation ===")
    final_acc, final_loss = validate(model, val_loader, device, config)
    print(f"Final Validation Metric: {final_acc}")

    print("\n=== Failure Analysis ===")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    all_losses = []

    # Compute per-sample losses
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            losses = criterion(outputs, targets)
            all_losses.extend(losses.cpu().numpy())

    # Load metadata to correlate
    val_df = pd.read_csv(config.val_metadata_path)
    # Ensure alignment
    if len(all_losses) > len(val_df):
        all_losses = all_losses[: len(val_df)]
    elif len(all_losses) < len(val_df):
        val_df = val_df.iloc[: len(all_losses)]

    val_df["error_magnitude"] = all_losses

    # Get file sizes for correlation analysis
    print("Calculating file size correlations...")
    file_sizes = []
    for rel_path in val_df["file_path"]:
        full_path = os.path.join(config.input_root, rel_path)
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    val_df["file_size"] = file_sizes

    # Calculate correlation
    corr = val_df["error_magnitude"].corr(val_df["file_size"])
    print(f"Correlation between Error Magnitude and File Size: {corr:.6f}")

    # ==========================================
    # Submission
    # ==========================================
    threshold = 0.9025367158754805
    if final_acc > threshold:
        print(f"\nMetric {final_acc} > {threshold}. Generating submission...")
        generate_submission(model, test_loader, device, config)
    else:
        print(f"\nMetric {final_acc} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    run()
