import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
from scipy.stats import pointbiserialr

# Import provided libraries
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_loaders, get_id_map
from library.model import get_model
from library.engine import train_one_epoch, validate, predict


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("Main")

    # Runtime Configuration Overrides for Speed (Target < 2 hours)
    # A100 is fast, but the dataset is large (186k images).
    # We reduce epochs to ensure the "Fast Baseline" completes on time.
    Config.PHASE_1["epochs"] = 2
    Config.PHASE_2["epochs"] = 1

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # ==========================================
    # PHASE 1: Representation Learning (224x224)
    # ==========================================
    logger.info("=== Phase 1: Representation Learning (224x224) ===")

    train_loader, val_loader, _, mixup_fn = get_loaders(Config.PHASE_1)

    model = get_model(num_classes=Config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.PHASE_1["lr"],
        weight_decay=Config.PHASE_1["weight_decay"],
    )

    # Cosine Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.PHASE_1["epochs"]
    )

    best_metric = float("inf")

    for epoch in range(1, Config.PHASE_1["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_fn
        )
        val_metrics = validate(model, val_loader, device)

        current_error = val_metrics["Top1_Error"]
        logger.info(f"Phase 1 Epoch {epoch} Val Error: {current_error:.4f}%")

        if current_error < best_metric:
            best_metric = current_error
            torch.save(model.state_dict(), Config.PHASE_1["save_path"])

        scheduler.step()

    # Cleanup to save memory
    del train_loader, val_loader, optimizer, scheduler, mixup_fn
    torch.cuda.empty_cache()

    # ==========================================
    # PHASE 2: Fine-Grained Refinement (384x384)
    # ==========================================
    logger.info("=== Phase 2: Fine-Grained Refinement (384x384) ===")

    # Load best weights from Phase 1
    logger.info(f"Loading best Phase 1 weights from {Config.PHASE_1['save_path']}")
    model.load_state_dict(torch.load(Config.PHASE_1["save_path"], map_location=device))

    # Get DataLoaders for Phase 2 (Larger resolution, No Mixup)
    train_loader, val_loader, test_loader, mixup_fn = get_loaders(Config.PHASE_2)

    # Re-initialize Optimizer with lower LR
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.PHASE_2["lr"],
        weight_decay=Config.PHASE_2["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.PHASE_2["epochs"]
    )

    best_metric_p2 = float("inf")

    for epoch in range(1, Config.PHASE_2["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, mixup_fn
        )
        val_metrics = validate(model, val_loader, device)

        current_error = val_metrics["Top1_Error"]
        logger.info(f"Phase 2 Epoch {epoch} Val Error: {current_error:.4f}%")

        if current_error < best_metric_p2:
            best_metric_p2 = current_error
            torch.save(model.state_dict(), Config.PHASE_2["save_path"])

        scheduler.step()

    # ==========================================
    # Final Evaluation & Failure Analysis
    # ==========================================
    logger.info("=== Final Evaluation ===")

    # Load best Phase 2 model
    model.load_state_dict(torch.load(Config.PHASE_2["save_path"], map_location=device))
    model.eval()

    # Collect predictions for full validation set
    all_preds = []
    all_targets = []

    # val_loader is sequential (shuffle=False)
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Final Metric (Fraction)
    errors = (all_preds != all_targets).astype(int)
    error_fraction = np.mean(errors)

    print(f"Final Validation Metric: {error_fraction:.10f}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")

    # 1. Class Frequency Correlation
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    class_counts = train_df["category_id"].value_counts().to_dict()
    _, idx2id = get_id_map()

    # Map target indices to frequencies
    target_freqs = [class_counts.get(idx2id[t], 0) for t in all_targets]

    if len(np.unique(errors)) > 1:
        corr_freq, _ = pointbiserialr(errors, target_freqs)
        print(f"Correlation between Error and Class Frequency: {corr_freq:.4f}")
    else:
        print("Correlation between Error and Class Frequency: Undefined")

    # 2. File Size Correlation
    val_file_names = val_loader.dataset.df["file_name"].values
    file_sizes = []
    for fname in val_file_names:
        full_path = os.path.join(Config.INPUT_DIR, fname)
        try:
            file_sizes.append(os.path.getsize(full_path))
        except:
            file_sizes.append(0)

    if len(np.unique(errors)) > 1:
        corr_size, _ = pointbiserialr(errors, file_sizes)
        print(f"Correlation between Error and File Size: {corr_size:.4f}")
    else:
        print("Correlation between Error and File Size: Undefined")

    # ==========================================
    # Submission
    # ==========================================
    THRESHOLD = 0.1945

    if error_fraction < THRESHOLD:
        logger.info(
            f"Metric {error_fraction:.4f} < {THRESHOLD}. Generating submission..."
        )
        predict(model, test_loader, device, Config.SUBMISSION_FILE)
    else:
        logger.info(f"Metric {error_fraction:.4f} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    run()
