import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything, init_logger, get_score
from library.dataset import CassavaDataset, get_transforms, CollateMixupCutmix
from library.model import CassavaModel
from library.engine import train_one_epoch, validate
from library.inference import predict_tta


def worker_init_fn(worker_id):
    """
    Sets random seed for each worker to ensure reproducibility.
    """
    np.random.seed(CFG.seed + worker_id)
    random.seed(CFG.seed + worker_id)


def analyze_failures(model, val_loader, device, val_df):
    """
    Performs failure analysis on the validation set.
    Calculates accuracy and correlations between error and features.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Run inference on validation set
    # We disable gradients for efficiency
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get hard predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate binary error (1 for incorrect, 0 for correct)
    errors = (all_preds != all_targets).astype(int)

    print("\n--- Failure Analysis ---")
    print(f"Total Samples: {len(errors)}")
    print(f"Total Errors: {errors.sum()}")

    # Correlation with True Label
    if len(np.unique(all_targets)) > 1:
        corr_label = np.corrcoef(errors, all_targets)[0, 1]
        print(f"Correlation between Error and True Label: {corr_label:.4f}")

    # Correlation with File Size (proxy for image complexity/quality)
    # Extract file sizes from the filesystem based on metadata
    file_sizes = []
    for _, row in val_df.iterrows():
        full_path = os.path.join(CFG.input_root, row["file_path"])
        if os.path.exists(full_path):
            file_sizes.append(os.path.getsize(full_path))
        else:
            file_sizes.append(0)

    if len(file_sizes) == len(errors) and np.std(file_sizes) > 0:
        corr_size = np.corrcoef(errors, file_sizes)[0, 1]
        print(f"Correlation between Error and File Size: {corr_size:.4f}")

    # Calculate final accuracy score
    return get_score(all_targets, all_preds)


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    logger = init_logger()

    # Adjust configuration for fast baseline execution
    # We limit epochs to ensure completion within 2 hours while allowing convergence
    CFG.epochs = 10

    logger.info(f"Starting execution with Device: {CFG.device}")
    logger.info(
        f"Training for {CFG.epochs} epochs (1 Warmup + {CFG.epochs - CFG.freeze_epochs} Fine-tune)"
    )

    # 2. Data Loading
    logger.info("Loading Metadata...")
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)

    # Create Datasets
    train_dataset = CassavaDataset(train_df, transform=get_transforms("train"))
    val_dataset = CassavaDataset(val_df, transform=get_transforms("valid"))

    # Create DataLoaders
    # Train loader uses CutMix/MixUp collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=CollateMixupCutmix(
            mix_p=CFG.mix_p, alpha=CFG.mix_alpha, n_classes=CFG.target_size
        ),
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # 3. Model Initialization
    logger.info(f"Initializing Model: {CFG.model_name}")
    model = CassavaModel(model_name=CFG.model_name, pretrained=True)
    model.to(CFG.device)

    # 4. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(CFG.output_dir, CFG.model_save_name)

    # --- Stage 1: Freeze Backbone (Warmup) ---
    logger.info("Stage 1: Frozen Backbone Training")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Optimizer for head only
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    for epoch in range(1, CFG.freeze_epochs + 1):
        train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            CFG.device,
            scheduler=None,
            logger=logger,
        )
        validate(model, val_loader, CFG.device, logger)

    # --- Stage 2: Unfreeze Backbone (Fine-tuning) ---
    logger.info("Stage 2: Full Model Training")
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Re-initialize optimizer for full model
    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=CFG.epochs - CFG.freeze_epochs, eta_min=CFG.min_lr
    )

    for epoch in range(CFG.freeze_epochs + 1, CFG.epochs + 1):

        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, CFG.device, scheduler, logger
        )
        val_metrics = validate(model, val_loader, CFG.device, logger)

        val_acc = val_metrics["accuracy"]

        if scheduler:
            scheduler.step()

        # Save Best Model
        if val_acc > best_acc:
            logger.info(f"Accuracy Improved: {best_acc:.6f} -> {val_acc:.6f}")
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Model saved to {best_model_path}")

    # 5. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))
    else:
        logger.warning("Best model file not found, using current model weights.")

    # Perform failure analysis and get exact final metric
    final_metric = analyze_failures(model, val_loader, CFG.device, val_df)

    # Print required metric with full precision
    print(f"Final Validation Metric: {final_metric}")

    # 6. Submission Generation
    # Threshold condition
    THRESHOLD = 0.8571428571428571

    if final_metric > THRESHOLD:
        logger.info("Validation metric meets threshold. Generating submission...")

        # Load Test Data
        test_df = pd.read_csv(CFG.test_csv)

        # Test Dataset (output_label=False for inference)
        test_dataset = CassavaDataset(
            test_df, transform=get_transforms("valid"), output_label=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Run Inference with TTA
        predictions = predict_tta(model, test_loader, CFG.device)

        # Create Submission DataFrame
        test_df["label"] = predictions
        submission_df = test_df[["image_id", "label"]]

        # Save
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")

    else:
        logger.info(
            f"Validation metric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
