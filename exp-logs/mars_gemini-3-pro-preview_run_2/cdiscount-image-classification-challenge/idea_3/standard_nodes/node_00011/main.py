import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.dataset import (
    BSONDataset,
    get_transforms,
    collate_flatten,
    collate_product,
)
from library.model import get_model
from library.engine import train_one_epoch, validate, inference
from library.utils import Mixup, save_checkpoint


def run_failure_analysis(val_loader, model, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlation between error magnitude and number of images per product.
    """
    model.eval()

    all_errors = []
    all_num_images = []

    print("\nRunning Failure Analysis...")

    with torch.no_grad():
        for images, pids, targets, sizes in val_loader:
            images = images.to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=Config.USE_AMP):
                outputs = model(images)

            # Split outputs by product
            split_outputs = torch.split(outputs, sizes.tolist())

            current_idx = 0

            for i, size in enumerate(sizes):
                # Average logits for this product (Late Fusion)
                prod_output = split_outputs[i].mean(dim=0)

                # Get prediction
                _, pred_idx = prod_output.topk(1, 0, True, True)
                pred_label = pred_idx.item()

                # Get target (all images in group share the same target)
                true_label = targets[current_idx].item()

                # Error: 1 if wrong, 0 if correct
                error = 1 if pred_label != true_label else 0

                all_errors.append(error)
                all_num_images.append(size.item())

                current_idx += size.item()

    # Calculate correlation
    if len(all_errors) > 1:
        corr, p_value = pearsonr(all_errors, all_num_images)
        print(
            f"Correlation between Error and Num_Images: {corr:.4f} (p-value: {p_value:.4e})"
        )
    else:
        print("Not enough samples for correlation analysis.")


def main():
    # 1. Setup
    Config.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Adjust Config for Fast Baseline
    # We limit data size and epochs to ensure execution within 2 hours
    TRAIN_DEBUG_SIZE = 200000
    VAL_DEBUG_SIZE = None
    EPOCHS = 1

    print(
        f"Configuration: Epochs={EPOCHS}, Train Size={TRAIN_DEBUG_SIZE}, Val Size={VAL_DEBUG_SIZE}"
    )

    # 2. Data Loading
    print("Initializing Datasets...")

    # Train Dataset
    train_dataset = BSONDataset(
        metadata_csv=Config.TRAIN_META,
        bson_file=Config.TRAIN_BSON,
        split="train",
        transform=get_transforms("train"),
        debug_size=TRAIN_DEBUG_SIZE,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_flatten,
        drop_last=True,
    )

    # Validation Dataset
    val_dataset = BSONDataset(
        metadata_csv=Config.VAL_META,
        bson_file=Config.TRAIN_BSON,  # Val is a split of train.bson
        split="val",
        transform=get_transforms("val"),
        debug_size=VAL_DEBUG_SIZE,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        collate_fn=collate_product,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = get_model(pretrained=True, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 4. Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=Config.LR_MAX,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    mixup_fn = Mixup(alpha=Config.MIXUP_ALPHA)

    # 5. Training Loop
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            device,
            mixup_fn,
        )

        # Validate
        val_acc = validate(val_loader, model, criterion, device)

        # Save Checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    # 6. Final Validation Metric
    # We use the accuracy from the last validation run (or best)
    print(f"Final Validation Metric: {best_acc}")

    # 7. Failure Analysis
    # Load best model for analysis
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth.tar")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for analysis...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    run_failure_analysis(val_loader, model, device)

    # 8. Conditional Submission
    THRESHOLD = 20.0
    if best_acc > THRESHOLD:
        print(
            f"\nValidation accuracy ({best_acc}) > {THRESHOLD}. Generating submission..."
        )

        # Load Test Dataset
        # We use the full test set for submission
        test_dataset = BSONDataset(
            metadata_csv=Config.TEST_META,
            bson_file=Config.TEST_BSON,
            split="test",
            transform=get_transforms("test"),
            debug_size=None,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            collate_fn=collate_product,
        )

        inference(test_loader, model, device)
    else:
        print(
            f"\nValidation accuracy ({best_acc}) <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
