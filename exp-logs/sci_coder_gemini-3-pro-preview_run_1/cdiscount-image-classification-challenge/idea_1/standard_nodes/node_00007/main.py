import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pointbiserialr

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config, seed_everything
from library.trainer import Trainer
from library.model import generate_submission, MultiViewResNet
from library.dataset import CdiscountDataset, collate_fn
from library.utils import get_transforms, load_checkpoint


def run_analysis(model, device):
    """
    Runs inference on the full validation set to compute the final metric
    and perform failure analysis.
    """
    print("\n==== Running Full Validation & Failure Analysis ====")

    # Load full validation set (no subsetting here)
    val_dataset = CdiscountDataset(
        metadata_path=Config.VAL_METADATA,
        bson_path=Config.TRAIN_BSON,
        transform=get_transforms("val"),
        mode="val",
    )

    # Use a larger batch size for inference to speed up processing
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    model.eval()
    all_preds = []
    all_targets = []
    all_num_imgs = []

    print(f"Validating on {len(val_dataset)} samples...")

    with torch.no_grad():
        for i, (images, indices, targets, _) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            # Count images per sample to analyze correlation later.
            # 'indices' maps each image to its sample index in the batch.
            # We use bincount to get the number of images for each sample.
            batch_sample_count = targets.size(0)
            if indices.numel() > 0:
                counts = torch.bincount(indices, minlength=batch_sample_count)
                all_num_imgs.extend(counts.cpu().numpy())
            else:
                # Handle edge case of empty batch (unlikely)
                all_num_imgs.extend([0] * batch_sample_count)

            # Mixed precision inference
            with torch.amp.autocast("cuda"):
                outputs = model(images, indices)

            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

            if (i + 1) % 100 == 0:
                print(f"Processed batch {i + 1}/{len(val_loader)}", end="\r")

    print()
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_num_imgs = np.array(all_num_imgs)

    # 1. Final Validation Metric
    accuracy = (all_preds == all_targets).mean()
    print(f"Final Validation Metric: {accuracy}")

    # 2. Failure Analysis
    # Error: 1 if prediction is wrong, 0 if correct
    errors = (all_preds != all_targets).astype(int)

    # Correlation between Error magnitude and Number of Images
    # We hypothesize that products with more images might be easier to classify.
    if len(np.unique(errors)) > 1:
        # Point Biserial Correlation is used for Binary (Error) vs Continuous/Ordinal (Num Images) variables
        corr, p_val = pointbiserialr(errors, all_num_imgs)
        print(
            f"Correlation between Error and Num_Images: {corr:.4f} (p-value: {p_val:.4e})"
        )

        # Display error rate broken down by image count
        df_analysis = pd.DataFrame({"num_imgs": all_num_imgs, "error": errors})
        print("Error Rate by Image Count:")
        print(df_analysis.groupby("num_imgs")["error"].mean())
    else:
        print("Skipping correlation analysis (insufficient variance in errors).")

    return accuracy


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # We use the full dataset (TRAIN_LIMIT=None).
    # Cite Lesson 00005 & 00006: Extended training with large batches to maximize convergence.
    TRAIN_LIMIT = None
    EPOCHS = 6

    print(
        f"Starting Full Training: Limit={TRAIN_LIMIT}, Epochs={EPOCHS}, Batch={Config.BATCH_SIZE}"
    )

    trainer = Trainer(Config)
    trainer.fit(num_epochs=EPOCHS, debug_limit=TRAIN_LIMIT)

    # 3. Validation & Analysis
    device = torch.device(Config.DEVICE)

    # Load the best model saved by Trainer
    model = MultiViewResNet(num_classes=Config.NUM_CLASSES, pretrained=False)
    epoch, best_acc = load_checkpoint(Config.MODEL_CHECKPOINT, model)
    model = model.to(device)
    print(
        f"Loaded best model from Epoch {epoch} (Val Acc: {best_acc:.4f}%) for full analysis."
    )

    accuracy = run_analysis(model, device)

    # 4. Submission
    # Generates submission.csv using the best checkpoint if threshold is met
    THRESHOLD = 0.6306776302037904
    if accuracy > THRESHOLD:
        print(
            f"Validation accuracy {accuracy:.6f} > {THRESHOLD}. Generating submission."
        )
        generate_submission()
    else:
        print(
            f"Validation accuracy {accuracy:.6f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
