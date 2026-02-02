import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import cv2
import gc
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import CFG
from library.utils import seed_everything, calculate_class_weights, calculate_roc_auc
from library.dataset import AppleDataset, get_transforms
from library.model import AppleResNet34, verify_initialization
from library.engine import train_one_epoch, valid_one_epoch
import library.workflow as workflow


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override CFG for this run to ensure efficiency and performance
    # 10 epochs is sufficient for ResNet34 to converge on this dataset size (Cite solution_lesson_node_00001)
    CFG.calibration_epochs = 10
    # Use 5 seeds for the production ensemble to maximize performance
    CFG.ensemble_seeds = [42, 2023, 777, 1990, 555]

    seed_everything(CFG.seed)

    # ==========================================
    # 2. Validation Phase (Proxy Calibration)
    # ==========================================
    # We train on train_metadata and validate on val_metadata to get the required metric
    # and determine the optimal epoch for production training.

    train_df = pd.read_csv(CFG.train_metadata_path)
    val_df = pd.read_csv(CFG.val_metadata_path)

    # Prepare Datasets
    train_dataset = AppleDataset(train_df, transform=get_transforms("train"))
    val_dataset = AppleDataset(val_df, transform=get_transforms("valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Calculate weights for this specific training set
    class_weights = calculate_class_weights(
        train_df, CFG.target_cols, load_cached_data=False
    )

    # Initialize Model
    model = AppleResNet34(pretrained=True).to(CFG.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.calibration_epochs, T_mult=1, eta_min=CFG.min_lr
    )

    # Verify Initialization
    verify_initialization(model, train_loader, criterion, CFG.device)

    best_auc = 0.0
    best_epoch = 0
    best_val_preds = None
    best_val_labels = None

    for epoch in range(CFG.calibration_epochs):
        _ = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, CFG.device
        )
        _, val_preds, val_labels = valid_one_epoch(
            model, val_loader, criterion, CFG.device
        )

        auc = calculate_roc_auc(val_labels, val_preds)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_val_preds = val_preds
            best_val_labels = val_labels

    # Required Output: Print the final validation metric
    print(f"Final Validation Metric: {best_auc}")

    # ==========================================
    # 3. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis on Validation Set...")

    # Calculate error magnitude (1 - probability of true class)
    # val_labels are one-hot, val_preds are probabilities
    true_class_indices = np.argmax(best_val_labels, axis=1)
    # extract prob of true class using advanced indexing
    true_class_probs = best_val_preds[
        np.arange(len(best_val_labels)), true_class_indices
    ]
    error_magnitudes = 1.0 - true_class_probs

    # Extract Image Features
    # We iterate val_df. The order matches val_loader (shuffle=False).
    widths = []
    heights = []
    intensities = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(CFG.input_dir, row["file_path"])
        try:
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                mean_intensity = img_rgb.mean()

                widths.append(w)
                heights.append(h)
                intensities.append(mean_intensity)
            else:
                widths.append(0)
                heights.append(0)
                intensities.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    analysis_df = pd.DataFrame(
        {
            "error_magnitude": error_magnitudes,
            "width": widths,
            "height": heights,
            "intensity": intensities,
        }
    )

    # Calculate correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # ==========================================
    # 4. Production & Submission
    # ==========================================
    THRESHOLD = 0.9871488489626378

    if best_auc > THRESHOLD:

        # Free memory before production training
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # Run Production Phase (Full Data, Seed Ensemble)
        # We use the best_epoch determined from the validation run
        workflow.run_production_phase(best_epoch, load_cached_data=False)

        # Generate Submission
        workflow.generate_submission()

    else:
        print(
            f"Validation metric ({best_auc}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
