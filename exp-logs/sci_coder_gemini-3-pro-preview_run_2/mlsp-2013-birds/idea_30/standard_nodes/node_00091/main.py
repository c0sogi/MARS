import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_pos_weights, compute_auc
from library.dataset import get_processed_data, BirdDataset
from library.models import get_model
from library.losses import WeightedBCE, DistillationLoss
from library.engine import (
    train_one_epoch,
    distill_one_epoch,
    valid_one_epoch,
    tta_inference,
)


def main():
    # 1. Setup
    # Override Config for fast baseline execution
    Config.EPOCHS = 8  # Reduced epochs for speed
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Data
    # We use the provided train/val split as a single fold
    train_images, train_labels, train_ids = get_processed_data("train")
    val_images, val_labels, val_ids = get_processed_data("val")
    test_images, _, test_ids = get_processed_data("test")

    print(f"Train shape: {train_images.shape}")
    print(f"Val shape: {val_images.shape}")

    # Calculate positive weights for loss
    # We need a DataFrame to use the utility function, reconstruct one temporarily
    train_df_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
    train_df = pd.DataFrame(train_labels, columns=train_df_cols)
    pos_weights = calculate_pos_weights(train_df, device)

    # =========================================================================
    # Phase 1: Anchor Training
    # =========================================================================
    print("\n=== Phase 1: Anchor Training ===")
    anchor_models = []

    for backbone in Config.ANCHOR_BACKBONES:
        print(f"Training Anchor: {backbone}")
        model = get_model(backbone, device)

        criterion = WeightedBCE(pos_weights)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=Config.EPOCHS // 2, gamma=0.1
        )

        train_dataset = BirdDataset(
            train_images, train_labels, rec_ids=train_ids, mode="train"
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        for epoch in range(Config.EPOCHS):
            loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )

        anchor_models.append(model)

    # =========================================================================
    # Phase 2: TTA-Target Generation
    # =========================================================================
    print("\n=== Phase 2: Generating Soft Targets ===")
    # Generate soft targets for the TRAINING set using the trained anchors
    # We use TTA inference to get robust targets
    soft_targets_accum = np.zeros_like(train_labels)

    for model in anchor_models:
        preds = tta_inference(
            model, train_images, train_ids, device, tta_steps=Config.TTA_STEPS
        )
        soft_targets_accum += preds

    # Average soft targets
    soft_targets = soft_targets_accum / len(anchor_models)

    # Clean up anchors to save memory
    del anchor_models
    torch.cuda.empty_cache()

    # =========================================================================
    # Phase 3: Born-Again Ensemble Training
    # =========================================================================
    print("\n=== Phase 3: Born-Again Ensemble Training ===")
    final_models = []

    for backbone in Config.BACKBONES:
        print(f"Training Student: {backbone}")
        model = get_model(backbone, device)

        # Distillation Loss
        criterion = DistillationLoss(
            pos_weights, lambda_distill=Config.DISTILLATION_LAMBDA
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=Config.EPOCHS // 2, gamma=0.1
        )

        # Dataset with soft targets
        train_dataset = BirdDataset(
            train_images,
            train_labels,
            soft_labels=soft_targets,
            rec_ids=train_ids,
            mode="train",
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        for epoch in range(Config.EPOCHS):
            loss = distill_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )

        final_models.append(model)

    # =========================================================================
    # Validation
    # =========================================================================
    print("\n=== Validation ===")
    val_preds_accum = np.zeros_like(val_labels)

    for model in final_models:
        preds = tta_inference(
            model, val_images, val_ids, device, tta_steps=Config.TTA_STEPS
        )
        val_preds_accum += preds

    val_preds_avg = val_preds_accum / len(final_models)

    # Compute Metric
    val_auc = compute_auc(val_labels, val_preds_avg)
    print(f"Final Validation Metric: {val_auc}")

    # =========================================================================
    # Failure Analysis
    # =========================================================================
    print("\n=== Failure Analysis ===")
    # Calculate error per sample (Mean Absolute Error)
    errors = np.abs(val_labels - val_preds_avg).mean(axis=1)

    # Calculate label cardinality (number of active species)
    cardinality = val_labels.sum(axis=1)

    # Calculate correlation
    if len(np.unique(cardinality)) > 1:
        corr, _ = pearsonr(errors, cardinality)
        print(f"Correlation between Error and Label Cardinality: {corr:.4f}")
    else:
        print("Correlation undefined (constant cardinality).")

    # =========================================================================
    # Submission
    # =========================================================================
    THRESHOLD = 0.92133638985917

    if val_auc > THRESHOLD:
        print("\n=== Generating Submission ===")
        test_preds_accum = np.zeros((len(test_images), Config.NUM_CLASSES))

        for model in final_models:
            preds = tta_inference(
                model, test_images, test_ids, device, tta_steps=Config.TTA_STEPS
            )
            test_preds_accum += preds

        test_preds_avg = test_preds_accum / len(final_models)

        # Format submission
        submission_rows = []
        for i, rec_id in enumerate(test_ids):
            for species_idx in range(Config.NUM_CLASSES):
                row_id = int(rec_id * 100 + species_idx)
                prob = test_preds_avg[i, species_idx]
                submission_rows.append([row_id, prob])

        sub_df = pd.DataFrame(submission_rows, columns=["Id", "Probability"])
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation metric {val_auc} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
