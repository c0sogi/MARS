import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(".")

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    OUTPUT_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_LR,
    PATIENCE,
)
from library.utils import seed_everything, EarlyStopping, get_roc_auc_score
from library.dataset import get_dataloaders
from library.model import get_multichannel_resnet
from library.engine import train_one_epoch, validate, generate_submission


def run():
    # 1. Setup Environment
    seed_everything(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # We use debug=False to load the full dataset.
    # The A100 GPU is sufficient to train 10 epochs on ~43k samples within the time limit.
    # Using debug=True would truncate the test set, preventing a valid submission.
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        TRAIN_METADATA_PATH,
        VAL_METADATA_PATH,
        TEST_METADATA_PATH,
        batch_size=BATCH_SIZE,
        debug=False,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = get_multichannel_resnet(pretrained=True)
    model = model.to(DEVICE)

    # 4. Optimizer, Scheduler, Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=PATIENCE, verbose=True, path="best_model.pth"
    )

    # 5. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scheduler
        )
        print(f"  Train Loss: {avg_train_loss:.6f}")

        # Validate
        avg_val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        # Track best score
        if val_auc > best_val_auc:
            best_val_auc = val_auc

        # Check Early Stopping
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # 6. Load Best Model for Analysis and Inference
    print("\nLoading best model checkpoint...")
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 7. Final Validation and Failure Analysis
    print("Performing Failure Analysis on Validation Set...")

    val_targets = []
    val_preds = []
    val_errors = []

    # Feature accumulators for correlation analysis
    feat_means = []
    feat_stds = []
    feat_maxs = []
    feat_contrasts = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # Collect predictions and targets
            batch_targets = targets.cpu().numpy()
            batch_probs = probs.squeeze(1).cpu().numpy()
            batch_errors = np.abs(batch_targets - batch_probs)

            val_targets.extend(batch_targets)
            val_preds.extend(batch_probs)
            val_errors.extend(batch_errors)

            # Calculate input features for failure analysis
            # inputs shape: (B, 6, H, W)
            # Note: Inputs are already normalized (mean~0, std~1) by the dataset class
            imgs = inputs.cpu().numpy()

            for i in range(imgs.shape[0]):
                img = imgs[i]

                # Global statistics
                feat_means.append(np.mean(img))
                feat_stds.append(np.std(img))
                feat_maxs.append(np.max(img))

                # Contrast: Mean(On-Target) - Mean(Off-Target)
                # The dataset stacks panels vertically: (1, 1638, 256). We reshape to recover (6, 273, 256).
                # Cite debug_lesson_1: Enforce Dimensional Consistency Between Dataset Output and Model Input/Analysis
                panels = img.squeeze(0).reshape(6, 273, 256)

                # On: Panels 0, 2, 4 | Off: Panels 1, 3, 5
                mean_on = np.mean(panels[[0, 2, 4], :, :])
                mean_off = np.mean(panels[[1, 3, 5], :, :])
                feat_contrasts.append(mean_on - mean_off)

    # Calculate Final Metric
    final_auc = get_roc_auc_score(np.array(val_targets), np.array(val_preds))
    print(f"Final Validation Metric: {final_auc}")

    # Calculate Correlations
    errors = np.array(val_errors)
    features = {
        "Mean Intensity": np.array(feat_means),
        "Std Intensity": np.array(feat_stds),
        "Max Intensity": np.array(feat_maxs),
        "Contrast (On-Off)": np.array(feat_contrasts),
    }

    print("\nFailure Analysis - Correlation of Error with Input Features:")
    for name, vals in features.items():
        # Avoid correlation calculation if variance is zero (e.g. due to normalization)
        if np.std(vals) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(errors, vals)
        print(f"  {name}: {corr:.4f}")

    # 8. Submission
    THRESHOLD = 0.5059716945491699

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(model, test_loader, DEVICE, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_auc}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
