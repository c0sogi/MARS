import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings

# Import from library
from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet
from library.train import train_one_epoch, validate, predict

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration & Setup
    # Limit epochs for fast baseline execution while ensuring convergence
    Config.EPOCHS = 10

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Using cached data as requested to speed up loading
    # Force re-processing to ensure data matches current 3-channel configuration (Cite debug_lesson_3)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # 3. Model Initialization
    model = WhaleEfficientNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_val_auc = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step()

        # Save Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(model, optimizer, epoch, val_auc, path=Config.MODEL_PATH)

    # 5. Final Validation & Failure Analysis
    # Load best model for evaluation
    load_checkpoint(model, path=Config.MODEL_PATH, device=Config.DEVICE)
    model.eval()

    all_targets = []
    all_preds = []

    # Features for failure analysis
    feat_means = []
    feat_stds = []
    feat_maxs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Extract basic features from input before passing to model for failure analysis
            # inputs shape: (B, 3, F, T)
            flat_inputs = inputs.view(inputs.size(0), -1)

            # Compute stats on GPU then move to CPU to save time
            feat_means.append(flat_inputs.mean(dim=1).cpu().numpy())
            feat_stds.append(flat_inputs.std(dim=1).cpu().numpy())
            feat_maxs.append(flat_inputs.max(dim=1).values.cpu().numpy())

            logits = model(inputs)
            preds = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets).flatten()
    all_preds = np.concatenate(all_preds).flatten()

    feat_means = np.concatenate(feat_means)
    feat_stds = np.concatenate(feat_stds)
    feat_maxs = np.concatenate(feat_maxs)

    # Calculate Metric
    final_auc = calculate_roc_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    errors = np.abs(all_targets - all_preds)

    # Correlations
    # Check correlation between error magnitude and input signal statistics
    corr_mean = np.corrcoef(errors, feat_means)[0, 1]
    corr_std = np.corrcoef(errors, feat_stds)[0, 1]
    corr_max = np.corrcoef(errors, feat_maxs)[0, 1]

    print("Failure Analysis (Correlation with Error Magnitude):")
    print(f"Input Mean: {corr_mean:.4f}")
    print(f"Input Std:  {corr_std:.4f}")
    print(f"Input Max:  {corr_max:.4f}")

    # 6. Conditional Submission
    THRESHOLD = 0.9959177895986835
    if final_auc > THRESHOLD:
        clips, probs = predict(model, test_loader, device)

        submission_df = pd.DataFrame({"clip": clips, "probability": probs})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
