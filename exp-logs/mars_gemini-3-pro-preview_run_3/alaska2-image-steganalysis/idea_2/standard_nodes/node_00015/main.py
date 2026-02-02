import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import MonoResidualEfficientNet, BCEWithLogitsLossWithSmoothing
from library.engine import train_one_epoch, predict_tta
from library.metrics import alaska_weighted_auc
from library.utils import seed_everything


def main():
    # 1. Configuration Setup
    # Update configuration for full training run (Cite solution_lesson_node_00012)
    Config.update(BATCH_SIZE=48, EPOCHS=5, NUM_WORKERS=4)

    # Initialize directories and seeds
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Step A: Get full loaders (debug=False)
    train_loader, _ = get_dataloaders(debug=False)

    # Step B: Get full validation loader for correct metric calculation (full hold-out set)
    # We ignore the returned train loader here as we only need the full validation set
    _, val_loader_full = get_dataloaders(debug=False)

    # 3. Model Initialization
    model = MonoResidualEfficientNet(pretrained=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )
    criterion = BCEWithLogitsLossWithSmoothing(label_smoothing=Config.LABEL_SMOOTHING)

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        # Train on the subsampled dataset
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Save the model checkpoint
        torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # 5. Validation and Failure Analysis
    # We iterate over the full validation set to compute the metric and gather stats for failure analysis
    model.eval()

    all_preds = []
    all_labels = []
    all_means = []
    all_stds = []

    with torch.no_grad():
        for images, labels in val_loader_full:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).view(-1)

            # Store predictions and labels
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Compute image statistics (Mean and Std) for failure analysis
            # images shape: (B, 1, H, W) -> Flatten spatial dims -> (B, H*W)
            flat_imgs = images.view(images.size(0), -1)
            batch_means = flat_imgs.mean(dim=1).cpu().numpy()
            batch_stds = flat_imgs.std(dim=1).cpu().numpy()

            all_means.append(batch_means)
            all_stds.append(batch_stds)

    # Concatenate all batches
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    img_means = np.concatenate(all_means)
    img_stds = np.concatenate(all_stds)

    # Calculate Final Validation Metric (Weighted AUC)
    final_metric = alaska_weighted_auc(y_true, y_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Calculate absolute error magnitude
    errors = np.abs(y_true - y_pred)

    # Calculate correlations between error and image statistics
    corr_mean, _ = pearsonr(errors, img_means)
    corr_std, _ = pearsonr(errors, img_stds)

    print(f"Correlation (Error vs Pixel Mean): {corr_mean}")
    print(f"Correlation (Error vs Pixel Std): {corr_std}")

    # 6. Submission Generation
    THRESHOLD = 0.8333036808

    if final_metric > THRESHOLD:
        test_loader = get_test_dataloader()
        if test_loader is not None:
            # Generate predictions using Test Time Augmentation
            ids, preds = predict_tta(model, test_loader, device)

            submission_df = pd.DataFrame({"Id": ids, "Label": preds})

            os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
            submission_df.to_csv(Config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    main()
