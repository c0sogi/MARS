import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library import config, utils, data, model, engine


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    engine.set_seed()
    config.setup_directories()
    device = torch.device(config.DEVICE)

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Initialize Datasets with caching enabled for ROI anchors
    train_dataset = data.MGMTDataset(
        df_train, transforms=data.get_transforms("train"), load_cached_anchors=True
    )
    val_dataset = data.MGMTDataset(
        df_val, transforms=data.get_transforms("valid"), load_cached_anchors=True
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    net = model.AsymmetricEfficientNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop (Fast Baseline)
    # -------------------------------------------------------------------------
    num_epochs = config.NUM_EPOCHS
    best_auc = 0.0

    for epoch in range(num_epochs):
        # Train for one epoch
        engine.train_one_epoch(net, train_loader, criterion, optimizer, device)

        # Validate
        _, val_auc = engine.validate(net, val_loader, criterion, device)

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(net.state_dict(), config.BEST_MODEL_PATH)

    # -------------------------------------------------------------------------
    # 5. Final Evaluation
    # -------------------------------------------------------------------------
    # Reload best model to ensure we evaluate the best state
    if os.path.exists(config.BEST_MODEL_PATH):
        net.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))

    net.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            # Forward pass
            logits = net(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    final_metric = roc_auc_score(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(np.array(all_targets) - np.array(all_preds))

    # Extract features for correlation
    # 1. Anchor Index (Anatomical position)
    anchors = val_dataset.anchors
    anchor_indices = [anchors.get(uid, 0) for uid in df_val["BraTS21ID"]]

    # 2. Target Class
    target_classes = np.array(all_targets)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "anchor_index": anchor_indices,
            "target_class": target_classes,
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["error"]
    print("Correlation between model error and input features:")
    print(correlations)

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    threshold = 0.6254545454545455

    if final_metric > threshold:
        print(f"Metric {final_metric} > {threshold}. Generating submission...")
        engine.generate_submission()
    else:
        print(f"Metric {final_metric} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
