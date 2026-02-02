import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import log_loss

# Import provided library modules
from library import config, utils, model, data_loader

# ==========================================
# CONFIGURATION OVERRIDES
# ==========================================
# Adjust hyperparameters for a fast baseline execution
config.NUM_EPOCHS = 20
config.PATIENCE = 5

# ==========================================
# HELPER FUNCTIONS
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return running_loss / total_samples


# ==========================================
# MAIN EXECUTION
# ==========================================


def run():
    # 1. Initialization
    utils.seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We will accumulate OOF predictions and targets
    oof_preds = []
    oof_targets = []
    oof_angles = []

    # 3. Training Loop (Stratified K-Fold Cross-Validation)
    # We use dynamic folds on the full dataset (Cite Lesson 00052)

    # Pre-fetch test loader (it's constant)
    _, _, test_loader = data_loader.get_loaders(load_cached_data=True, fold_idx=0)
    test_ensemble_preds = []

    for fold_idx in range(config.N_FOLDS):
        print(f"Starting Fold {fold_idx}...")
        # Seed for this fold
        utils.seed_everything(config.SEED + fold_idx)

        # Get Fold Loaders
        train_loader, val_loader, _ = data_loader.get_loaders(
            load_cached_data=True, fold_idx=fold_idx
        )

        # Initialize Model (A2SHN)
        net = model.A2SHN().to(device)

        # Setup Training Components
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=config.PATIENCE // 2,
            min_lr=config.MIN_LR,
        )
        early_stopping = utils.EarlyStopping(patience=config.PATIENCE, verbose=False)

        # Epoch Loop
        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_one_epoch(
                net, train_loader, criterion, optimizer, device
            )
            val_loss = validate(net, val_loader, criterion, device)

            scheduler.step(val_loss)
            early_stopping(val_loss, net)

            if early_stopping.early_stop:
                break

        # Save Best Model
        save_path = config.get_model_path(fold_idx)
        torch.save(early_stopping.best_model_state, save_path)

        # Load Best Weights for Inference
        net.load_state_dict(early_stopping.best_model_state)
        net.eval()

        # Generate OOF Predictions for this fold
        fold_oof_preds = []
        fold_targets = []
        fold_angles = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = net(images, angles)
                probs = torch.sigmoid(outputs)

                fold_oof_preds.append(probs.cpu().numpy())
                fold_targets.append(labels.numpy())
                fold_angles.append(angles.cpu().numpy())

        oof_preds.append(np.vstack(fold_oof_preds))
        oof_targets.append(np.concatenate(fold_targets))
        oof_angles.append(np.concatenate(fold_angles))

        # Generate Test Predictions for this fold
        fold_test_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = net(images, angles)
                probs = torch.sigmoid(outputs)
                fold_test_preds.append(probs.cpu().numpy())

        test_ensemble_preds.append(np.vstack(fold_test_preds))

    # 4. Metrics (OOF)
    # Concatenate all OOF predictions
    all_oof_preds = np.concatenate(oof_preds).flatten()
    all_oof_targets = np.concatenate(oof_targets)
    all_oof_angles = np.concatenate(oof_angles)

    # Calculate Metric
    final_metric = log_loss(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    # Calculate absolute error
    errors = np.abs(all_oof_targets - all_oof_preds)

    # Calculate correlation between Error and Incidence Angle
    if len(np.unique(all_oof_angles)) > 1:
        corr_matrix = np.corrcoef(all_oof_angles, errors)
        corr = corr_matrix[0, 1]
    else:
        corr = 0.0

    print(f"Failure Analysis - Correlation (Error vs Inc_Angle): {corr}")

    # 6. Submission Generation
    THRESHOLD = 0.17493283735739185

    if final_metric < THRESHOLD:
        # Load Test IDs
        df_test = pd.read_csv(config.TEST_META_CSV)
        test_ids = df_test["id"].values

        # Average Test Predictions
        avg_test_preds = np.mean(test_ensemble_preds, axis=0).flatten()

        # Save Submission
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
        df_sub.to_csv(config.SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    run()
