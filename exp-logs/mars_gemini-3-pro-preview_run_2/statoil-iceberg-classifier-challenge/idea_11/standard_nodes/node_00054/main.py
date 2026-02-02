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

    # 2. Data Preparation
    # We load raw data and stats to perform custom Stratified K-Fold on train.csv
    # This implements Lesson ID: solution_lesson_node_00052 (Prioritize K-Fold Ensembling)
    raw_data, stats, id_to_idx = data_loader.get_raw_data_and_stats()

    # Load Train Metadata for K-Fold
    df_train_meta = pd.read_csv(config.TRAIN_META_CSV)
    X_train_ids = df_train_meta["id"].values
    y_train = df_train_meta["is_iceberg"].values

    # Load Hold-Out Validation Metadata (for final evaluation)
    # We strictly respect the hold-out set for metric comparison (Cite solution_lesson_node_00036)
    df_val_meta = pd.read_csv(config.VAL_META_CSV)
    val_ids = df_val_meta["id"].values

    # Create Hold-Out Validation Loader
    val_loader = data_loader.create_loader_from_ids(
        val_ids,
        raw_data,
        stats,
        id_to_idx,
        config.BATCH_SIZE,
        shuffle=False,
        transform=False,
    )

    # 3. Training Loop: Stratified K-Fold on TRAIN set
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    for fold_idx, (train_idx, inner_val_idx) in enumerate(
        skf.split(X_train_ids, y_train)
    ):
        print(f"\n{'='*20}")
        print(f"Starting Training: Fold {fold_idx}")
        print(f"{'='*20}")

        # Seed for reproducibility
        utils.seed_everything(config.SEED + fold_idx)

        # Create Fold Loaders
        fold_train_ids = X_train_ids[train_idx]
        fold_val_ids = X_train_ids[inner_val_idx]

        fold_train_loader = data_loader.create_loader_from_ids(
            fold_train_ids,
            raw_data,
            stats,
            id_to_idx,
            config.BATCH_SIZE,
            shuffle=True,
            transform=True,
        )
        fold_val_loader = data_loader.create_loader_from_ids(
            fold_val_ids,
            raw_data,
            stats,
            id_to_idx,
            config.BATCH_SIZE,
            shuffle=False,
            transform=False,
        )

        # Initialize Model
        net = model.WEBN().to(device)

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
                net, fold_train_loader, criterion, optimizer, device
            )
            # Validate on the INNER validation set for early stopping
            val_loss = validate(net, fold_val_loader, criterion, device)

            scheduler.step(val_loss)
            early_stopping(val_loss, net)

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save Best Model
        save_path = config.get_model_path(fold_idx)
        torch.save(early_stopping.best_model_state, save_path)
        print(
            f"Best model for Fold {fold_idx} saved (Val Loss: {early_stopping.best_score:.4f})"
        )

    # 4. Ensemble Validation on Hold-Out Set
    ensemble_preds = []
    val_targets = []
    val_angles = []

    # Extract targets and angles from hold-out loader
    for _, angles, labels in val_loader:
        val_targets.append(labels.numpy())
        val_angles.append(angles.numpy())
    val_targets = np.concatenate(val_targets)
    val_angles = np.concatenate(val_angles)

    # Generate predictions from each fold model on the SAME hold-out set
    for fold_idx in range(config.N_FOLDS):
        net = model.WEBN().to(device)
        state_dict = torch.load(config.get_model_path(fold_idx), map_location=device)
        net.load_state_dict(state_dict)
        net.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles, _ in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = net(images, angles)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        ensemble_preds.append(np.vstack(fold_preds))

    # Average predictions (Ensemble)
    avg_preds = np.mean(ensemble_preds, axis=0).flatten()

    # Calculate Metric
    final_metric = log_loss(val_targets, avg_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    errors = np.abs(val_targets - avg_preds)
    if len(np.unique(val_angles)) > 1:
        corr_matrix = np.corrcoef(val_angles, errors)
        corr = corr_matrix[0, 1]
    else:
        corr = 0.0

    print(f"Failure Analysis - Correlation (Error vs Inc_Angle): {corr}")

    # 6. Submission Generation
    THRESHOLD = 0.17493283735739185

    if final_metric < THRESHOLD:
        print("Metric improved. Generating submission...")
        # Load Test IDs and create loader
        df_test = pd.read_csv(config.TEST_META_CSV)
        test_ids = df_test["id"].values

        test_loader = data_loader.create_loader_from_ids(
            test_ids,
            raw_data,
            stats,
            id_to_idx,
            config.BATCH_SIZE,
            shuffle=False,
            transform=False,
        )

        test_ensemble_preds = []

        # Predict with each fold
        for fold_idx in range(config.N_FOLDS):
            net = model.WEBN().to(device)
            state_dict = torch.load(
                config.get_model_path(fold_idx), map_location=device
            )
            net.load_state_dict(state_dict)
            net.eval()

            fold_test_preds = []
            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)
                    outputs = net(images, angles)
                    probs = torch.sigmoid(outputs)
                    fold_test_preds.append(probs.cpu().numpy())

            test_ensemble_preds.append(np.vstack(fold_test_preds))

        # Average Test Predictions
        avg_test_preds = np.mean(test_ensemble_preds, axis=0).flatten()

        # Save Submission
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})
        df_sub.to_csv(config.SUBMISSION_PATH, index=False)
        print("Submission saved.")


if __name__ == "__main__":
    run()
