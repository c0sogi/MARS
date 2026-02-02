import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef
from library import config


class KinematicMLP(nn.Module):
    """
    Kinematic Multi-Layer Perceptron (K-MLP) for contact detection.

    Architecture:
    - Input Layer: Flattened kinematic features
    - Hidden Layers: Dense -> BatchNorm -> ReLU -> Dropout
    - Output Layer: Dense -> Sigmoid
    """

    def __init__(self, input_dim):
        super(KinematicMLP, self).__init__()

        layers = []
        curr_dim = input_dim

        # Build hidden layers dynamically from config
        for hidden_dim in config.HIDDEN_LAYERS:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT_RATE))
            curr_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(train_loader, val_loader, input_dim):
    """
    Trains the K-MLP model with early stopping and weighted loss.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Integer size of feature vector

    Returns:
        best_threshold: Float, optimal decision threshold based on validation MCC
    """
    device = config.DEVICE
    model = KinematicMLP(input_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # Class imbalance handling
    pos_weight = torch.tensor(config.POS_WEIGHT).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_threshold = 0.5

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)

            # Manual weighted BCELoss
            # loss = - [w * y * log(p) + (1-y) * log(1-p)]
            # weights = POS_WEIGHT if y=1 else 1
            loss_fn = nn.BCELoss(reduction="none")
            loss_unreduced = loss_fn(outputs, targets)
            weights = targets * (pos_weight - 1) + 1
            loss = (loss_unreduced * weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                targets = batch["target"].to(device).unsqueeze(1)

                outputs = model(features)

                loss_fn = nn.BCELoss(reduction="none")
                loss_unreduced = loss_fn(outputs, targets)
                weights = targets * (pos_weight - 1) + 1
                loss = (loss_unreduced * weights).mean()

                val_loss += loss.item() * features.size(0)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # --- Metrics & Threshold Tuning ---
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Find best threshold for this epoch
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_epoch_mcc = -1.0
        best_epoch_thresh = 0.5

        for t in thresholds:
            binary_preds = (all_preds > t).astype(int)
            # Handle edge case where all preds are same class causing MCC warning/error
            if len(np.unique(binary_preds)) > 1:
                mcc = matthews_corrcoef(all_targets, binary_preds)
            else:
                mcc = 0.0

            if mcc > best_epoch_mcc:
                best_epoch_mcc = mcc
                best_epoch_thresh = t

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCC: {best_epoch_mcc:.6f} | "
            f"Thresh: {best_epoch_thresh:.2f}"
        )

        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = best_epoch_thresh
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(
        f"Training complete. Best Val Loss: {best_val_loss:.6f}, Best Threshold: {best_threshold:.2f}"
    )
    return best_threshold


def predict_and_submit(test_loader, input_dim, threshold):
    """
    Generates predictions for the test set and saves to submission file.

    Args:
        test_loader: DataLoader for test data
        input_dim: Integer size of feature vector
        threshold: Float decision threshold
    """
    device = config.DEVICE
    model = KinematicMLP(input_dim).to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print(f"Generating predictions using threshold {threshold:.2f}...")

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            contact_ids = batch["contact_id"]  # Tuple of strings

            outputs = model(features)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(contact_ids)

    all_preds = np.concatenate(all_preds)
    binary_preds = (all_preds > threshold).astype(int)

    # Create submission DataFrame
    df_sub = pd.DataFrame({"contact_id": all_ids, "contact": binary_preds.flatten()})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_FILE_PATH), exist_ok=True)

    df_sub.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE_PATH} with {len(df_sub)} rows.")
