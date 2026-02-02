import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.models import AttentionalRNN
from library.datasets import FeatureSequenceDataset
from library.utils import save_predictions

# =========================================================================
# Custom Loss Function
# =========================================================================


class WeightedLogLoss(nn.Module):
    """
    PyTorch implementation of the competition's weighted log loss.
    L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]
    """

    def __init__(self, device):
        super(WeightedLogLoss, self).__init__()
        # Extract weights in the order of TARGET_COLS
        weights_list = [Config.LOSS_WEIGHTS.get(col, 1.0) for col in Config.TARGET_COLS]
        self.weights = (
            torch.tensor(weights_list, dtype=torch.float32).to(device).view(1, -1)
        )

    def forward(self, logits, targets):
        # BCEWithLogitsLoss combines Sigmoid + BCE.
        # reduction='none' gives us the loss per element (batch_size, num_classes)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Apply class weights
        weighted_loss = bce_loss * self.weights

        # Average over all elements (batch * classes)
        return weighted_loss.mean()


# =========================================================================
# Training Function
# =========================================================================


def train_aggregator(train_df, val_df, feature_dir):
    """
    Trains the Stage 3 Attentional RNN Aggregator on pre-computed features.

    Args:
        train_df: DataFrame with training metadata.
        val_df: DataFrame with validation metadata.
        feature_dir: Directory containing .npy feature files.
    """
    print("Initializing Aggregator Training...")

    # Create Checkpoint Directory
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    device = Config.DEVICE

    # Datasets & Loaders
    train_dataset = FeatureSequenceDataset(train_df, feature_dir)
    val_dataset = FeatureSequenceDataset(val_df, feature_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = AttentionalRNN(
        input_dim=Config.ENCODER_HIDDEN_DIM,
        hidden_dim=Config.SEQ_HIDDEN_DIM,
        num_layers=Config.SEQ_NUM_LAYERS,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.SEQ_LR)
    criterion = WeightedLogLoss(device=device)

    best_val_loss = float("inf")
    model_save_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    print(f"Starting training for {Config.SEQ_EPOCHS} epochs on {device}...")

    for epoch in range(Config.SEQ_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for features, masks, labels in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(features, masks)

            # Calculate loss
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, masks, labels in val_loader:
                features = features.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                logits = model(features, masks)
                loss = criterion(logits, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.10f} | Val Loss: {avg_val_loss:.10f}"
        )

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")

    print("Aggregator training completed.")


# =========================================================================
# Inference Pipeline
# =========================================================================


def predict_pipeline(test_df, feature_dir):
    """
    Runs the final prediction pipeline using the trained aggregator.

    Args:
        test_df: DataFrame with test metadata (StudyInstanceUID).
        feature_dir: Directory containing extracted test features.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    print("Running Inference Pipeline...")

    device = Config.DEVICE

    # Load Model
    model = AttentionalRNN(
        input_dim=Config.ENCODER_HIDDEN_DIM,
        hidden_dim=Config.SEQ_HIDDEN_DIM,
        num_layers=Config.SEQ_NUM_LAYERS,
    ).to(device)

    weights_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_aggregator.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained aggregator weights.")
    else:
        print("WARNING: No trained aggregator found. Using random weights.")

    model.eval()

    # Dataset & Loader
    test_dataset = FeatureSequenceDataset(test_df, feature_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.SEQ_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_probs = []

    with torch.no_grad():
        for features, masks, _ in test_loader:
            features = features.to(device)
            masks = masks.to(device)

            # Forward
            logits = model(features, masks)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0)
    else:
        # Handle empty test set case
        all_probs = np.zeros((len(test_df), 8))

    # Get Study IDs
    study_ids = test_df["StudyInstanceUID"].tolist()

    # Format and Save Submission
    submission_df = save_predictions(study_ids, all_probs)
    print(f"Submission saved to {Config.SUBMISSION_DIR}/submission.csv")

    return submission_df
