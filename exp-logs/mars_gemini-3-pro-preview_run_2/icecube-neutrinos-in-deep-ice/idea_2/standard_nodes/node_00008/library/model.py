import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
from library.utils import vector_to_angles


class GeometricPulseAggregator(nn.Module):
    """
    A simplified PointNet-like architecture for Neutrino Direction Prediction.
    Treats the event as a cloud of pulses, applying point-wise features and
    symmetric aggregation (max pooling) to learn a global event descriptor.
    """

    def __init__(
        self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM
    ):
        super(GeometricPulseAggregator, self).__init__()

        # Shared MLP (Point-wise Feature Extraction)
        # Implemented as Conv1d with kernel_size=1
        # Input: (Batch, Features, N_Pulses)
        self.mlp_local = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Prediction Head
        # Input: (Batch, Hidden_Dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, N_Pulses, Features)
        Returns:
            torch.Tensor: Output vector of shape (Batch, 3)
        """
        # Permute to (Batch, Features, N_Pulses) for Conv1d
        x = x.permute(0, 2, 1)

        # 1. Extract local features for each pulse
        x = self.mlp_local(x)  # Output: (Batch, Hidden_Dim, N_Pulses)

        # 2. Symmetric Aggregation (Global Max Pooling)
        # Max pool over the N_Pulses dimension (dim=2)
        x = torch.max(x, 2)[0]  # Output: (Batch, Hidden_Dim)

        # 3. Predict direction vector
        x = self.head(x)  # Output: (Batch, Output_Dim)

        return x


class CosineDistanceLoss(nn.Module):
    """
    Loss function based on Cosine Similarity.
    Minimizes 1 - cos(theta) between prediction and target.
    """

    def __init__(self):
        super(CosineDistanceLoss, self).__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, pred, target):
        # pred: (Batch, 3) - raw output from model
        # target: (Batch, 3) - ground truth unit vector
        loss = 1.0 - self.cosine_sim(pred, target)
        return loss.mean()


def calculate_angular_error(pred, target):
    """
    Calculates the mean angular error in radians between predicted and target vectors.
    """
    # Normalize predictions to unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)

    # Dot product
    dot = torch.sum(pred_norm * target_norm, dim=1)

    # Clamp for numerical stability to avoid NaNs in acos
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angle in radians
    angles = torch.acos(dot)
    return angles.mean().item()


def train_model(
    model, train_loader, val_loader, optimizer, device, epochs, patience, save_path
):
    """
    Executes the training loop with validation and early stopping.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        optimizer: PyTorch optimizer.
        device: 'cuda' or 'cpu'.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model weights.
    """
    criterion = CosineDistanceLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_angular_error = 0.0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * features.size(0)
                val_angular_error += calculate_angular_error(
                    outputs, targets
                ) * features.size(0)

        val_loss /= len(val_loader.dataset)
        val_angular_error /= len(val_loader.dataset)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Angular Error: {val_angular_error}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def predict_and_submit(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test set (must preserve order).
        device: 'cuda' or 'cpu'.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    model.to(device)

    event_ids = []
    azimuths = []
    zeniths = []

    print("Generating predictions for test set...")

    # We need to access event_ids from the dataset.
    # We assume the test_loader iterates sequentially (shuffle=False).
    dataset = test_loader.dataset
    batch_start_idx = 0

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)

            # Forward pass
            outputs = model(features)

            # Convert 3D vectors to Azimuth and Zenith
            # vector_to_angles handles normalization
            az, ze = vector_to_angles(outputs)

            # Collect results
            azimuths.extend(az.cpu().numpy())
            zeniths.extend(ze.cpu().numpy())

            # Retrieve corresponding event_ids
            batch_size = features.size(0)
            batch_end_idx = batch_start_idx + batch_size

            # Get event_ids from metadata for this batch range
            batch_event_ids = dataset.meta_df.iloc[batch_start_idx:batch_end_idx][
                "event_id"
            ].values
            event_ids.extend(batch_event_ids)

            batch_start_idx = batch_end_idx

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"event_id": event_ids, "azimuth": azimuths, "zenith": zeniths}
    )

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
