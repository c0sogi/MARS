import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from library.config import Config
from library.utils import vector_to_angles


class TemporalCNN(nn.Module):
    """
    Temporal 1D Convolutional Neural Network for Neutrino Direction Prediction.
    """

    def __init__(self):
        super(TemporalCNN, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.output_dim = Config.OUTPUT_DIM
        self.num_layers = Config.NUM_LAYERS
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout_p = Config.DROPOUT

        # Feature Extractor: Stack of 1D Convolutions
        layers = []
        in_channels = self.input_dim

        for _ in range(self.num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            in_channels = self.hidden_dim

        self.features = nn.Sequential(*layers)

        # Prediction Head
        # Global Max Pooling is applied in forward(), reducing (B, C, L) to (B, C)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Features)
        Returns:
            Tensor of shape (Batch, 3) representing direction vector.
        """
        # Permute for Conv1d: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)

        # Convolutional Layers
        x = self.features(x)
        # Shape: (Batch, Hidden_Dim, Seq_Len)

        # Global Max Pooling: Max over the time dimension
        x, _ = torch.max(x, dim=2)
        # Shape: (Batch, Hidden_Dim)

        # MLP Head
        x = self.head(x)
        # Shape: (Batch, Output_Dim)

        return x


def cosine_similarity_loss(pred, target):
    """
    Computes 1 - cosine_similarity between predicted and target vectors.
    """
    # Normalize vectors to ensure they are unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)
    target_norm = F.normalize(target, p=2, dim=1)

    # Cosine similarity is dot product of normalized vectors
    cos_sim = torch.sum(pred_norm * target_norm, dim=1)

    # Loss is 1 - mean similarity
    return 1.0 - cos_sim.mean()


def compute_angular_error(pred, target):
    """
    Computes mean angular error in radians between predicted and target vectors.
    """
    with torch.no_grad():
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        # Dot product clamped to [-1, 1] to avoid numerical issues with acos
        dot_prod = torch.sum(pred_norm * target_norm, dim=1)
        dot_prod = torch.clamp(dot_prod, -1.0, 1.0)

        angles = torch.acos(dot_prod)
        return angles.mean().item()


def train_model(model, train_loader, val_loader, device):
    """
    Trains the model with Early Stopping and saves the best checkpoint.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.IDEA_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.NUM_EPOCHS):
        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        start_time = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = cosine_similarity_loss(pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps if train_steps > 0 else 0.0

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_error_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = cosine_similarity_loss(pred, y)
                error = compute_angular_error(pred, y)

                val_loss_sum += loss.item()
                val_error_sum += error
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
        avg_val_error = val_error_sum / val_steps if val_steps > 0 else 0.0

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed}s | "
            f"Train Loss: {avg_train_loss} | "
            f"Val Loss: {avg_val_loss} | "
            f"Val MAE: {avg_val_error}"
        )

        # Scheduler Step
        scheduler.step(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights before returning
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set and saves them to submission.csv.
    """
    model.eval()
    model = model.to(device)

    event_ids = []
    azimuths = []
    zeniths = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for x, batch_event_ids in test_loader:
            x = x.to(device)
            pred_vecs = model(x)

            # Convert vectors to angles
            # vector_to_angles handles torch tensors and returns normalized angles
            az, zen = vector_to_angles(pred_vecs)

            # Move to CPU and numpy
            az = az.cpu().numpy()
            zen = zen.cpu().numpy()
            ids = batch_event_ids.numpy()

            event_ids.extend(ids)
            azimuths.extend(az)
            zeniths.extend(zen)

    # Create DataFrame
    df = pd.DataFrame({"event_id": event_ids, "azimuth": azimuths, "zenith": zeniths})

    # Sort by event_id to match sample submission structure if needed
    df = df.sort_values("event_id")

    # Save to CSV
    save_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {save_path}...")
    df.to_csv(save_path, index=False)
    print("Submission saved successfully.")
