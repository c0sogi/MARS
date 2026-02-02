import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import (
    WINDOW_SIZE,
    KERNEL_SIZES,
    CNN_CHANNELS,
    CNN_DROPOUT,
    CONTEXT_EMBED_DIM,
    FUSION_HIDDEN_DIM,
    FUSION_DROPOUT,
    KINEMATIC_FEATURES,
    CONTEXT_FEATURES,
    WORKING_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SAMPLE_SUBMISSION_PATH,
)
from library.dataset import get_dataloader

# ==========================================
# Model Architecture
# ==========================================


class KinematicEncoder(nn.Module):
    """
    Multi-Scale 1D CNN for processing kinematic sequence data.
    Extracts features from the center of the temporal window using parallel convolutions.
    """

    def __init__(self):
        super().__init__()
        in_channels = len(KINEMATIC_FEATURES)
        self.branches = nn.ModuleList()

        for k in KERNEL_SIZES:
            # Calculate padding to maintain temporal dimension (assuming stride=1)
            # padding = (kernel_size - 1) // 2
            padding = (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(in_channels, CNN_CHANNELS, kernel_size=k, padding=padding),
                nn.BatchNorm1d(CNN_CHANNELS),
                nn.ReLU(),
                nn.Dropout(CNN_DROPOUT),
                nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=k, padding=padding),
                nn.BatchNorm1d(CNN_CHANNELS),
                nn.ReLU(),
                nn.Dropout(CNN_DROPOUT),
            )
            self.branches.append(branch)

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> Transpose to (Batch, Features, Time)
        x = x.transpose(1, 2)

        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)

        # Concatenate along channel dimension: (Batch, Total_Channels, Time)
        out = torch.cat(branch_outputs, dim=1)

        # Extract feature vector corresponding to the center timestamp
        # Window size is fixed, so center index is constant
        center_idx = out.shape[2] // 2
        center_features = out[:, :, center_idx]

        return center_features


class ContextEncoder(nn.Module):
    """
    MLP for processing aggregated environmental context features.
    """

    def __init__(self):
        super().__init__()
        in_dim = len(CONTEXT_FEATURES)
        self.net = nn.Sequential(
            nn.Linear(in_dim, CONTEXT_EMBED_DIM * 2),
            nn.BatchNorm1d(CONTEXT_EMBED_DIM * 2),
            nn.ReLU(),
            nn.Linear(CONTEXT_EMBED_DIM * 2, CONTEXT_EMBED_DIM),
            nn.BatchNorm1d(CONTEXT_EMBED_DIM),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleKinematicCNN(nn.Module):
    """
    Fused architecture combining Kinematic and Context streams to predict residuals.
    """

    def __init__(self):
        super().__init__()
        self.kinematic_encoder = KinematicEncoder()
        self.context_encoder = ContextEncoder()

        # Calculate fusion input dimension
        kinematic_out_dim = CNN_CHANNELS * len(KERNEL_SIZES)
        fusion_in_dim = kinematic_out_dim + CONTEXT_EMBED_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_HIDDEN_DIM, FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(
                FUSION_HIDDEN_DIM // 2, 2
            ),  # Output: Delta East, Delta North (meters)
        )

    def forward(self, kinematic_seq, context_feats):
        k_emb = self.kinematic_encoder(kinematic_seq)
        c_emb = self.context_encoder(context_feats)

        # Concatenate embeddings
        fused = torch.cat([k_emb, c_emb], dim=1)

        # Predict residuals
        residuals = self.fusion_head(fused)
        return residuals


# ==========================================
# Training & Inference Utilities
# ==========================================


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    patience=PATIENCE,
):
    """
    Training loop with Early Stopping and Scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )
    criterion = nn.L1Loss()  # MAE Loss

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            kin_seq = batch["kinematic_sequence"].to(device)
            ctx_feats = batch["context_features"].to(device)
            targets = batch["target_residual"].to(device)

            optimizer.zero_grad()
            outputs = model(kin_seq, ctx_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * kin_seq.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                kin_seq = batch["kinematic_sequence"].to(device)
                ctx_feats = batch["context_features"].to(device)
                targets = batch["target_residual"].to(device)

                outputs = model(kin_seq, ctx_feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * kin_seq.size(0)

        val_loss /= len(val_loader.dataset)

        # Scheduler Step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model checkpoint
            torch.save(best_model_state, os.path.join(WORKING_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation MAE: {best_val_loss:.6f}")

    # Load best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def predict(model, test_loader, device):
    """
    Generate predictions for the test set.
    Returns predicted residuals (East, North) in meters.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            kin_seq = batch["kinematic_sequence"].to(device)
            ctx_feats = batch["context_features"].to(device)

            outputs = model(kin_seq, ctx_feats)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def generate_submission(model, device, load_cached_data=True):
    """
    Full inference pipeline: Load test data, predict residuals, reconstruct coords, save CSV.
    """
    print("Generating submission...")

    # Load Test Data (Shuffle=False to maintain order with metadata)
    test_loader = get_dataloader(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        load_cached_data=load_cached_data,
    )

    # Get Predictions (Delta East, Delta North in meters)
    pred_residuals = predict(model, test_loader, device)

    # Get Metadata to reconstruct coordinates
    # The dataset object holds the metadata dataframe corresponding to the loaded data
    test_meta = test_loader.dataset.meta

    # Ensure lengths match
    if len(pred_residuals) != len(test_meta):
        raise ValueError(
            f"Mismatch: Predictions {len(pred_residuals)} vs Metadata {len(test_meta)}"
        )

    # Reconstruction
    # We assume the 'LatitudeDegrees' and 'LongitudeDegrees' in test_meta (from sample_submission)
    # represent the baseline WLS estimate.
    # Formula:
    # Lat_new = Lat_old + (dNorth / 111320)
    # Lon_new = Lon_old + (dEast / (111320 * cos(Lat)))

    meters_per_deg = 111320.0

    baseline_lats = test_meta["LatitudeDegrees"].values
    baseline_lons = test_meta["LongitudeDegrees"].values

    d_east = pred_residuals[:, 0]
    d_north = pred_residuals[:, 1]

    new_lats = baseline_lats + (d_north / meters_per_deg)
    new_lons = baseline_lons + (
        d_east / (meters_per_deg * np.cos(np.radians(baseline_lats)))
    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {
            "tripId": test_meta["tripId"],
            "UnixTimeMillis": test_meta["UnixTimeMillis"],
            "LatitudeDegrees": new_lats,
            "LongitudeDegrees": new_lons,
        }
    )

    # Save
    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_pipeline(max_samples=None, epochs=NUM_EPOCHS, load_cached_data=True):
    """
    Main entry point for training and inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader = get_dataloader(
        "train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )
    val_loader = get_dataloader(
        "validation",
        batch_size=BATCH_SIZE,
        shuffle=False,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 2. Initialize Model
    model = MultiScaleKinematicCNN().to(device)

    # 3. Train
    model = train_model(model, train_loader, val_loader, device, epochs=epochs)

    # 4. Generate Submission
    generate_submission(model, device, load_cached_data=load_cached_data)
