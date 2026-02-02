import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_mean, scatter_max

from library.config import Config
from library.data import get_train_val_loaders, get_test_loader
from library.utils import set_seed, inverse_log_transform


class AtomicStream(nn.Module):
    """
    Context-Aware Point Processor.
    Encodes per-atom features into a latent representation.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Projection
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.
    Encodes macroscopic properties.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Projection
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class SCC_WDS_Net(nn.Module):
    """
    Soft-Chemical Context Wide Deep Sets Network.
    Integrates Atomic and Global streams via Late Fusion.
    """

    def __init__(self):
        super().__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=Config.ATOMIC_FEATURE_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=Config.GLOBAL_FEATURE_DIM,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            dropout=Config.DROPOUT,
        )

        # 3. Fusion Head
        # Input: (Mean Pool + Max Pool) + Global Embedding
        fusion_input_dim = (2 * Config.HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 2),  # Output: formation_energy, bandgap_energy
        )

    def forward(self, atomic_features, global_features, batch_index):
        # Process Atomic Stream
        atom_emb = self.atomic_stream(atomic_features)

        # Aggregation (Dual Pooling)
        # scatter_mean: (Batch_Size, Hidden_Dim)
        mean_pool = scatter_mean(atom_emb, batch_index, dim=0)
        # scatter_max: (values, indices) -> take values
        max_pool, _ = scatter_max(atom_emb, batch_index, dim=0)

        # Handle potential size mismatch if batch_index doesn't cover 0..B-1 (unlikely with proper collate)
        # torch_scatter automatically sizes dim 0 based on max(index) + 1

        atomic_repr = torch.cat([mean_pool, max_pool], dim=1)

        # Process Global Stream
        global_repr = self.global_stream(global_features)

        # Late Fusion
        combined = torch.cat([atomic_repr, global_repr], dim=1)

        # Prediction
        output = self.fusion_head(combined)

        return output


def train_model(
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    epochs=Config.EPOCHS,
    patience=Config.PATIENCE,
    load_cached_data=True,
):
    """
    Trains the SCC_WDS_Net model.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Load Data
    train_loader, val_loader = get_train_val_loaders(batch_size, load_cached_data)

    # Initialize Model
    model = SCC_WDS_Net().to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training on {device}...")
    print(f"Model: SCC_WDS_Net")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(atomic_feats, global_feats, batch_idx)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                atomic_feats = batch["atomic_features"].to(device)
                global_feats = batch["global_features"].to(device)
                batch_idx = batch["batch_index"].to(device)
                targets = batch["target"].to(device)

                outputs = model(atomic_feats, global_feats, batch_idx)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

        val_loss /= len(val_loader.dataset)

        # Update Scheduler
        scheduler.step(val_loss)

        # Logging
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


def generate_submission(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Generates predictions for the test set using the best trained model.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Load Test Data
    test_loader = get_test_loader(batch_size, load_cached_data)

    # Load Model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train first."
        )

    model = SCC_WDS_Net().to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []
    ids = []

    print("\nGenerating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            batch_ids = batch["id"].numpy()

            outputs = model(atomic_feats, global_feats, batch_idx)

            # Inverse transform log-targets
            preds = inverse_log_transform(outputs.cpu().numpy())

            predictions.append(preds)
            ids.append(batch_ids)

    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID
    submission_df = submission_df.sort_values("id")

    # Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
    print(submission_df.head())
