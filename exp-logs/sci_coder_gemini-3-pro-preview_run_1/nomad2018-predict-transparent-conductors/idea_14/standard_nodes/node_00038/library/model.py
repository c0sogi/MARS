import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicEncoder(nn.Module):
    """
    Encodes per-atom features into a latent representation.
    Input: (N_atoms, 12)
    Output: (N_atoms, 128)
    """

    def __init__(self, input_dim=12, hidden_dim=512, latent_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    """
    Encodes global crystal features into a latent representation.
    Input: (Batch_size, 12)
    Output: (Batch_size, 64)
    """

    def __init__(self, input_dim=12, hidden_dim=256, latent_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class DCPDS_Model(nn.Module):
    """
    Dual-Coordinate Potential Deep Sets Model.
    Fuses atomic and global information using dual pooling.
    """

    def __init__(self, config=Config):
        super().__init__()

        # Atomic Stream
        self.atomic_encoder = AtomicEncoder(
            input_dim=12,
            hidden_dim=config.ATOMIC_HIDDEN_DIM,
            latent_dim=config.ATOMIC_LATENT_DIM,
            dropout=config.DROPOUT_RATE,
        )

        # Global Stream
        self.global_encoder = GlobalEncoder(
            input_dim=12,
            hidden_dim=config.GLOBAL_HIDDEN_DIM,
            latent_dim=config.GLOBAL_LATENT_DIM,
            dropout=config.DROPOUT_RATE,
        )

        # Fusion Head
        # Input: Mean Pool (128) + Max Pool (128) + Global (64) = 320
        fusion_input_dim = (2 * config.ATOMIC_LATENT_DIM) + config.GLOBAL_LATENT_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Predicts formation_energy and bandgap_energy
        )

    def forward(self, atomic_feats, batch_idx, global_feats):
        # 1. Encode Atoms
        atom_emb = self.atomic_encoder(atomic_feats)

        # 2. Dual Pooling (Scatter)
        # batch_idx maps atoms to crystals. dim_size ensures output matches batch size.
        batch_size = global_feats.shape[0]
        mean_pool = scatter_mean(atom_emb, batch_idx, dim=0, dim_size=batch_size)
        max_pool, _ = scatter_max(atom_emb, batch_idx, dim=0, dim_size=batch_size)

        # 3. Encode Global Context
        glob_emb = self.global_encoder(global_feats)

        # 4. Fusion
        combined = torch.cat([mean_pool, max_pool, glob_emb], dim=1)

        # 5. Regression
        output = self.head(combined)
        return output


def train_model(model, train_loader, val_loader, config=Config):
    """
    Trains the DCPDS model with early stopping and LR scheduling.
    """
    device = torch.device(config.DEVICE)
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )
    criterion = (
        nn.MSELoss()
    )  # MSE on log-transformed targets corresponds to MSLE on original

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(f"Model: DCPDS (Dual-Coordinate Potential Deep Sets)")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atomic_feats, batch_idx, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_idx = batch_idx.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(atomic_feats, batch_idx, global_feats)
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
                atomic_feats, batch_idx, global_feats, targets, _ = batch

                atomic_feats = atomic_feats.to(device)
                batch_idx = batch_idx.to(device)
                global_feats = global_feats.to(device)
                targets = targets.to(device)

                outputs = model(atomic_feats, batch_idx, global_feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)

        val_loss /= len(val_loader.dataset)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f} | LR: {current_lr:.2e}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print(f"  -> Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")


def generate_submission(model, test_loader, config=Config):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device(config.DEVICE)
    model.to(device)

    # Load best weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No checkpoint found, using current model weights.")

    model.eval()
    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_feats, batch_idx, global_feats, _, ids = batch

            atomic_feats = atomic_feats.to(device)
            batch_idx = batch_idx.to(device)
            global_feats = global_feats.to(device)

            outputs = model(atomic_feats, batch_idx, global_feats)

            # Inverse transform: exp(x) - 1 (since we trained on log1p)
            preds = torch.expm1(outputs).cpu().numpy()

            for i, sample_id in enumerate(ids):
                results.append(
                    {
                        "id": sample_id,
                        "formation_energy_ev_natom": preds[i, 0],
                        "bandgap_energy_ev": preds[i, 1],
                    }
                )

    # Save to CSV
    df = pd.DataFrame(results)
    # Ensure column order
    df = df[["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]
    df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
