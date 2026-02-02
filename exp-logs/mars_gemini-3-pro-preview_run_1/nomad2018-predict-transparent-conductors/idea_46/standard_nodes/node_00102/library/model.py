import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOM_INPUT_DIM,
    GLOBAL_INPUT_DIM,
    ATOM_HIDDEN_DIM,
    GLOBAL_HIDDEN_DIM,
    DROPOUT_RATE,
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TEST_METADATA_PATH,
)
from library.data_processing import get_dataloaders

# Set random seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# ==========================================
# Model Architecture
# ==========================================


class AtomicEncoder(nn.Module):
    """
    Wide MLP to encode atomic features into a high-dimensional latent space.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GlobalEncoder(nn.Module):
    """
    High-Capacity MLP to encode global crystal features.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
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
        )

    def forward(self, x):
        return self.net(x)


class GatedFusionHead(nn.Module):
    """
    Fuses pooled atomic features and global features using a gating mechanism.
    The global context modulates the atomic features before regression.
    """

    def __init__(self, atom_pooled_dim, global_dim, output_dim, dropout):
        super().__init__()
        # Gate generator: Global -> Gate for Atom
        self.gate_generator = nn.Sequential(
            nn.Linear(global_dim, atom_pooled_dim), nn.Sigmoid()
        )

        # Regressor
        fusion_dim = atom_pooled_dim + global_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, z_atom, z_global):
        # z_atom: (Batch, atom_pooled_dim)
        # z_global: (Batch, global_dim)

        # Generate gate from global features
        gate = self.gate_generator(z_global)

        # Modulate atomic features
        z_atom_gated = z_atom * gate

        # Concatenate and regress
        z_fused = torch.cat([z_atom_gated, z_global], dim=1)
        return self.regressor(z_fused)


class GPIMSDS(nn.Module):
    """
    Gated Physics-Informed Multi-Scale Deep Sets (GPI-MS-DS).
    """

    def __init__(self):
        super().__init__()
        self.atom_encoder = AtomicEncoder(ATOM_INPUT_DIM, ATOM_HIDDEN_DIM, DROPOUT_RATE)
        self.global_encoder = GlobalEncoder(
            GLOBAL_INPUT_DIM, GLOBAL_HIDDEN_DIM, DROPOUT_RATE
        )

        # Dual pooling (Mean + Max) results in 2 * ATOM_HIDDEN_DIM
        self.fusion_head = GatedFusionHead(
            ATOM_HIDDEN_DIM * 2, GLOBAL_HIDDEN_DIM, 2, DROPOUT_RATE
        )

    def forward(self, atom_features, global_features, batch_index):
        # 1. Encode Atoms
        h_atom = self.atom_encoder(atom_features)

        # 2. Pool Atoms (Dual Pooling: Mean + Max)
        batch_size = global_features.size(0)

        # Mean Pooling
        h_mean = scatter_mean(h_atom, batch_index, dim=0, dim_size=batch_size)

        # Max Pooling
        h_max, _ = scatter_max(h_atom, batch_index, dim=0, dim_size=batch_size)

        # Concatenate pooled features
        z_atom = torch.cat([h_mean, h_max], dim=1)

        # 3. Encode Global
        z_global = self.global_encoder(global_features)

        # 4. Fuse and Predict
        out = self.fusion_head(z_atom, z_global)

        return out


# ==========================================
# Training and Evaluation Logic
# ==========================================


def train_model(load_cached_data=True):
    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPIMSDS().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atom_feats = batch["atom_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            targets = batch["targets"].to(device)

            # Log transform targets: log(1 + y)
            targets_log = torch.log1p(targets)

            optimizer.zero_grad()
            outputs = model(atom_feats, global_feats, batch_idx)
            loss = criterion(outputs, targets_log)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_rmsle = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atom_feats = batch["atom_features"].to(device)
                global_feats = batch["global_features"].to(device)
                batch_idx = batch["batch_index"].to(device)
                targets = batch["targets"].to(device)

                targets_log = torch.log1p(targets)

                outputs = model(atom_feats, global_feats, batch_idx)
                loss = criterion(outputs, targets_log)
                val_loss += loss.item() * targets.size(0)

                # Calculate RMSLE on original scale (which matches MSE on log scale)
                # RMSLE = sqrt(MSE(log(1+pred), log(1+true)))
                # Since outputs are already log(1+pred), this is just sqrt(MSE)
                # But let's calculate explicitly for clarity if needed, or rely on loss
                # The competition metric is RMSLE.
                # Our loss is MSE on log space. sqrt(MSE_log) is roughly RMSLE.

        val_loss /= len(val_loader.dataset)
        val_rmsle = np.sqrt(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss (MSE_Log): {train_loss:.6f} - Val Loss (MSE_Log): {val_loss:.6f} - Val RMSLE: {val_rmsle:.6f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")
    return model, test_loader, device


def generate_submission(model, test_loader, device):
    print("Generating submission...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    predictions = []
    ids = []

    with torch.no_grad():
        for batch in test_loader:
            atom_feats = batch["atom_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_idx = batch["batch_index"].to(device)
            batch_ids = batch["ids"].numpy()

            outputs = model(atom_feats, global_feats, batch_idx)

            # Inverse transform: exp(x) - 1
            preds = torch.expm1(outputs).cpu().numpy()

            # Ensure non-negative
            preds = np.maximum(preds, 0.0)

            predictions.append(preds)
            ids.append(batch_ids)

    predictions = np.concatenate(predictions, axis=0)
    ids = np.concatenate(ids, axis=0)

    # Create submission DataFrame
    sub_df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": predictions[:, 0],
            "bandgap_energy_ev": predictions[:, 1],
        }
    )

    # Sort by ID
    sub_df = sub_df.sort_values("id")

    # Save
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(sub_df.head())


def run_pipeline():
    model, test_loader, device = train_model(load_cached_data=True)
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    run_pipeline()
