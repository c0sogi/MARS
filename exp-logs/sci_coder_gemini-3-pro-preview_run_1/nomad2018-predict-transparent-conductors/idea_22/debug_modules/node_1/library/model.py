import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library.utils import get_device, rmsle


class AtomicStream(nn.Module):
    """
    Atomic Stream: Multi-Scalar Point Processor.
    Encodes local atomic features (One-hot, Coords, d_min, d_mean) into a latent space.
    Uses a Wide MLP with Batch Normalization and Dropout.
    """

    def __init__(
        self,
        input_dim=Config.ATOMIC_INPUT_DIM,
        hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        output_dim=Config.LATENT_DIM,
        dropout=Config.DROPOUT_RATE,
        use_bn=Config.USE_BATCH_NORM,
    ):
        super().__init__()

        layers = []

        # Layer 1: Expansion
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Layer 2: Wide Processing
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Layer 3: Projection (No activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, N_atoms, Input_Dim)
        # Flatten for linear layers: (Batch * N_atoms, Input_Dim)
        b, n, d = x.shape
        x_flat = x.view(-1, d)
        out_flat = self.net(x_flat)
        # Reshape back: (Batch, N_atoms, Output_Dim)
        return out_flat.view(b, n, -1)


class GlobalStream(nn.Module):
    """
    Global Stream: Thermodynamic Context Encoder.
    Encodes macroscopic features (Lattice, Volume, Density, Stoichiometry).
    """

    def __init__(
        self,
        input_dim=Config.GLOBAL_INPUT_DIM,
        hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        output_dim=Config.LATENT_DIM,
        dropout=Config.DROPOUT_RATE,
        use_bn=Config.USE_BATCH_NORM,
    ):
        super().__init__()

        layers = []

        # Layer 1
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Layer 2
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Layer 3: Projection
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        return self.net(x)


class MSNWDSModel(nn.Module):
    """
    Multi-Scalar Neighborhood Wide Deep Sets (MSN-WDS).
    Fuses local atomic embeddings (aggregated via Dual Pooling) with global context.
    """

    def __init__(self):
        super().__init__()

        # Streams
        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Dimension:
        # Atomic (Mean Pool + Max Pool) = 2 * Latent
        # Global = 1 * Latent
        # Total = 3 * Latent
        fusion_dim = Config.LATENT_DIM * 3

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256) if Config.USE_BATCH_NORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 2),  # Predicts 2 targets
        )

    def forward(self, atomic_x, global_x, mask):
        """
        Args:
            atomic_x: (Batch, N, Atomic_Dim)
            global_x: (Batch, Global_Dim)
            mask: (Batch, N) Boolean mask (True for real atoms)
        """
        # 1. Process Atomic Stream
        # atomic_emb: (Batch, N, Latent)
        atomic_emb = self.atomic_stream(atomic_x)

        # 2. Dual Pooling (Mean + Max)
        # Masking for Mean Pooling
        mask_float = mask.unsqueeze(-1).float()  # (B, N, 1)
        sum_emb = torch.sum(atomic_emb * mask_float, dim=1)
        counts = torch.sum(mask_float, dim=1).clamp(min=1.0)
        mean_emb = sum_emb / counts

        # Masking for Max Pooling (fill padding with -inf)
        # Use a large negative number instead of -inf to avoid NaNs in gradients if all masked
        atomic_emb_max = atomic_emb.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_emb = torch.max(atomic_emb_max, dim=1)[0]

        # 3. Process Global Stream
        # global_emb: (Batch, Latent)
        global_emb = self.global_stream(global_x)

        # 4. Fusion
        # Concatenate: [Mean, Max, Global]
        fused = torch.cat([mean_emb, max_emb, global_emb], dim=1)

        # 5. Prediction
        output = self.head(fused)

        return output


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=Config.NUM_EPOCHS,
    lr=Config.LEARNING_RATE,
    device=None,
):
    """
    Training loop with validation and early stopping.
    """
    if device is None:
        device = get_device()

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atomic = batch["atomic"].to(device)
            glob = batch["global"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(atomic, glob, mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * atomic.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                atomic = batch["atomic"].to(device)
                glob = batch["global"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["target"].to(device)

                outputs = model(atomic, glob, mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * atomic.size(0)

                # Collect for RMSLE calculation (inverse transform log1p)
                # Targets are already log1p transformed in preprocessing
                val_preds.append(torch.expm1(outputs).cpu().numpy())
                val_targets.append(torch.expm1(targets).cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Calculate RMSLE on original scale
        val_preds_concat = np.concatenate(val_preds, axis=0)
        val_targets_concat = np.concatenate(val_targets, axis=0)

        # Calculate column-wise RMSLE
        rmsle_form = rmsle(val_targets_concat[:, 0], val_preds_concat[:, 0])
        rmsle_band = rmsle(val_targets_concat[:, 1], val_preds_concat[:, 1])
        avg_rmsle = (rmsle_form + rmsle_band) / 2.0

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | RMSLE Form: {rmsle_form:.6f} | RMSLE Band: {rmsle_band:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def generate_submission(model, test_loader, device=None):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    if device is None:
        device = get_device()

    model.eval()
    ids = []
    preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            atomic = batch["atomic"].to(device)
            glob = batch["global"].to(device)
            mask = batch["mask"].to(device)
            batch_ids = batch["id"]

            outputs = model(atomic, glob, mask)

            # Inverse transform: exp(x) - 1
            # Clamp to avoid negative energy predictions if any
            outputs_orig = torch.expm1(outputs).cpu().numpy()
            outputs_orig = np.maximum(outputs_orig, 0.0)

            ids.extend(batch_ids)
            preds.append(outputs_orig)

    all_preds = np.concatenate(preds, axis=0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": all_preds[:, 0],
            "bandgap_energy_ev": all_preds[:, 1],
        }
    )

    # Sort by ID
    df = df.sort_values("id")

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    return df
