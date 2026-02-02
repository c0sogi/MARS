import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A residual block consisting of Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU.
    Handles both 2D (batch, features) and 3D (batch, atoms, features) inputs automatically.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x shape can be (B, D) or (B, A, D)
        residual = x

        # Check if input is 3D (Batch, Atoms, Dim)
        is_3d = x.dim() == 3

        # First dense layer
        out = self.linear1(x)

        # BatchNorm expects (N, C) or (N, C, L).
        # If 3D, we treat Dim as Channels (C) and Atoms as Length (L).
        if is_3d:
            out = out.permute(0, 2, 1)  # (B, D, A)
            out = self.bn1(out)
            out = F.relu(out)
            out = out.permute(0, 2, 1)  # (B, A, D)
        else:
            out = self.bn1(out)
            out = F.relu(out)

        out = self.dropout(out)

        # Second dense layer
        out = self.linear2(out)

        if is_3d:
            out = out.permute(0, 2, 1)
            out = self.bn2(out)
            out = out.permute(0, 2, 1)
        else:
            out = self.bn2(out)

        # Residual connection
        out += residual
        out = F.relu(out)
        return out


class SIRDSModel(nn.Module):
    """
    Symmetry-Informed Residual Deep Sets Model.

    Architecture:
    1. Atomic Stream: Projects atomic features, processes with Residual Blocks,
       and aggregates via Mean/Max/Std pooling.
    2. Global Stream: Projects global features and processes with a Residual Block.
    3. Symmetry Stream: Embeds spacegroup ID.
    4. Fusion Head: Concatenates all streams and regresses targets.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.HIDDEN_DIM
        self.dropout = config.DROPOUT

        # --- Atomic Stream ---
        self.atomic_project = nn.Linear(config.ATOMIC_FEATURE_DIM, self.hidden_dim)
        self.atomic_blocks = nn.ModuleList(
            [
                ResidualBlock(self.hidden_dim, self.dropout)
                for _ in range(config.NUM_RES_BLOCKS)
            ]
        )

        # --- Global Stream ---
        self.global_project = nn.Linear(config.GLOBAL_FEATURE_DIM, self.hidden_dim)
        self.global_block = ResidualBlock(self.hidden_dim, self.dropout)

        # --- Symmetry Stream ---
        # +1 because spacegroups are 1-indexed (1-230), usually 0 is padding or unused
        self.symmetry_embed = nn.Embedding(
            config.NUM_SPACEGROUPS + 1, config.SYMMETRY_EMBED_DIM
        )

        # --- Fusion Head ---
        # Atomic pooling (3 * hidden) + Global (hidden) + Symmetry (embed_dim)
        fusion_input_dim = (
            (3 * self.hidden_dim) + self.hidden_dim + config.SYMMETRY_EMBED_DIM
        )

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, config.NUM_TARGETS),
        )

    def forward(self, atomic_features, global_features, symmetry, mask):
        """
        Args:
            atomic_features: (Batch, Max_Atoms, Feat_A)
            global_features: (Batch, Feat_G)
            symmetry: (Batch,) - Spacegroup IDs
            mask: (Batch, Max_Atoms) - Boolean mask (True for valid atoms)
        """

        # 1. Atomic Stream Processing
        x_atomic = self.atomic_project(atomic_features)  # (B, A, H)
        for block in self.atomic_blocks:
            x_atomic = block(x_atomic)

        # 2. Tri-Pooling (Mean, Max, Std)
        # Expand mask for broadcasting: (B, A, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Zero out invalid atoms
        x_atomic_masked = x_atomic * mask_expanded

        # Number of atoms per crystal (avoid div by zero)
        n_atoms = mask_expanded.sum(dim=1)
        n_atoms = torch.clamp(n_atoms, min=1.0)

        # Mean Pooling
        sum_atomic = x_atomic_masked.sum(dim=1)
        mean_pool = sum_atomic / n_atoms  # (B, H)

        # Max Pooling
        # Fill invalid positions with very small number before max
        x_atomic_fill = x_atomic.clone()
        x_atomic_fill[~mask] = -1e9
        max_pool = x_atomic_fill.max(dim=1)[0]  # (B, H)

        # Std Pooling
        # Var = E[x^2] - (E[x])^2
        sum_sq_atomic = (x_atomic_masked**2).sum(dim=1)
        mean_sq = sum_sq_atomic / n_atoms
        var = mean_sq - (mean_pool**2)
        # Clamp variance to avoid negative values due to precision, then sqrt
        std_pool = torch.sqrt(torch.clamp(var, min=1e-6))  # (B, H)

        # Concatenate atomic representations
        atomic_repr = torch.cat([mean_pool, max_pool, std_pool], dim=1)  # (B, 3H)

        # 3. Global Stream Processing
        x_global = self.global_project(global_features)  # (B, H)
        x_global = self.global_block(x_global)  # (B, H)

        # 4. Symmetry Stream Processing
        x_sym = self.symmetry_embed(symmetry)  # (B, Emb)

        # 5. Fusion and Regression
        combined = torch.cat([atomic_repr, x_global, x_sym], dim=1)  # (B, Fusion_Dim)
        output = self.regressor(combined)  # (B, Targets)

        return output


def train_model(model, train_loader, val_loader, config, device):
    """
    Standard training loop with Early Stopping and Learning Rate Scheduling.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            atomic = batch["atomic_features"].to(device)
            global_f = batch["global_features"].to(device)
            sym = batch["symmetry"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            preds = model(atomic, global_f, sym, mask)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * atomic.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                atomic = batch["atomic_features"].to(device)
                global_f = batch["global_features"].to(device)
                sym = batch["symmetry"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["targets"].to(device)

                preds = model(atomic, global_f, sym, mask)
                loss = criterion(preds, targets)
                val_loss += loss.item() * atomic.size(0)

        val_loss /= len(val_loader.dataset)

        # Scheduler step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Best Validation Loss: {best_val_loss:.8f}")


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.
    Applies inverse transformation (expm1) to targets since training was on log1p.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            atomic = batch["atomic_features"].to(device)
            global_f = batch["global_features"].to(device)
            sym = batch["symmetry"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["ids"]

            preds = model(atomic, global_f, sym, mask)

            # Inverse transform: log1p -> expm1
            # Targets were log(1+y), so predictions are log scale.
            # We need exp(pred) - 1 to get back to original scale.
            preds_orig = torch.expm1(preds)

            all_preds.append(preds_orig.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids
