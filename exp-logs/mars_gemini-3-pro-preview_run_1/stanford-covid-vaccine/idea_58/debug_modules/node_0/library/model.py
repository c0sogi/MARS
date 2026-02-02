import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import calculate_mcrmse

# =========================================================================
# Model Architecture
# =========================================================================


class VectorScaledResidualBlock(nn.Module):
    """
    A residual block with Pre-LayerNorm, BiGRU, Dropout, and learnable Vector Scaling.
    Maintains the full residual stream width (Wide-Stream) of 512.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        # BiGRU projecting to the same width (hidden_dim).
        # Since it is bidirectional, output dim is 2 * hidden_size.
        # To maintain stream width W=512, hidden_size must be 256.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Channel-Wise (Vector) Residual Scaling
        # Learnable diagonal matrix (vector) initialized to 1.0 (Identity)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        residual = x

        # Pre-LayerNorm
        out = self.norm(x)

        # BiGRU
        out, _ = self.gru(out)

        # Dropout (Applied within residual branch)
        out = self.dropout(out)

        # Vector Scaling (Element-wise multiplication)
        out = out * self.scale

        # Residual Connection
        return residual + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of the input layers (Global Static Aggregation).
    """

    def __init__(self, n_layers):
        super().__init__()
        # Initialize weights to zeros (resulting in uniform distribution after softmax initially)
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, layers):
        """
        Args:
            layers: List of tensors, each of shape (Batch, Seq, Dim)
        Returns:
            Tensor of shape (Batch, Seq, Dim)
        """
        # Stack layers: (Batch, Seq, Dim, N_layers)
        stacked = torch.stack(layers, dim=-1)

        # Compute normalized weights via Softmax
        w = F.softmax(self.weights, dim=0)

        # Weighted sum
        out = (stacked * w).sum(dim=-1)
        return out


class RNAModel(nn.Module):
    """
    Vector-Scaled High-Capacity Wide-Stream BiGRU Model.
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embed = nn.Embedding(4, Config.EMBED_DIM_SEQ)
        self.loop_embed = nn.Embedding(7, Config.EMBED_DIM_LOOP)
        # Note: Distance embedding is fixed sinusoidal (passed as input tensor)

        # 2. Stem
        # Projects concatenated inputs (256) to the hidden dimension (512)
        self.stem_gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,
            bidirectional=True,
            batch_first=True,
        )
        # No dropout after stem as per instructions

        # 3. Backbone (Residual Blocks)
        # 6 Blocks maintaining 512 width
        self.blocks = nn.ModuleList(
            [
                VectorScaledResidualBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation (Scalar Mixture)
        # Aggregates Stem output + 6 Block outputs = 7 layers
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # 5. Output Head
        # Shared Linear Projection to 3 targets
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, seq, loop, dist):
        # seq: (B, L)
        # loop: (B, L)
        # dist: (B, L, 64)

        # Embeddings
        emb_seq = self.seq_embed(seq)
        emb_loop = self.loop_embed(loop)

        # Concatenate Heterogeneous Features
        # (B, L, 128+64+64=256)
        x = torch.cat([emb_seq, emb_loop, dist], dim=-1)

        # Stem
        x, _ = self.stem_gru(x)  # (B, L, 512)

        # Collect outputs for mixture (Stem is first)
        layer_outputs = [x]

        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        # Aggregate
        x_agg = self.mixture(layer_outputs)

        # Projection
        logits = self.head(x_agg)  # (B, L, 3)

        return logits


# =========================================================================
# Training & Inference Functions
# =========================================================================


def masked_mse_loss(preds, targets, mask):
    """
    Calculates MSE loss only on valid positions specified by the mask.
    Args:
        preds: (B, L, C)
        targets: (B, L, C)
        mask: (B, L) - 1.0 for valid positions, 0.0 otherwise
    """
    # Squared Error
    se = (preds - targets) ** 2

    # Expand mask to cover channels: (B, L, 1)
    mask_expanded = mask.unsqueeze(-1)

    # Apply mask
    masked_se = se * mask_expanded

    # Average over valid elements
    # Count of valid elements = sum(mask) * num_channels
    # Add epsilon to avoid division by zero
    valid_count = mask_expanded.sum() * preds.shape[-1] + 1e-8

    loss = masked_se.sum() / valid_count
    return loss


def train_model(train_loader, val_loader):
    """
    Trains the RNAModel with the specified configuration.
    """
    device = Config.DEVICE
    model = RNAModel().to(device)

    # Optimizer: AdamW with low weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")
    print(
        f"Model: Vector-Scaled High-Capacity Wide-Stream BiGRU (Dim: {Config.HIDDEN_DIM})"
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        # Training Loop
        for seq, loop, dist, targets, mask in train_loader:
            seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)
            targets, mask = targets.to(device), mask.to(device)

            optimizer.zero_grad()

            preds = model(seq, loop, dist)
            loss = masked_mse_loss(preds, targets, mask)

            loss.backward()

            # Gradient Clipping (Critical for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for seq, loop, dist, targets, mask in val_loader:
                seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)
                targets = targets.to(device)

                preds = model(seq, loop, dist)

                # Slice to scored positions (first 68) for metric calculation
                pred_len = Config.PRED_LEN
                preds_sliced = preds[:, :pred_len, :]
                targets_sliced = targets[:, :pred_len, :]

                val_preds_list.append(preds_sliced.cpu().numpy())
                val_targets_list.append(targets_sliced.cpu().numpy())

        # Concatenate
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)

        # Calculate MCRMSE (Average of column-wise RMSEs)
        val_mcrmse = calculate_mcrmse(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.6f}")

    # Load best model for return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict_and_submit(model, test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = Config.DEVICE
    model.eval()

    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for seq, loop, dist, _, _, ids in test_loader:
            seq, loop, dist = seq.to(device), loop.to(device), dist.to(device)

            # Forward pass
            preds = model(seq, loop, dist)  # (B, 107, 3)
            preds = preds.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds)

    # Concatenate predictions: (N_samples, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model predicts: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (2)
    # Unscored: deg_pH10, deg_50C (Set to 0.0)

    submission_rows = []
    seq_len = Config.SEQ_LEN  # 107

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 3)

        for pos in range(seq_len):
            id_seqpos = f"{sample_id}_{pos}"

            reactivity = float(sample_preds[pos, 0])
            deg_Mg_pH10 = float(sample_preds[pos, 1])
            deg_Mg_50C = float(sample_preds[pos, 2])

            # Unscored columns
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": id_seqpos,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_path = Config.SUBMISSION_PATH
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
