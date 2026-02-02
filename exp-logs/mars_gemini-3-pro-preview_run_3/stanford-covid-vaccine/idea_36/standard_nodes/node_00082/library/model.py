import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.data import get_loaders


# ==========================================
# Component: Decoupled Structural Interaction
# ==========================================
class DecoupledInteraction(nn.Module):
    """
    Implements the Decoupled Structural Interaction mechanism with Post-Normalization.

    Logic:
    1. Gather neighbor states h_j based on pair_indices.
    2. Mask h_j if the base is unpaired (zero-masking).
    3. Compute message m_ij derived ONLY from h_j (Decoupled).
    4. Compute gate g_ij based on context [h_i; h_j].
    5. Update h_i = h_i + g_ij * m_ij.
    6. Apply LayerNorm (Post-Norm).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)
        self.w_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases
            pair_masks: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        B, L, D = h.shape

        # 1. Gather Neighbor States
        # Expand indices to match hidden dim: (B, L, D)
        gather_indices = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        # Gather along sequence dimension (dim=1)
        h_neighbor = torch.gather(h, 1, gather_indices)

        # 2. Zero-Masking for Unpaired Bases
        # mask shape: (B, L, 1)
        mask = pair_masks.unsqueeze(-1)
        h_neighbor = h_neighbor * mask

        # 3. Decoupled Message
        # m = GELU(W_msg(h_neighbor))
        m = F.gelu(self.w_msg(h_neighbor))

        # 4. Context-Aware Gating
        # cat_input: (B, L, 2*D)
        cat_input = torch.cat([h, h_neighbor], dim=-1)
        g = torch.sigmoid(self.w_gate(cat_input))

        # 5. Residual Injection
        h_res = h + g * m

        # 6. Post-Normalization (Critical for stability)
        h_out = self.layer_norm(h_res)

        return h_out


# ==========================================
# Model: Deep Decoupled Post-Norm BiGRU
# ==========================================
class DDPNBiGRU(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Convolutional Stem
        # Projects sparse one-hot features to dense embedding
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.NUM_NODE_FEATURES,
                out_channels=Config.STEM_FILTERS,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input dimension for the first GRU layer
        current_dim = Config.STEM_FILTERS

        # BiGRU Hidden Dimension (per direction)
        gru_hidden = Config.HIDDEN_DIM
        # BiGRU Output Dimension (concatenated)
        gru_out_dim = gru_hidden * 2

        for i in range(Config.NUM_LAYERS):
            # Bidirectional GRU
            self.gru_layers.append(
                nn.GRU(
                    input_size=current_dim,
                    hidden_size=gru_hidden,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Interaction Module
            # Applied after every block EXCEPT the final block
            if i < Config.NUM_LAYERS - 1:
                self.interaction_layers.append(DecoupledInteraction(gru_out_dim))

            # Dropout
            # Applied between blocks
            if i < Config.NUM_LAYERS - 1:
                self.dropouts.append(nn.Dropout(Config.DROPOUT))

            # Next layer input is current layer output
            current_dim = gru_out_dim

        # 3. Output Head
        self.head = nn.Linear(gru_out_dim, Config.NUM_TARGETS)

    def forward(self, x, pair_indices, pair_masks):
        # x: (B, L, 14)

        # Stem: Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = x.permute(0, 2, 1)  # Back to (B, L, C)

        # Backbone
        for i in range(Config.NUM_LAYERS):
            # GRU
            x, _ = self.gru_layers[i](x)

            # Interaction (if applicable for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_masks)

            # Dropout (if applicable)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)

        # Head
        logits = self.head(x)
        return logits


# ==========================================
# Training & Utility Functions
# ==========================================
def loss_fn(preds, targets, mask):
    """
    Differentiable MCRMSE approximation for the 3 scored columns.
    """
    # Scored columns indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    loss = 0.0
    # Add small epsilon for numerical stability in sqrt
    eps = 1e-6

    for idx in scored_indices:
        p = preds[:, :, idx]
        t = targets[:, :, idx]

        # Squared Error
        se = (p - t) ** 2

        # Masking
        # Mask is (B, L), 1 for valid positions
        se = se * mask

        # Mean Squared Error over valid positions
        # Sum over batch and sequence
        total_valid = mask.sum()
        if total_valid < 1:
            total_valid = 1.0

        mse = se.sum() / total_valid

        # RMSE
        rmse = torch.sqrt(mse + eps)
        loss += rmse

    # Average over the 3 columns
    return loss / len(scored_indices)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move to device
        X = batch["X"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        y = batch["y"].to(device)
        target_masks = batch["target_masks"].to(device)

        optimizer.zero_grad()

        # Forward
        preds = model(X, pair_indices, pair_masks)

        # Loss
        loss = loss_fn(preds, y, target_masks)

        # Backward
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            y = batch["y"]  # Keep on CPU for metric calc

            preds = model(X, pair_indices, pair_masks)
            all_preds.append(preds.cpu())
            all_targets.append(y)

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    mcrmse, col_scores = calculate_mcrmse(all_preds, all_targets)
    return mcrmse, col_scores


def generate_submission(model, loader, device, output_path):
    model.eval()
    ids = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            batch_ids = batch["id"]

            # Predict
            preds = model(X, pair_indices, pair_masks)  # (B, 107, 5)

            ids.extend(batch_ids)
            preds_list.append(preds.cpu().numpy())

    preds_array = np.concatenate(preds_list, axis=0)  # (N, 107, 5)

    # Format for submission
    # We need to flatten: id_seqpos
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids):
        sample_preds = preds_array[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_data = {
                "id_seqpos": row_id,
                "reactivity": sample_preds[seqpos, 0],
                "deg_Mg_pH10": sample_preds[seqpos, 1],
                "deg_pH10": sample_preds[seqpos, 2],
                "deg_Mg_50C": sample_preds[seqpos, 3],
                "deg_50C": sample_preds[seqpos, 4],
            }
            submission_rows.append(row_data)

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, test_loader = get_loaders()

    # Initialize Model
    model = DDPNBiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mcrmse, val_scores = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  New best model saved! Scores: {val_scores}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for submission
    print(f"Loading best model (MCRMSE: {best_mcrmse:.6f})...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    # Generate Submission
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
