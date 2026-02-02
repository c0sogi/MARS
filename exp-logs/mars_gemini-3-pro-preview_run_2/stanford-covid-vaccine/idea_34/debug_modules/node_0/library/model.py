import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import (
    HIDDEN_DIM,
    FEEDBACK_DIM,
    DROPOUT,
    NUM_LAYERS,
    KERNEL_SIZE,
    DILATIONS,
    RNN_HIDDEN_DIM,
    RNN_LAYERS,
    SCORED_INDICES,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    LOSS_FACTOR_FINAL,
    LOSS_FACTOR_AUX,
    INPUT_DIR,
)
from library.utils import mcrmse_loss, set_seed
from library.data import get_dataloaders

# =============================================================================
# MODEL COMPONENTS
# =============================================================================


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # Padding to maintain sequence length: (kernel_size - 1) * dilation // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)  # Applied on (B, L, C)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if dimensions mismatch
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, C_in, L)
        residual = x

        out = self.conv(x)  # (B, C_out, L)

        # Permute for LayerNorm: (B, L, C)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # Back to (B, C, L)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


class StaticBackbone(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, kernel_size, dilations, dropout):
        super().__init__()

        # Initial projection
        self.initial_conv = nn.Conv1d(input_dim, hidden_dim, 1)

        self.blocks = nn.ModuleList()
        current_dim = hidden_dim

        # DenseNet-style growth: Input to block i is concatenation of all previous outputs
        for i in range(layers):
            # Input dim grows as we concatenate previous features
            # Block input: current_dim
            # Block output: hidden_dim
            dilation = dilations[i] if i < len(dilations) else 1

            block = DilatedResidualBlock(
                in_channels=current_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_dim += hidden_dim  # Grow input for next layer

    def forward(self, x):
        # x: (B, L, In_Dim) -> Permute to (B, In_Dim, L)
        x = x.permute(0, 2, 1)

        features = []

        # Initial feature
        out = self.initial_conv(x)  # (B, Hidden, L)
        features.append(out)

        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)

            # Compute block output
            out = block(dense_input)
            features.append(out)

        # Final dense representation: Concatenate everything
        # Shape: (B, Hidden * (Layers + 1), L)
        h_dense = torch.cat(features, dim=1)
        return h_dense


class LatentFeedbackInteraction(nn.Module):
    def __init__(self, dense_dim, latent_dim, feedback_dim, dropout):
        super().__init__()

        # Project static dense features to latent space Z
        self.proj_dense = nn.Conv1d(dense_dim, latent_dim, 1)

        # Project feedback predictions to embedding E
        self.proj_pred = nn.Linear(5, feedback_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_dense, prev_pred, partner_indices):
        """
        h_dense: (B, Dense_Dim, L) - Static backbone features
        prev_pred: (B, L, 5) - Recycled predictions
        partner_indices: (B, L) - Indices of partners
        """
        B, _, L = h_dense.shape

        # 1. Project Dense Features -> Z (B, Latent, L)
        z = self.proj_dense(h_dense)
        z = z.permute(0, 2, 1)  # (B, L, Latent)

        # 2. Project Feedback -> E (B, L, Feedback)
        e = self.proj_pred(prev_pred)

        # 3. Concatenate Self Features: [Z_i, E_i]
        # Shape: (B, L, Latent + Feedback)
        node_feat = torch.cat([z, e], dim=2)

        # 4. Gather Partner Features
        # Handle -1 indices (unpaired) by clamping to 0 and masking later
        # partner_indices is (B, L)
        gather_indices = partner_indices.clone()
        gather_indices[gather_indices == -1] = 0

        # Expand indices for gather: (B, L, Feat_Dim)
        feat_dim = node_feat.size(2)
        expanded_indices = gather_indices.unsqueeze(2).expand(-1, -1, feat_dim)

        # Gather
        partner_feat = torch.gather(node_feat, 1, expanded_indices)

        # Mask unpaired positions
        mask = (partner_indices != -1).unsqueeze(2).float()
        partner_feat = partner_feat * mask

        # 5. Fusion: [Self, Partner]
        # Shape: (B, L, (Latent + Feedback) * 2)
        combined = torch.cat([node_feat, partner_feat], dim=2)
        combined = self.dropout(combined)

        return combined


class LFDCN(nn.Module):
    def __init__(self):
        super().__init__()

        # Dimensions
        # Input: 18
        # Dense Output: HIDDEN_DIM * (NUM_LAYERS + 1)
        dense_out_dim = HIDDEN_DIM * (NUM_LAYERS + 1)

        self.backbone = StaticBackbone(
            input_dim=18,
            hidden_dim=HIDDEN_DIM,
            layers=NUM_LAYERS,
            kernel_size=KERNEL_SIZE,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        self.interaction = LatentFeedbackInteraction(
            dense_dim=dense_out_dim,
            latent_dim=HIDDEN_DIM,
            feedback_dim=FEEDBACK_DIM,
            dropout=DROPOUT,
        )

        # RNN Input Dim: (Latent(64) + Feedback(32)) * 2 = 192
        rnn_input_dim = (HIDDEN_DIM + FEEDBACK_DIM) * 2

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=RNN_HIDDEN_DIM,
            num_layers=RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Head: BiGRU output is RNN_HIDDEN_DIM * 2
        self.head = nn.Linear(RNN_HIDDEN_DIM * 2, 5)

    def forward_pass(self, h_dense, prev_pred, partner_indices):
        # Interaction
        h_interact = self.interaction(h_dense, prev_pred, partner_indices)

        # RNN
        h_rnn, _ = self.gru(h_interact)

        # Head
        pred = self.head(h_rnn)
        return pred

    def forward(self, x, partner_indices, targets=None):
        """
        x: (B, L, 18)
        partner_indices: (B, L)
        targets: (B, L, 5) or None
        """
        # 1. Static Backbone (Run Once)
        h_dense = self.backbone(x)

        # 2. Iterative Refinement Loop

        # Pass 1: Zero Initialization
        B, L, _ = x.shape
        prev_pred_0 = torch.zeros(B, L, 5, device=x.device)
        pred_1 = self.forward_pass(h_dense, prev_pred_0, partner_indices)

        # Pass 2: Feedback from Pass 1
        # Detach gradients from the feedback signal
        prev_pred_1 = pred_1.detach()
        pred_2 = self.forward_pass(h_dense, prev_pred_1, partner_indices)

        if targets is not None:
            # Training Mode: Compute Loss
            # Loss = MCRMSE(pred_2) + 0.5 * MCRMSE(pred_1)
            loss_final = mcrmse_loss(pred_2, targets, SCORED_INDICES)
            loss_aux = mcrmse_loss(pred_1, targets, SCORED_INDICES)

            total_loss = (LOSS_FACTOR_FINAL * loss_final) + (LOSS_FACTOR_AUX * loss_aux)
            return total_loss, pred_2
        else:
            # Inference Mode: Return final prediction
            return pred_2


# =============================================================================
# TRAINING & INFERENCE UTILS
# =============================================================================


def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(
        load_cached_data=True, batch_size=BATCH_SIZE
    )

    # Initialize Model
    model = LFDCN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    early_stop_count = 0

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss_avg = 0.0

        for batch in train_loader:
            inputs, partner_indices, targets, masks = batch
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            loss, _ = model(inputs, partner_indices, targets)

            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item()

        train_loss_avg /= len(train_loader)

        # Validation
        model.eval()
        val_loss_avg = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, partner_indices, targets, masks = batch
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)
                targets = targets.to(device)

                # Forward returns (loss, pred) when targets provided
                loss, _ = model(inputs, partner_indices, targets)
                val_loss_avg += loss.item()

        val_loss_avg /= len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}"
        )

        scheduler.step(val_loss_avg)

        # Checkpointing
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            early_stop_count = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New Best Model Saved! Loss: {best_val_loss:.6f}")
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print("Early stopping triggered.")
                break


def generate_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating Submission...")

    # Load Test Data
    _, _, test_loader = get_dataloaders(load_cached_data=True, batch_size=BATCH_SIZE)

    # Load Model
    model = LFDCN().to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")

    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, partner_indices, _, masks = batch
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Inference: targets=None -> returns pred_2
            preds = model(inputs, partner_indices, targets=None)

            # Move to CPU
            preds = preds.cpu().numpy()

            # We need to collect IDs from the dataset, but DataLoader shuffles/batches.
            # The dataset returns (inputs, partners, targets, masks).
            # IDs are not in the batch tuple. We need to access them carefully.
            # Actually, RNADataset stores IDs. But DataLoader doesn't yield them by default unless modified.
            # The provided `data.py` `__getitem__` returns 4 items. IDs are not returned.
            # However, `test_ds.ids` exists. Since `test_loader` is not shuffled, we can index.
            # But batching makes it tricky.
            # To be safe, we rely on the order being preserved (shuffle=False for test).
            all_preds.append(preds)

    # Concatenate all predictions: (N_samples, L, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Get IDs from dataset
    test_ds = test_loader.dataset
    ids = test_ds.ids

    # Prepare Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    # Target columns in order
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        # Get seq_scored from metadata?
        # The provided `preprocess_data` uses `seq_scored` to create masks, but we don't have it here explicitly.
        # However, the submission format requires predictions for *each* sequence position (seq_length=107).
        # "Positions greater than the seq_scored value ... still need a value".
        # So we simply dump all 107 positions.

        sample_preds = all_preds[i]  # (107, 5)

        for pos in range(sample_preds.shape[0]):
            row_id = f"{sample_id}_{pos}"
            row_vals = sample_preds[pos]

            row_dict = {"id_seqpos": row_id}
            for t_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_vals[t_idx])

            submission_rows.append(row_dict)

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    # This block is strictly for local testing if run as a script.
    # The evaluation harness might import functions directly.
    train_model()
    generate_submission()
