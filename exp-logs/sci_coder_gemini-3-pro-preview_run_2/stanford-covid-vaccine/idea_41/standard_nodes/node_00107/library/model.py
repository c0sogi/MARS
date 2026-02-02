import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from library.config import Config
from library.data import load_data

# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class DenseDilatedBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, C, L) -> Norm expects (B, L, C)
        out = x.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)

        out = self.activation(out)
        out = self.conv(out)
        out = self.dropout(out)
        return out


class StaticBackbone(nn.Module):
    def __init__(
        self, in_channels, growth_rate, kernel_size, dilations, dropout, hidden_dim
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(current_dim, growth_rate, kernel_size, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

        self.out_proj = nn.Conv1d(current_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        features = [x]
        for block in self.blocks:
            # Dense connection: concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final projection of all concatenated features
        total_features = torch.cat(features, dim=1)
        z = self.out_proj(total_features)
        return z


class RawConditionedFeedback(nn.Module):
    def __init__(self, num_targets, raw_features_dim, feedback_dim, dropout):
        super().__init__()
        # Project raw context to a smaller dimension
        self.context_proj = nn.Conv1d(raw_features_dim, 16, kernel_size=1)

        # Input: Targets + Context
        in_dim = num_targets + 16

        # Lightweight TCN for feedback processing
        self.tcn = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, feedback_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, y_prev, raw_context):
        # y_prev: (B, 5, L)
        # raw_context: (B, 18, L)

        # Mask unscored targets in y_prev to avoid noise injection
        # Scored indices: 0, 1, 3. Unscored: 2, 4.
        mask = torch.zeros_like(y_prev)
        mask[:, [0, 1, 3], :] = 1.0
        y_masked = y_prev * mask

        # Process context
        ctx = self.context_proj(raw_context)

        # Concatenate and process
        inp = torch.cat([y_masked, ctx], dim=1)
        e_fb = self.tcn(inp)
        return e_fb


class RCRDN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Static Backbone (Heavy Feature Extraction)
        self.backbone = StaticBackbone(
            in_channels=Config.NUM_NODE_FEATURES,
            growth_rate=Config.GROWTH_RATE,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            hidden_dim=Config.HIDDEN_DIM,
        )

        # 2. Feedback Module (Lightweight, Context-Aware)
        self.feedback_module = RawConditionedFeedback(
            num_targets=Config.NUM_TARGETS,
            raw_features_dim=Config.NUM_NODE_FEATURES,
            feedback_dim=Config.FEEDBACK_DIM,
            dropout=Config.DROPOUT,
        )

        # 3. Interaction & Aggregation
        # Input to RNN: (Z + E_fb) for Self + (Z + E_fb) for Partner
        rnn_input_dim = (Config.HIDDEN_DIM + Config.FEEDBACK_DIM) * 2

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(256, Config.NUM_TARGETS)  # 128 * 2 directions

    def forward(self, x, partner_idx, partner_mask):
        # x: (B, L, C) -> Transpose to (B, C, L)
        x_t = x.transpose(1, 2)

        # 1. Run Backbone Once
        z = self.backbone(x_t)  # (B, 64, L)

        B, _, L = z.shape
        y_preds = []

        # Initialize previous predictions as zero
        y_curr = torch.zeros(B, Config.NUM_TARGETS, L, device=x.device)

        # 2. Recycling Loop
        for _ in range(Config.N_CYCLES):
            # Compute Feedback Embedding
            # Detach input y_curr to control gradient flow (stop gradient from previous cycle)
            e_fb = self.feedback_module(y_curr.detach(), x_t)  # (B, 32, L)

            # Concatenate Static Z and Dynamic E_fb
            h = torch.cat([z, e_fb], dim=1)  # (B, 96, L)

            # Prepare for gathering
            h_t = h.transpose(1, 2)  # (B, L, 96)

            # Gather Partner Features
            # partner_idx: (B, L) -> Expand to (B, L, 96)
            idx_expanded = partner_idx.unsqueeze(-1).expand(-1, -1, h_t.size(2))
            h_partner = torch.gather(h_t, 1, idx_expanded)

            # Mask unpaired interactions
            mask_expanded = partner_mask.unsqueeze(-1)
            h_partner = h_partner * mask_expanded

            # Concatenate Self and Partner vectors
            rnn_in = torch.cat([h_t, h_partner], dim=2)  # (B, L, 192)

            # Global Aggregation via RNN
            rnn_out, _ = self.gru(rnn_in)  # (B, L, 256)

            # Prediction Head
            logits = self.head(rnn_out)  # (B, L, 5)

            # Transpose to (B, 5, L) for next iteration
            y_next = logits.transpose(1, 2)
            y_preds.append(y_next)
            y_curr = y_next

        return y_preds


# ==================================================================================
# LOSS & UTILS
# ==================================================================================


def mcrmse_loss(pred, target, scored_indices):
    """
    Calculates MCRMSE for specific columns.
    pred: (B, L, 5)
    target: (B, L, 5)
    """
    # Select scored columns
    pred_scored = pred[:, :, scored_indices]
    target_scored = target[:, :, scored_indices]

    # MSE over Batch and Length
    mse = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

    # RMSE per column
    rmse = torch.sqrt(mse)

    # Mean of RMSEs
    return torch.mean(rmse)


def criterion(preds_list, targets):
    """
    Weighted loss over recycling iterations.
    Only calculates loss on the first 68 positions (Config.SCORED_LENGTH).
    """
    loss = 0
    weights = [0.5, 1.0]  # Weight earlier iterations less

    # Targets are (B, L, 5). We only care about first 68.
    t_sliced = targets[:, : Config.SCORED_LENGTH, :]

    for i, pred in enumerate(preds_list):
        # pred is (B, 5, L) -> Transpose to (B, L, 5)
        p = pred.transpose(1, 2)
        p_sliced = p[:, : Config.SCORED_LENGTH, :]

        l = mcrmse_loss(p_sliced, t_sliced, Config.SCORED_COLS_INDICES)
        loss += weights[i] * l

    return loss


# ==================================================================================
# TRAINING & EVALUATION
# ==================================================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, p_idx, p_mask, y in loader:
        x, p_idx, p_mask, y = (
            x.to(device),
            p_idx.to(device),
            p_mask.to(device),
            y.to(device),
        )

        optimizer.zero_grad()
        preds_list = model(x, p_idx, p_mask)
        loss = criterion(preds_list, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, p_idx, p_mask, y in loader:
            x, p_idx, p_mask, y = (
                x.to(device),
                p_idx.to(device),
                p_mask.to(device),
                y.to(device),
            )

            preds_list = model(x, p_idx, p_mask)
            # Validation metric based on final prediction
            final_pred = preds_list[-1]  # (B, 5, L)

            # Transpose and slice
            p = final_pred.transpose(1, 2)[:, : Config.SCORED_LENGTH, :]
            t = y[:, : Config.SCORED_LENGTH, :]

            loss = mcrmse_loss(p, t, Config.SCORED_COLS_INDICES)
            total_loss += loss.item()

    return total_loss / len(loader)


def generate_submission(model, device):
    print("Generating submission...")

    # Load Test Data
    test_dataset = load_data("test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Test Metadata for IDs
    df_test = pd.read_csv(Config.TEST_METADATA)
    ids = df_test["id"].values

    model.eval()
    preds_all = []

    with torch.no_grad():
        for x, p_idx, p_mask, _ in test_loader:
            x, p_idx, p_mask = x.to(device), p_idx.to(device), p_mask.to(device)

            preds_list = model(x, p_idx, p_mask)
            final_pred = preds_list[-1]  # (B, 5, L)

            # Transpose to (B, L, 5)
            final_pred = final_pred.transpose(1, 2)
            preds_all.append(final_pred.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)  # (N, 107, 5)

    # Flatten for submission
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        # Get predictions for this sample
        sample_preds = preds_all[i]  # (107, 5)

        for j in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{j}"
            row_vals = sample_preds[j]

            # Ensure values are not negative (physical constraint)
            # Though some ground truth is negative due to normalization,
            # clipping at 0 is often a safe baseline or we leave raw.
            # We will leave raw as per standard practice unless specified.

            row_dict = {"id_seqpos": row_id}
            for k, col in enumerate(cols):
                row_dict[col] = float(row_vals[k])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_and_submission():
    # Set Seeds
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_dataset = load_data("train", load_cached_data=True, debug=Config.DEBUG)
    val_dataset = load_data("val", load_cached_data=True, debug=Config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Initialize Model
    model = RCRDN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for submission
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    generate_submission(model, device)
