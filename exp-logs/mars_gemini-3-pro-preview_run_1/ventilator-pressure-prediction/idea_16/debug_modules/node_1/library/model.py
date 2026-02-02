import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

# Import configuration and library functions
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_BLOCKS,
    EXPANSION_FACTOR,
    KERNEL_SIZES,
    DROPOUT,
    AUX_WEIGHT,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PCT_START,
    GRAD_CLIP,
    FEATURE_NAMES,
    SEED,
    DEVICE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    SAMPLE_SUBMISSION_PATH,
)
from library.utils import seed_everything
from library.dataset import get_ventilator_datasets

# ==================================================================================
# MODEL ARCHITECTURE
# ==================================================================================


class MultiScaleStem(nn.Module):
    """
    Inception-style Multi-Scale 1D Convolutional Stem.
    Captures both fine-grained signal noise and smoothed trend derivatives.
    """

    def __init__(self, input_dim, hidden_dim, kernel_sizes):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        # Project concatenated outputs to hidden_dim
        concat_dim = hidden_dim * len(kernel_sizes)
        self.proj = nn.Conv1d(concat_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq_Len, Features) -> (Batch, Features, Seq_Len)
        x = x.transpose(1, 2)
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)
        out = self.proj(out)
        out = self.act(out)
        # Return to (Batch, Seq_Len, Hidden)
        return out.transpose(1, 2)


class CompositeBlock(nn.Module):
    """
    Wide-State Weight-Normalized Physics-Injected Composite Block.
    Maintains high-magnitude gradient highway and high-capacity channel mixing.
    """

    def __init__(self, input_dim, hidden_dim, physics_dim, expansion_factor, dropout):
        super().__init__()

        # 1. Wide-State Temporal Mixing (Bi-LSTM)
        # Input includes Deep Context Injection (Previous State + Physics Features)
        self.lstm = nn.LSTM(
            input_size=input_dim + physics_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Bi-LSTM output dimension is 2 * hidden_dim
        lstm_out_dim = 2 * hidden_dim

        # 2. Projected Residual 1
        # Projects the identity path to match LSTM output dimension if necessary
        if input_dim != lstm_out_dim:
            self.res1_proj = nn.Linear(input_dim, lstm_out_dim)
        else:
            self.res1_proj = nn.Identity()

        # 3. High-Capacity Channel Mixing (Weight-Norm FFN)
        ffn_inner_dim = lstm_out_dim * expansion_factor
        self.ffn = nn.Sequential(
            weight_norm(nn.Linear(lstm_out_dim, ffn_inner_dim)),
            nn.GELU(),
            weight_norm(nn.Linear(ffn_inner_dim, lstm_out_dim)),
        )

        # 4. Residual 2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, physics_feats):
        # Deep Context Injection: Concatenate input with physics features
        lstm_input = torch.cat([x, physics_feats], dim=-1)

        # LSTM Processing (No compression of output)
        lstm_out, _ = self.lstm(lstm_input)

        # Residual 1
        res1 = self.res1_proj(x) + lstm_out

        # Channel Mixing
        ffn_out = self.ffn(res1)

        # Residual 2
        out = res1 + self.dropout(ffn_out)

        return out


class VentilatorModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Identify indices for Physics Features Injection
        # We need R, C, u_in * R, vol / C
        target_feats = ["R", "C", "u_in_R", "vol_C"]
        self.phys_indices = []
        for feat in target_feats:
            if feat in FEATURE_NAMES:
                self.phys_indices.append(FEATURE_NAMES.index(feat))
            else:
                raise ValueError(f"Required feature {feat} not found in FEATURE_NAMES")

        physics_dim = len(self.phys_indices)

        # Stem
        self.stem = MultiScaleStem(INPUT_DIM, HIDDEN_DIM, KERNEL_SIZES)

        # Backbone
        self.blocks = nn.ModuleList()
        current_dim = HIDDEN_DIM

        for i in range(NUM_BLOCKS):
            block = CompositeBlock(
                input_dim=current_dim,
                hidden_dim=HIDDEN_DIM,
                physics_dim=physics_dim,
                expansion_factor=EXPANSION_FACTOR,
                dropout=DROPOUT,
            )
            self.blocks.append(block)
            # After the first block, the dimension expands to 2 * HIDDEN_DIM due to Bi-LSTM
            current_dim = 2 * HIDDEN_DIM

        # Heads
        self.aux_head = nn.Linear(2 * HIDDEN_DIM, 1)
        self.head = nn.Linear(2 * HIDDEN_DIM, 1)

    def forward(self, x):
        # Extract Physics Features for Injection
        phys_feats = x[:, :, self.phys_indices]

        # Stem
        h = self.stem(x)

        aux_pred = None

        for i, block in enumerate(self.blocks):
            h = block(h, phys_feats)

            # Deep Supervision: Auxiliary Head after Block 2 (index 1)
            if i == 1:
                aux_pred = self.aux_head(h)

        # Final Prediction
        final_pred = self.head(h)

        return final_pred, aux_pred


# ==================================================================================
# TRAINING UTILITIES
# ==================================================================================


def masked_mae_loss(pred, target, u_out):
    """
    Computes L1 loss only for the inspiratory phase (u_out == 0).
    """
    # Mask: 1 where u_out == 0 (inspiration), 0 otherwise
    mask = 1 - u_out
    loss = torch.abs(pred.squeeze(-1) - target) * mask
    # Normalize by the number of valid elements to avoid batch size bias
    return loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        pred, aux_pred = model(x)

        # Composite Loss
        loss_main = masked_mae_loss(pred, y, u_out)
        loss_aux = masked_mae_loss(aux_pred, y, u_out)
        loss = loss_main + AUX_WEIGHT * loss_aux

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_mae = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            pred, _ = model(x)

            # Metric: MAE on inspiratory phase
            mae = masked_mae_loss(pred, y, u_out)
            total_mae += mae.item()

    return total_mae / len(loader)


def predict_test(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            ids = batch["ids"]

            pred, _ = model(x)

            all_preds.append(pred.squeeze(-1).cpu().numpy())
            all_ids.append(ids.numpy())

    # Flatten
    flat_preds = np.concatenate(all_preds).flatten()
    flat_ids = np.concatenate(all_ids).flatten()

    return flat_ids, flat_preds


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def run_task():
    # 1. Setup
    seed_everything(SEED)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    print("Preparing DataLoaders...")
    train_ds, val_ds, test_ds = get_ventilator_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = VentilatorModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=PCT_START,
    )

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_val_mae = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        val_mae = validate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (MAE: {best_val_mae:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # 5. Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    ids, preds = predict_test(model, test_loader, DEVICE)

    submission_df = pd.DataFrame({"id": ids, "pressure": preds})

    # Sort by ID just in case
    submission_df.sort_values("id", inplace=True)

    submission_df.to_csv(SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_FILE_PATH}")


if __name__ == "__main__":
    run_task()
