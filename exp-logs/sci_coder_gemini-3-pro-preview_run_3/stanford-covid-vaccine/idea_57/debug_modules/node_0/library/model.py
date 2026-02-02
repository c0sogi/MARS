import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, mcrmse_numpy
from library.data import get_dataloaders

# ==========================================
# Model Components
# ==========================================


class DecoupledInteractionModule(nn.Module):
    """
    Stabilized Decoupled Interaction Module.

    Features:
    - Point-to-Point Gathering via adjacency indices.
    - Input Zero-Masking for unpaired bases (avoids self-loops).
    - Bias-Refined Message: Unpaired bases generate a bias-driven 'loop embedding'.
    - Stabilized MLP Gate: Internal LayerNorm, no logit normalization.
    - Post-Normalization on the residual output.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Message projection: h_j -> m_ij
        # Bias is crucial here for unpaired bases where input is 0
        self.W_msg = nn.Linear(d_model, d_model, bias=True)

        # Gating Network
        # Input: Concatenation of [h_i, h_j] -> 2 * d_model
        self.W_g1 = nn.Linear(2 * d_model, d_model, bias=True)
        self.ln_gate = nn.LayerNorm(d_model)  # Internal Normalization
        self.W_g2 = nn.Linear(d_model, d_model, bias=True)

        # Final Post-Normalization
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, pair_index, pair_mask):
        """
        Args:
            x: (Batch, SeqLen, Dim) - Hidden states from GRU
            pair_index: (Batch, SeqLen) - Indices of paired bases
            pair_mask: (Batch, SeqLen) - 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, dim = x.shape

        # 1. Gather h_j
        # Flatten indices for efficient gathering
        # Create offset for batch dimension
        batch_offset = (torch.arange(batch_size, device=x.device) * seq_len).unsqueeze(
            1
        )
        flat_idx = (pair_index + batch_offset).view(-1)

        # Gather
        h_flat = x.reshape(-1, dim)
        h_j = h_flat[flat_idx].view(batch_size, seq_len, dim)

        # 2. Input Zero-Masking
        # If unpaired (mask=0), force h_j to 0.
        # This allows W_msg(0) + b = b (Bias-Refined Loop Embedding)
        h_j = h_j * pair_mask.unsqueeze(-1)

        # 3. Compute Message
        # m_ij = GELU(W_msg(h_j) + b)
        m_ij = F.gelu(self.W_msg(h_j))

        # 4. Stabilized Gate
        # z_raw = W_g1([h_i, h_j])
        cat_input = torch.cat([x, h_j], dim=-1)
        z_raw = self.W_g1(cat_input)

        # Internal Normalization (prevents saturation)
        z_norm = self.ln_gate(z_raw)
        z_act = F.gelu(z_norm)

        # Logits and Sigmoid (No logit norm)
        logits = self.W_g2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection & Post-Norm
        h_res = x + g_ij * m_ij
        h_out = self.ln_out(h_res)

        return h_out


class SDBR_BiGRU(nn.Module):
    """
    Stabilized Decoupled Bias-Refined BiGRU.

    Architecture:
    1. Conv1D Stem
    2. 3-Layer Backbone:
       - Layer 1: BiGRU -> Interaction
       - Layer 2: BiGRU -> Interaction
       - Layer 3: BiGRU (No Interaction)
    3. Linear Head
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.input_dim = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM  # 384

        # BiGRU hidden size per direction to sum up to hidden_dim
        # 384 // 2 = 192 per direction -> Output 384
        self.gru_hidden = self.hidden_dim // 2

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.conv_filters,
                kernel_size=Config.CONV_KERNEL_SIZE,
                padding=1,
            ),
            nn.GELU(),
        )

        # 2. Backbone
        # Block 1
        self.gru1 = nn.GRU(
            self.conv_filters, self.gru_hidden, batch_first=True, bidirectional=True
        )
        self.inter1 = DecoupledInteractionModule(self.hidden_dim)

        # Block 2
        self.gru2 = nn.GRU(
            self.hidden_dim, self.gru_hidden, batch_first=True, bidirectional=True
        )
        self.inter2 = DecoupledInteractionModule(self.hidden_dim)

        # Block 3 (Final, no interaction)
        self.gru3 = nn.GRU(
            self.hidden_dim, self.gru_hidden, batch_first=True, bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Head
        self.head = nn.Linear(self.hidden_dim, Config.NUM_TARGETS)

    def forward(self, x, pair_index, pair_mask):
        # x: (B, L, 14)
        # Permute for Conv1d: (B, 14, L)
        x = x.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back for GRU: (B, L, C)
        x = x.permute(0, 2, 1)

        # Block 1
        x, _ = self.gru1(x)  # Out: (B, L, 384)
        x = self.inter1(x, pair_index, pair_mask)
        x = self.dropout(x)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, pair_index, pair_mask)
        x = self.dropout(x)

        # Block 3
        x, _ = self.gru3(x)
        x = self.dropout(x)

        # Head
        out = self.head(x)  # (B, L, 5)

        return out


# ==========================================
# Training & Evaluation Logic
# ==========================================


def loss_fn(outputs, targets):
    """
    MCRMSE Loss for training.
    Computes RMSE per column, then averages.
    """
    # outputs: (B, L, 5)
    # targets: (B, L, 5)
    mse = F.mse_loss(outputs, targets, reduction="none")  # (B, L, 5)
    # Mean over batch and sequence
    mse_col = torch.mean(mse, dim=(0, 1))
    rmse_col = torch.sqrt(mse_col)
    return torch.mean(rmse_col)


def train_model(epochs=Config.EPOCHS, debug=False):
    seed_everything(Config.SEED)
    Config.setup_directories()

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Loading dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    if debug:
        print("Debug mode: reducing epochs to 2")
        epochs = 2

    # Init Model
    model = SDBR_BiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            pair_idx = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 5)

            optimizer.zero_grad()

            # Forward
            preds = model(inputs, pair_idx, pair_mask)  # (B, 107, 5)

            # Slice predictions to match targets (68)
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            # Loss (Multi-Task on all 5 columns)
            loss = loss_fn(preds_sliced, targets)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            train_loss_accum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["inputs"].to(device)
                pair_idx = batch["pair_index"].to(device)
                pair_mask = batch["pair_mask"].to(device)
                targets = batch["targets"].numpy()  # Keep as numpy for metric

                preds = model(inputs, pair_idx, pair_mask)
                preds = preds.cpu().numpy()

                val_preds_list.append(preds)
                val_targets_list.append(targets)

        # Concatenate
        val_preds_all = np.concatenate(val_preds_list, axis=0)  # (N, 107, 5)
        val_targets_all = np.concatenate(val_targets_list, axis=0)  # (N, 68, 5)

        # Calculate Metric (Slices internally to 68 and selects 3 scored columns)
        val_mcrmse = mcrmse_numpy(val_targets_all, val_preds_all)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! ({val_mcrmse:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")


def predict_and_submit():
    """
    Generates predictions for the test set and creates the submission file.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()

    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = SDBR_BiGRU().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_idx = batch["pair_index"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]

            preds = model(inputs, pair_idx, pair_mask)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            all_preds.append(preds)
            all_ids.extend(ids)

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Prepare Submission DataFrame
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # We need to flatten the predictions

    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
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

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
