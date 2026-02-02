import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library.utils import set_seed, MCRMSELoss, metric_mcrmse
from library.data import get_dataloaders

# =========================================================================
# Structural Interaction Module
# =========================================================================


class GLUDecoupledInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    Implements point-to-point message passing with:
    1. Explicit Zero-Masking for unpaired bases.
    2. Decoupled GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
       Only depends on neighbor state h_j, not h_i.
    3. Wide Stabilized MLP Gate: Controls injection strength.
    4. Post-Normalization: Ensures stability in deep stacks.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(GLUDecoupledInteraction, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Components (Decoupled)
        # Input: h_j (hidden_dim) -> Output: hidden_dim
        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_g = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Wide Stabilized MLP Gate
        # Input: [h_i; h_j] (2 * hidden_dim) -> Output: hidden_dim
        # Projects to full width to avoid bottlenecks
        self.W_in = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.ln_gate = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Post-Normalization
        self.ln_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim) - Node features.
            pair_indices: (Batch, Seq_Len) - Indices of paired partners.
            pair_masks: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired.
        """
        B, L, H = x.shape

        # 1. Gather Neighbor States (h_j)
        # Expand indices to match feature dimension: (B, L, H)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(x, 1, idx)

        # 2. Input Zero-Masking
        # Force h_j to 0 if unpaired.
        # This ensures the GLU bias term learns a "loop embedding".
        mask = pair_masks.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = Content * Gate
        content = self.W_c(h_j)
        gate_signal = torch.sigmoid(self.W_g(h_j))
        m_ij = content * gate_signal

        # 4. Wide Stabilized MLP Gate
        # Calculate gating coefficient g_ij based on both h_i and h_j
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2H)
        z_raw = self.W_in(cat_input)
        z_norm = self.ln_gate(z_raw)  # Internal Normalization
        z_act = self.act(z_norm)
        g_ij = self.sigmoid(self.W_out(z_act))

        # 5. Injection (Residual)
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.ln_out(h_res)

        return self.dropout(h_out)


# =========================================================================
# Main Model Architecture
# =========================================================================


class RNAModel(nn.Module):
    """
    High-Capacity Stabilized GLU-Decoupled BiGRU.

    Architecture:
    1. Conv1d Stem (local feature extraction).
    2. 4-Layer BiGRU Backbone (768 hidden units).
    3. Interleaved Interaction Modules (Layers 0, 1, 2).
    4. Linear Head (5 targets).
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # 1. Convolutional Stem
        self.conv = nn.Conv1d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL,
            padding=Config.CONV_KERNEL // 2,
        )
        self.act = nn.GELU()

        # 2. Backbone & Interactions
        self.backbone = nn.ModuleList()
        self.interactions = nn.ModuleList()

        # Dimensions
        input_size = Config.CONV_FILTERS
        hidden_size = Config.HIDDEN_DIM  # 384 per direction
        total_hidden = hidden_size * 2  # 768 total

        for i in range(Config.NUM_LAYERS):
            # BiGRU Layer
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.backbone.append(rnn)

            # Interaction Module (applied after layers 0, 1, 2)
            # Not applied after the final layer (3)
            if i < Config.NUM_LAYERS - 1:
                self.interactions.append(
                    GLUDecoupledInteraction(total_hidden, Config.DROPOUT)
                )

            # Next layer input is current layer output
            input_size = total_hidden

        # 3. Output Head
        self.head = nn.Linear(total_hidden, Config.NUM_TARGETS)

    def forward(self, x, pair_indices, pair_masks):
        # x: (B, L, 14)

        # Permute for Conv1d: (B, 14, L)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv(x))
        # Permute back: (B, L, 256)
        x = x.permute(0, 2, 1)

        # Pass through Backbone
        for i in range(Config.NUM_LAYERS):
            # RNN
            x, _ = self.backbone[i](x)

            # Interaction
            if i < Config.NUM_LAYERS - 1:
                x = self.interactions[i](x, pair_indices, pair_masks)

        # Head
        out = self.head(x)
        return out


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_model(epochs=Config.EPOCHS, debug=False):
    """
    Executes the training loop with Multi-Task Learning, Gradient Clipping,
    and Cosine Annealing.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = RNAModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function (Multi-Task)
    criterion = MCRMSELoss()

    best_mcrmse = float("inf")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if debug and batch_idx > 5:
                break

            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            preds = model(inputs, pair_indices, pair_masks)
            loss = criterion(preds, targets)

            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if debug and batch_idx > 5:
                    break

                inputs = batch["inputs"].to(device)
                pair_indices = batch["pair_indices"].to(device)
                pair_masks = batch["pair_masks"].to(device)
                targets = batch["targets"].to(device)

                preds = model(inputs, pair_indices, pair_masks)

                val_preds_list.append(preds.cpu())
                val_targets_list.append(targets.cpu())

        # Concatenate for global metric calculation
        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate Metric on scored columns only
        val_mcrmse = metric_mcrmse(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! (MCRMSE: {best_mcrmse:.6f})")

        scheduler.step()

    return best_mcrmse


def generate_submission():
    """
    Loads the best model, generates predictions for the test set,
    and saves the formatted submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = RNAModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: No trained model found. Using initialized weights.")

    model.eval()

    all_ids = []
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]  # List of strings

            preds = model(inputs, pair_indices, pair_masks)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Format Submission
    # Rows: id_seqpos
    # Cols: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        # Prediction for this sample: (107, 5)
        sample_pred = all_preds[i]

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


def run_workflow(epochs=Config.EPOCHS, debug=False):
    """
    Runs the full pipeline: Train -> Inference.
    """
    print("==== Starting Workflow ====")
    train_model(epochs=epochs, debug=debug)
    generate_submission()
    print("==== Workflow Completed ====")
