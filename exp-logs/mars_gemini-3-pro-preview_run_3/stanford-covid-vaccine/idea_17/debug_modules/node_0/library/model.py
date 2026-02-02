import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders

# ==========================================
# 1. Structural Interaction Module
# ==========================================


class StructuralInteractionModule(nn.Module):
    """
    Gated Structural Interaction Module.

    Uses the secondary structure (pair indices) to gather hidden states from paired bases.
    A gating mechanism determines how much of the paired state's information should be
    integrated into the current state, effectively validating the provided structure
    against the learned sequence context.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(StructuralInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Gate network: Determines trust/relevance of the connection
        # Input: Concatenation of current state (h_i) and paired state (h_j)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )

        # Projection network: Transforms the paired state before injection
        self.proj_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input hidden states. Shape (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases. Shape (Batch, Seq_Len).
                                         -1 indicates unpaired.
        """
        batch_size, seq_len, dim = x.shape

        # 1. Create a mask for valid pairs (where index != -1)
        # pair_indices is (B, L)
        valid_mask = pair_indices != -1  # (B, L)

        # 2. Prepare indices for gathering
        # Replace -1 with 0 to allow gather to work (we will mask the result later)
        safe_indices = pair_indices.clone()
        safe_indices[~valid_mask] = 0

        # Expand indices for gathering across the feature dimension
        # Shape: (B, L, Dim)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, dim)

        # 3. Gather paired states h_j
        # x is (B, L, Dim)
        paired_x = torch.gather(x, 1, gather_indices)

        # 4. Mask out invalid gathers (where original index was -1)
        paired_x = paired_x * valid_mask.unsqueeze(-1).float()

        # 5. Compute Gate
        # Concatenate h_i and h_j
        concat = torch.cat([x, paired_x], dim=-1)  # (B, L, 2*Dim)
        gate = self.gate_net(concat)  # (B, L, Dim)

        # 6. Compute Update
        proj = self.proj_net(paired_x)  # (B, L, Dim)

        # 7. Inject and Residual Connection
        update = gate * proj
        out = x + update

        return self.layer_norm(out)


# ==========================================
# 2. Main Model Architecture
# ==========================================


class DISR_BiGRU(nn.Module):
    """
    Deep Iterative Structural-Refinement BiGRU.

    Structure:
    1. 1D Conv Stem (Local Feature Extraction)
    2. Stacked BiGRU Backbone with Interleaved Structural Interaction Modules
    3. Linear Output Head
    """

    def __init__(self, config: Config):
        super(DISR_BiGRU, self).__init__()

        self.seq_len = config.seq_len
        self.num_targets = config.num_targets

        # --- 1. Convolutional Stem ---
        # Projects sparse one-hot features into dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_channels,
                out_channels=config.conv_filters,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --- 2. Iterative Refinement Backbone ---
        # BiGRU hidden size is config.hidden_dim.
        # Since it's bidirectional, the output dimension is 2 * hidden_dim.
        gru_input_dim = config.conv_filters
        gru_hidden_dim = config.hidden_dim
        gru_output_dim = 2 * gru_hidden_dim

        # Block 1
        self.gru1 = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.interact1 = StructuralInteractionModule(gru_output_dim, config.dropout)

        # Block 2
        self.gru2 = nn.GRU(
            input_size=gru_output_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.interact2 = StructuralInteractionModule(gru_output_dim, config.dropout)

        # Block 3
        self.gru3 = nn.GRU(
            input_size=gru_output_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # --- 3. Output Head ---
        self.head = nn.Linear(gru_output_dim, self.num_targets)

    def forward(self, features, pair_indices):
        """
        Args:
            features (torch.Tensor): (Batch, Seq_Len, Channels)
            pair_indices (torch.Tensor): (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = features.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Block 1
        x, _ = self.gru1(x)
        x = self.interact1(x, pair_indices)

        # Block 2
        x, _ = self.gru2(x)
        x = self.interact2(x, pair_indices)

        # Block 3
        x, _ = self.gru3(x)

        # Head
        out = self.head(x)

        return out


# ==========================================
# 3. Training Logic
# ==========================================


def train_model(config: Config):
    """
    Executes the training pipeline:
    1. Loads data
    2. Initializes model, optimizer, scheduler
    3. Runs training loop with gradient clipping
    4. Runs validation loop with global metric aggregation
    5. Implements early stopping
    6. Saves best model
    """
    set_seed(config.seed)

    # Load DataLoaders
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize Model
    model = DISR_BiGRU(config).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    # Training State
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.device}...")
    print(
        f"Model: DISR-BiGRU | Layers: {config.num_layers} | Hidden: {config.hidden_dim}"
    )

    for epoch in range(config.epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            features = batch["features"].to(config.device)
            pair_indices = batch["pair_indices"].to(config.device)
            targets = batch["targets"].to(config.device)
            mask = batch["mask"].to(config.device)  # (B, L)

            optimizer.zero_grad()

            preds = model(features, pair_indices)  # (B, L, 5)

            # Apply mask to loss calculation
            # We only care about positions where mask == 1 (first 68 bases)
            # Expand mask for targets: (B, L) -> (B, L, 1)
            mask_expanded = mask.unsqueeze(-1)

            # Mask predictions and targets
            preds_masked = preds * mask_expanded
            targets_masked = targets * mask_expanded

            # Calculate Loss (MCRMSE on all 5 columns)
            loss = mcrmse_loss(preds_masked, targets_masked)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase (Global Metric) ---
        model.eval()

        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(config.device)
                pair_indices = batch["pair_indices"].to(config.device)
                targets = batch["targets"].to(config.device)
                mask = batch["mask"].to(config.device)

                preds = model(features, pair_indices)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(mask.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)  # (N, L, 5)
        all_targets = torch.cat(all_targets, dim=0)  # (N, L, 5)
        all_masks = torch.cat(all_masks, dim=0)  # (N, L)

        # Filter for Scored Columns only (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Indices: 0, 1, 3
        scored_indices = [0, 1, 3]

        all_preds_scored = all_preds[:, :, scored_indices]
        all_targets_scored = all_targets[:, :, scored_indices]

        # Apply Mask
        mask_expanded = all_masks.unsqueeze(-1)  # (N, L, 1)
        all_preds_scored = all_preds_scored * mask_expanded
        all_targets_scored = all_targets_scored * mask_expanded

        # Calculate Validation Metric
        val_loss = mcrmse_loss(all_preds_scored, all_targets_scored).item()

        # Update Scheduler
        scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE (Scored): {val_loss:.10f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  [+] Saved best model to {config.model_save_path}")
        else:
            patience_counter += 1
            print(f"  [-] Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_val_loss:.10f}")


# ==========================================
# 4. Submission Logic
# ==========================================


def generate_submission(config: Config):
    """
    Generates the submission file for the test set.
    1. Loads the best trained model.
    2. Predicts on the test set (all 107 positions).
    3. Formats the output into the required CSV format.
    """
    set_seed(config.seed)

    # Load Test Loader
    _, _, test_loader = get_dataloaders(config)

    # Load Model
    model = DISR_BiGRU(config).to(config.device)
    if os.path.exists(config.model_save_path):
        model.load_state_dict(
            torch.load(config.model_save_path, map_location=config.device)
        )
        print(f"Loaded model from {config.model_save_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {config.model_save_path}")

    model.eval()

    # Store predictions
    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(config.device)
            pair_indices = batch["pair_indices"].to(config.device)
            ids = batch["id"]  # tuple of strings

            # Predict
            preds = model(features, pair_indices)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(preds)

    # Concatenate all predictions: (Total_Test_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    # We need to flatten: (Sample * 107) rows
    submission_data = []

    # Target columns in order
    target_cols = config.target_cols

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # (107, 5)

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            # Create a dictionary for the row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
    print(f"Submission shape: {submission_df.shape}")
