import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import mcrmse_loss, get_scored_metrics, set_seed
from library.dataset import get_loader


class ZeroMaskedInteractionModule(nn.Module):
    """
    Implements the Zero-Masked Non-Linear Channel-Gated structural interaction.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message transformation: Non-linear projection of the neighbor's state
        self.msg_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

        # Gating mechanism: Channel-wise control based on self and neighbor
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adjacency, mask):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim) - Current hidden states
            adjacency: (Batch, Seq_Len) - Indices of paired bases
            mask: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired
        """
        # 1. Gather neighbor states
        # Expand adjacency to match hidden_dim for gathering
        # adjacency is (B, L), x is (B, L, D)
        # We want to gather along dim 1
        adj_expanded = adjacency.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        x_neighbor = torch.gather(x, 1, adj_expanded)

        # 2. Zero-Masking (Critical)
        # Force gathered vectors of unpaired bases to zero.
        # mask is (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1)
        x_neighbor = x_neighbor * mask_expanded

        # 3. Non-Linear Message
        m = self.msg_proj(x_neighbor)

        # 4. Channel-Wise Gating
        # Concatenate self (x) and neighbor (x_neighbor)
        concat = torch.cat([x, x_neighbor], dim=-1)
        g = self.gate_proj(concat)

        # 5. Injection & Stabilization
        update = g * m
        out = self.norm(x + update)

        return out


class RNAModel(nn.Module):
    """
    Deep Iterative Structural-Refinement Model with Zero-Masked Gating.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.input_dim = config.INPUT_CHANNELS
        self.stem_filters = config.STEM_FILTERS
        self.hidden_dim = config.HIDDEN_DIM
        # BiGRU outputs hidden_dim * 2
        self.gru_output_dim = self.hidden_dim * 2

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim,
                self.stem_filters,
                kernel_size=config.STEM_KERNEL_SIZE,
                padding=config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone (3 Blocks)
        # Block 1: BiGRU + Interaction
        self.gru1 = nn.GRU(
            self.stem_filters, self.hidden_dim, batch_first=True, bidirectional=True
        )
        self.inter1 = ZeroMaskedInteractionModule(self.gru_output_dim)

        # Block 2: BiGRU + Interaction
        self.gru2 = nn.GRU(
            self.gru_output_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )
        self.inter2 = ZeroMaskedInteractionModule(self.gru_output_dim)

        # Block 3: BiGRU only (Finalizing sequential features)
        self.gru3 = nn.GRU(
            self.gru_output_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, config.NUM_TARGETS)

    def forward(self, x, adjacency, mask):
        # x: (B, L, 14)

        # Permute for Conv1d: (B, 14, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back: (B, L, 256)
        x = x.permute(0, 2, 1)

        # Block 1
        x, _ = self.gru1(x)
        x = self.inter1(x, adjacency, mask)
        x = self.dropout(x)

        # Block 2
        x, _ = self.gru2(x)
        x = self.inter2(x, adjacency, mask)
        x = self.dropout(x)

        # Block 3
        x, _ = self.gru3(x)
        x = self.dropout(x)

        # Head
        out = self.head(x)  # (B, L, 5)

        return out


def train_model():
    """
    Executes the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)

    # Initialize Model
    model = RNAModel(Config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    best_score = float("inf")
    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            inputs = batch["sequence"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            preds = model(inputs, adj, mask)

            # Slice predictions to match scored length (68)
            preds_sliced = preds[:, : Config.SEQ_SCORED, :]

            loss = mcrmse_loss(targets, preds_sliced)
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
                inputs = batch["sequence"].to(device)
                adj = batch["adjacency"].to(device)
                mask = batch["mask"].to(device)
                targets = batch["target"].to(device)

                preds = model(inputs, adj, mask)
                preds_sliced = preds[:, : Config.SEQ_SCORED, :]

                val_preds_list.append(preds_sliced.cpu())
                val_targets_list.append(targets.cpu())

        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Metrics
        val_loss = mcrmse_loss(val_targets, val_preds).item()
        val_score = get_scored_metrics(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Score: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Val Score: {best_score:.6f}")


def predict():
    """
    Runs inference on the test set and generates the submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    test_loader = get_loader("test", shuffle=False)

    # Load Model
    model = RNAModel(Config).to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: No model checkpoint found. Using initialized weights.")

    model.eval()
    results = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["sequence"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["id"]

            preds = model(inputs, adj, mask)  # (B, 107, 5)
            preds = preds.cpu().numpy()

            # Format predictions
            for i, sample_id in enumerate(ids):
                sample_preds = preds[i]  # (107, 5)

                for seqpos in range(Config.SEQ_LEN):
                    row_id = f"{sample_id}_{seqpos}"
                    row_preds = sample_preds[seqpos]

                    # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    results.append([row_id] + row_preds.tolist())

    # Save Submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = pd.DataFrame(results, columns=cols)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
