import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_loader

# ==================================================================================
# Model Components
# ==================================================================================


class SinusoidalEncoding(nn.Module):
    """
    Generates fixed sinusoidal encodings for signed scalar distances.
    """

    def __init__(self, dim, max_len=500):
        super().__init__()
        self.dim = dim
        # Precompute denominators: 10000^(2i/dim)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        Args:
            x: (Batch, Seq_Len) containing signed distances (float).
        Returns:
            (Batch, Seq_Len, Dim)
        """
        # Unsqueeze for broadcasting: (B, L, 1) * (D/2)
        x_expanded = x.unsqueeze(-1)
        phase = x_expanded * self.div_term

        # Sin and Cos
        pe = torch.zeros(x.shape[0], x.shape[1], self.dim, device=x.device)
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        return pe


class WideBiGRUBlock(nn.Module):
    """
    Wide-Stream Residual Block with Pre-LayerNorm and BiGRU.
    Maintains full residual stream width (W=384).
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # BiGRU doubles the hidden size, so we use hidden_dim // 2
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return res + x


class RNAModel(nn.Module):
    """
    Scalar-Aggregated Wide-Stream Residual BiGRU.
    Removed Structure-Biased Attention (Cite solution_lesson_node_00106).
    """

    def __init__(self):
        super().__init__()

        # Proportional Feature Embeddings
        self.seq_emb = nn.Embedding(Config.vocab_size, Config.embedding_dim)
        self.loop_emb = nn.Embedding(Config.loop_vocab_size, Config.loop_dim)
        self.pair_enc = SinusoidalEncoding(Config.pair_dim)

        # Stem: Projects concatenated inputs to residual stream width
        self.stem_gru = nn.GRU(
            Config.input_dim,
            Config.hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

        # Backbone: 6 Wide-Stream Residual Blocks
        self.blocks = nn.ModuleList(
            [
                WideBiGRUBlock(Config.hidden_dim, Config.dropout)
                for _ in range(Config.num_layers)
            ]
        )

        # Aggregation: Scalar Mixture of Stem + 6 Blocks
        self.mix_weights = nn.Parameter(torch.zeros(Config.num_layers + 1))

        # Output Head
        self.head = nn.Linear(Config.hidden_dim, Config.num_targets)

    def forward(self, sequence, loop_type, pair_offset):
        # 1. Embeddings
        emb_seq = self.seq_emb(sequence)
        emb_loop = self.loop_emb(loop_type)
        emb_pair = self.pair_enc(pair_offset)

        # Concatenate: (B, L, Input_Dim)
        x = torch.cat([emb_seq, emb_loop, emb_pair], dim=-1)

        # 2. Stem
        x, _ = self.stem_gru(x)

        # 3. Backbone (Collect states)
        states = [x]
        for block in self.blocks:
            x = block(x)
            states.append(x)

        # 4. Scalar Mixture Aggregation
        # Stack: (B, L, H, Layers+1)
        stacked = torch.stack(states, dim=-1)
        weights = F.softmax(self.mix_weights, dim=0)
        x_agg = torch.sum(stacked * weights, dim=-1)

        # 5. Output Head (Directly from aggregated features)
        out = self.head(x_agg)
        return out


# ==================================================================================
# Training and Inference Logic
# ==================================================================================


def train_step(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop_type"].to(device)
        pair = batch["pair_offset"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        preds = model(seq, loop, pair)

        # Masked MSE: Only calculate loss for the first 68 positions
        preds_scored = preds[:, : Config.pred_len, :]
        targets_scored = targets[:, : Config.pred_len, :]

        loss = criterion(preds_scored, targets_scored)
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            pair = batch["pair_offset"].to(device)
            targets = batch["targets"].to(device)

            preds = model(seq, loop, pair)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # MCRMSE handles slicing to pred_len internally
    score = mcrmse_loss(all_targets, all_preds).item()
    return score


def inference(model, loader, device):
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            pair = batch["pair_offset"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, pair)

            ids_list.extend(ids)
            preds_list.append(preds.cpu().numpy())

    return ids_list, np.concatenate(preds_list, axis=0)


def run_pipeline():
    set_seed()
    device = Config.device
    print(f"Device: {device}")

    # 1. Data Loading
    print("Loading data...")
    train_loader = get_loader("train", shuffle=True)
    val_loader = get_loader("val", shuffle=False)
    test_loader = get_loader("test", shuffle=False)

    # 2. Model Initialization
    model = RNAModel().to(device)
    print("Model initialized.")

    # 3. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )
    criterion = nn.MSELoss()

    # 4. Training Loop
    best_score = float("inf")
    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        # Step scheduler per epoch
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    ids, preds = inference(model, test_loader, device)

    # 6. Submission Generation
    print("Generating submission...")
    submission_data = []
    # preds shape: (N_samples, 107, 3)
    # Output columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Submission requires: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 3)

        for j in range(Config.seq_len):
            row_id = f"{sample_id}_{j}"

            # Extract predicted values
            reactivity = float(sample_preds[j, 0])
            deg_Mg_pH10 = float(sample_preds[j, 1])
            deg_Mg_50C = float(sample_preds[j, 2])

            # Fill unscored columns with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    df_sub = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


if __name__ == "__main__":
    # Execute pipeline
    run_pipeline()
