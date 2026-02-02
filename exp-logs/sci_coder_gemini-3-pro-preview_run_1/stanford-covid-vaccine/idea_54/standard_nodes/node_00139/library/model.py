import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.data import load_data, RNADataset
from library.utils import format_submission

# =========================================================================================
# MODEL ARCHITECTURE
# =========================================================================================


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes scalar distances using signed sinusoidal embeddings.
    Preserves the sign information by encoding the value directly into sin/cos functions.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Use a fixed scale for frequencies, similar to Transformer PE but for distance
        # inv_freq shape: (d_model // 2,)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: (B, L) containing signed distances
        # sin_inp shape: (B, L, d_model/2)
        sin_inp = x.unsqueeze(-1) * self.inv_freq

        # Concatenate sin and cos to get (B, L, d_model)
        # sin(-x) = -sin(x), cos(-x) = cos(x) -> Sign is preserved in the sine component
        pos_emb = torch.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
        return pos_emb


class ResBiGRUBlock(nn.Module):
    """
    Stabilized Residual BiGRU Block with Pre-LayerNorm.
    Cite solution_lesson_node_00135: Superiority of Pre-LayerNorm Residual Blocks.
    Cite solution_lesson_node_00138: Efficacy of Pure Recurrent Backbones.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm Residual connection
        res = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        return res + out


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, tensors):
        # tensors: List of (B, L, C)
        # Stack: (N, B, L, C)
        stacked = torch.stack(tensors, dim=0)

        # Softmax over layers
        w = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)

        # Weighted sum
        weighted_sum = (stacked * w).sum(dim=0)
        return weighted_sum


class RNAModel(nn.Module):
    """
    Stabilized Wide-Stream Residual BiGRU Model.
    """

    def __init__(self, config):
        super().__init__()

        # Embeddings
        self.seq_embed = nn.Embedding(4, config.SEQ_EMBED_DIM)
        self.loop_embed = nn.Embedding(7, config.LOOP_EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.DIST_EMBED_DIM)

        input_dim = config.SEQ_EMBED_DIM + config.LOOP_EMBED_DIM + config.DIST_EMBED_DIM

        # Stem: BiGRU projecting to HIDDEN_DIM
        # No dropout after stem (Cite solution_lesson_node_00109)
        self.stem = nn.GRU(
            input_size=input_dim,
            hidden_size=config.HIDDEN_DIM // 2,
            batch_first=True,
            bidirectional=True,
        )

        # Backbone: 6 Residual BiGRU Blocks
        self.blocks = nn.ModuleList(
            [
                ResBiGRUBlock(config.HIDDEN_DIM, dropout=config.DROPOUT)
                for _ in range(config.NUM_BLOCKS)
            ]
        )

        # Aggregation: Mixture of Stem + 6 Blocks
        self.mixture = ScalarMixture(config.NUM_BLOCKS + 1)

        # Head: Project to 3 targets
        self.head = nn.Linear(config.HIDDEN_DIM, 3)

    def forward(self, sequence, loop, distance):
        # Embed inputs
        emb_seq = self.seq_embed(sequence)
        emb_loop = self.loop_embed(loop)
        emb_dist = self.dist_embed(distance)

        # Concatenate: (B, L, 256)
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)

        # Stem
        x, _ = self.stem(x)  # (B, L, 512)

        # Collect outputs for mixture (Stem is first)
        outputs = [x]

        # Backbone
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregation
        agg_out = self.mixture(outputs)  # (B, L, 512)

        # Head
        logits = self.head(agg_out)  # (B, L, 3)

        return logits


# =========================================================================================
# TRAINING LOGIC
# =========================================================================================


def train_epoch(model, loader, optimizer, criterion, device, scheduler, clip_norm):
    model.train()
    running_loss = 0.0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["distance"].to(device)
        target = batch["target"].to(device)  # (B, 68, 3)

        optimizer.zero_grad()

        # Forward
        pred = model(seq, loop, dist)  # (B, 107, 3)

        # Masked Loss: Only first 68 positions are scored
        pred_scored = pred[:, : Config.SEQ_SCORED, :]

        loss = criterion(pred_scored, target)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * seq.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)
            target = batch["target"].to(device)

            pred = model(seq, loop, dist)
            pred_scored = pred[:, : Config.SEQ_SCORED, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # MCRMSE Calculation: Mean of Column-wise RMSE
    # (N, 68, 3) -> Mean over (0, 1) -> (3,) -> Sqrt -> Mean
    mse_per_col = np.mean((all_targets - all_preds) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)
    mcrmse = np.mean(rmse_per_col)

    return mcrmse


def run_pipeline():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Load Data
    print("Loading data...")
    train_ids, train_seq, train_loop, train_dist, train_tgt = load_data("train")
    val_ids, val_seq, val_loop, val_dist, val_tgt = load_data("val")

    train_dataset = RNADataset(train_seq, train_loop, train_dist, train_tgt)
    val_dataset = RNADataset(val_seq, val_loop, val_dist, val_tgt)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model
    model = RNAModel(Config).to(device)

    # Optimizer: AdamW with Low Weight Decay (1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss: MSE
    criterion = nn.MSELoss()

    # Training Loop
    best_mcrmse = float("inf")
    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler,
            Config.CLIP_NORM,
        )
        val_mcrmse = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), save_path)

    print(f"Best Val MCRMSE: {best_mcrmse:.10f}")

    # Inference
    print("Generating submission...")
    test_ids, test_seq, test_loop, test_dist = load_data("test")
    test_dataset = RNADataset(test_seq, test_loop, test_dist, None)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["distance"].to(device)

            pred = model(seq, loop, dist)  # (B, 107, 3)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Format and Save
    format_submission(test_ids, all_preds, Config.SUBMISSION_PATH)


# Execute Pipeline
run_pipeline()
