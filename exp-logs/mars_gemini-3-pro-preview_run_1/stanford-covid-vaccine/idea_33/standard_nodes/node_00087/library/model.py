import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_dataset
from library.utils import seed_everything, mcrmse_loss, format_submission


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed scalar distances using fixed sinusoidal functions.
    Preserves the sign to distinguish upstream/downstream dependencies.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Inverse frequencies for the positional encoding
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, x):
        # x: [batch, seq_len] containing signed distances
        x_expanded = x.unsqueeze(-1)  # [B, L, 1]
        freqs = self.inv_freq.to(x.device)  # [dim/2]
        args = x_expanded * freqs  # [B, L, dim/2]
        # Concatenate sin and cos to form the embedding
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, L, dim]
        return pe


class ResidualBiGRUBlock(nn.Module):
    """
    A single residual block with Pre-LayerNorm and Bidirectional GRU.
    Maintains the full stream width (no bottleneck).
    """

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.gru = nn.GRU(
            config.HIDDEN_DIM,
            config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        # Pre-LayerNorm
        residual = x
        out = self.norm(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        # Residual connection
        return residual + out


class RNAModel(nn.Module):
    """
    Scalar-Aggregated Wide-Stream Residual BiGRU.
    Integrates sequence, loop, and distance embeddings, processes them with a deep
    residual BiGRU backbone, and uses scalar mixture aggregation for the readout.
    Reverts the structural shortcut to a standard MLP readout.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Embeddings
        self.seq_embed = nn.Embedding(4, config.EMBED_DIM)
        self.loop_embed = nn.Embedding(7, config.EMBED_DIM)
        self.dist_embed = SinusoidalPositionalEmbedding(config.EMBED_DIM)

        # Input Projection: Concatenate 3 embeddings -> Hidden Dim
        input_dim = config.EMBED_DIM * 3
        self.input_proj = nn.Linear(input_dim, config.HIDDEN_DIM)

        # 2. Recurrent Stem (Cite solution_lesson_node_00046)
        self.stem = nn.GRU(
            config.HIDDEN_DIM,
            config.HIDDEN_DIM // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone: Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBiGRUBlock(config) for _ in range(config.N_LAYERS)]
        )

        # 4. Aggregation: Scalar Mixture (Cite solution_lesson_node_00049)
        # Weights for Stem + N_LAYERS blocks
        self.agg_weights = nn.Parameter(torch.zeros(config.N_LAYERS + 1))

        # 5. Readout Head
        # Reverted to standard MLP without structural shortcut (Cite solution_lesson_node_00086)
        self.readout = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.N_TARGETS),
        )

    def forward(self, batch):
        seq = batch["sequence"]
        loop = batch["loop_type"]
        dist = batch["distance"]

        # --- Embeddings ---
        emb_seq = self.seq_embed(seq)
        emb_loop = self.loop_embed(loop)
        emb_dist = self.dist_embed(dist)

        # Concatenate and project
        x = torch.cat([emb_seq, emb_loop, emb_dist], dim=-1)
        x = self.input_proj(x)

        # --- Stem ---
        x, _ = self.stem(x)
        layer_outputs = [x]

        # --- Residual Blocks ---
        current_x = x
        for block in self.blocks:
            current_x = block(current_x)
            layer_outputs.append(current_x)

        # --- Aggregation ---
        # Compute softmax weights
        weights = F.softmax(self.agg_weights, dim=0)

        # Iterative accumulation (Cite solution_lesson_node_00063)
        # Avoids creating a large [L, B, Seq, H] tensor
        agg_state = torch.zeros_like(layer_outputs[0])
        for i, output in enumerate(layer_outputs):
            agg_state += output * weights[i]

        # --- Readout ---
        logits = self.readout(agg_state)  # [B, L, 3]

        return logits


def train_one_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        preds = model(batch)
        targets = batch["targets"]

        # Calculate loss only on scored positions
        preds_scored = preds[:, : config.SEQ_SCORED, :]
        targets_scored = targets[:, : config.SEQ_SCORED, :]

        # Standard MSE Loss
        loss = F.mse_loss(preds_scored, targets_scored)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            targets = batch["targets"]

            # Slice to scored region
            preds_scored = preds[:, : config.SEQ_SCORED, :]
            targets_scored = targets[:, : config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using utility function
    score = mcrmse_loss(all_targets, all_preds).item()
    return score


def run_training(config=Config):
    """
    Executes the training pipeline.
    """
    seed_everything(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print("Loading datasets...")
    train_ds = get_dataset("train", config)
    val_ds = get_dataset("val", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = RNAModel(config).to(config.DEVICE)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, config.DEVICE, config
        )
        val_mcrmse = validate(model, val_loader, config.DEVICE, config)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_mcrmse:.10f}"
        )

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")
    return best_model_path


def run_inference(config=Config):
    """
    Executes the inference pipeline and generates submission.csv.
    """
    print("Generating submission...")
    seed_everything(config.SEED)

    # Load Test Data
    test_ds = get_dataset("test", config)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = RNAModel(config).to(config.DEVICE)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config.DEVICE)

            ids = batch["ids"]
            preds = model(batch)  # [B, 107, 3]

            all_preds.append(preds.cpu())
            all_ids.extend(ids)

    # Concatenate predictions: [N_test, 107, 3]
    all_preds = torch.cat(all_preds, dim=0)

    # Format submission
    df_sub = format_submission(all_ids, all_preds, seq_length=config.SEQ_LENGTH)

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
