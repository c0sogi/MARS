import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.layers import LayerNormBiLSTM, SinusoidalPositionalEmbedding
from library.utils import get_device, mcrmse
from library.dataset import get_dataloader


class ScalarMixture(nn.Module):
    """
    Computes a learnable weighted sum of a list of tensors.
    Used to aggregate outputs from the Stem and all Backbone blocks.
    """

    def __init__(self, num_layers):
        super(ScalarMixture, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        # tensors: List of (Batch, Seq, Hidden)
        # Stack them: (Batch, Seq, Hidden, Num_Layers)
        stacked = torch.stack(tensors, dim=-1)

        # Softmax over weights to ensure stability
        norm_weights = torch.softmax(self.weights, dim=0)

        # Weighted sum along the last dimension
        # Broadcast weights: (1, 1, 1, Num_Layers)
        weighted_sum = torch.sum(stacked * norm_weights.view(1, 1, 1, -1), dim=-1)

        return weighted_sum


class RNAModel(nn.Module):
    """
    Internally-Normalized Wide-Stream Residual BiLSTM.

    Architecture:
    1. Embeddings: Sequence (Atomic), Loop Type, Signed Pair Distance (Sinusoidal).
    2. Stem: BiLSTM (No Dropout) projecting to Stream Width (512).
    3. Backbone: 6x LayerNormBiLSTM blocks with Residual Connections and Inter-layer Dropout.
    4. Aggregation: Scalar Mixture of Stem + 6 Blocks.
    5. Head: Linear Projection to 3 targets.
    """

    def __init__(self, config=Config):
        super(RNAModel, self).__init__()
        self.config = config

        # 1. Embeddings
        self.seq_emb = nn.Embedding(4, config.EMBEDDING_DIM)
        self.loop_emb = nn.Embedding(7, config.LOOP_EMBEDDING_DIM)
        self.pair_emb = SinusoidalPositionalEmbedding(config.PAIR_EMBEDDING_DIM)

        input_dim = (
            config.EMBEDDING_DIM + config.LOOP_EMBEDDING_DIM + config.PAIR_EMBEDDING_DIM
        )

        # Dimensions
        # Stream width W = 512.
        # Since we use BiLSTM, the hidden size of each direction is W // 2 = 256.
        self.stream_dim = config.HIDDEN_DIM
        lstm_hidden = self.stream_dim // 2

        # 2. Stem
        # Projects concatenated embeddings to the residual stream width.
        # No dropout here to preserve initial projection fidelity.
        self.stem = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 3. Backbone
        self.blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(config.NUM_LAYERS):
            # LayerNormBiLSTM applies LN internally to gates.
            # Input size is stream_dim (512), output is stream_dim (512).
            self.blocks.append(LayerNormBiLSTM(self.stream_dim, lstm_hidden))
            self.dropouts.append(nn.Dropout(config.DROPOUT))

        # 4. Aggregation
        # Mixture of Stem output + 6 Block outputs = 7 tensors
        self.mixture = ScalarMixture(config.NUM_LAYERS + 1)

        # 5. Head
        self.head = nn.Linear(self.stream_dim, config.NUM_CLASSES)

    def forward(self, sequence, loop_type, pairing_distance):
        # Embed inputs
        emb_s = self.seq_emb(sequence)
        emb_l = self.loop_emb(loop_type)
        emb_p = self.pair_emb(pairing_distance)

        # Concatenate
        x = torch.cat([emb_s, emb_l, emb_p], dim=-1)

        # Stem
        x, _ = self.stem(x)
        # x shape: (Batch, Seq, 512)

        # Collect outputs for mixture
        layer_outputs = [x]

        # Backbone with Residuals
        current = x
        for block, dropout in zip(self.blocks, self.dropouts):
            out = block(current)
            out = dropout(out)
            current = current + out
            layer_outputs.append(current)

        # Scalar Mixture
        aggregated = self.mixture(layer_outputs)

        # Head
        logits = self.head(aggregated)

        return logits


def loss_fn(pred, target):
    """
    Masked Mean Squared Error.
    Calculates MSE only for the first 68 positions (seq_scored).
    """
    # pred: (Batch, 107, 3)
    # target: (Batch, 68, 3)

    # Slice prediction to match target length
    seq_scored = target.shape[1]
    pred_scored = pred[:, :seq_scored, :]

    # Compute MSE
    mse = nn.MSELoss()(pred_scored, target)
    return mse


def train_epoch(model, loader, optimizer, device, clip_grad):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        seq = batch["sequence"].to(device)
        loop = batch["loop_type"].to(device)
        dist = batch["pairing_distance"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        pred = model(seq, loop, dist)
        loss = loss_fn(pred, target)

        loss.backward()

        # Gradient Clipping to stabilize high-capacity backbone
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pairing_distance"].to(device)
            target = batch["target"].to(device)

            pred = model(seq, loop, dist)

            # Slice to scored region for metric calculation
            seq_scored = target.shape[1]
            pred_scored = pred[:, :seq_scored, :]

            all_preds.append(pred_scored.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(all_targets, all_preds)
    return score


def generate_submission(model, config, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    test_loader = get_dataloader("test", shuffle=False)

    ids = []
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            loop = batch["loop_type"].to(device)
            dist = batch["pairing_distance"].to(device)
            batch_ids = batch["id"]

            output = model(seq, loop, dist)

            preds.append(output.cpu().numpy())
            ids.extend(batch_ids)

    preds = np.concatenate(preds, axis=0)  # (N, 107, 3)

    # Prepare data for DataFrame
    submission_rows = []

    # Target columns in prediction: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Target columns in submission: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    for i, sample_id in enumerate(ids):
        sample_pred = preds[i]  # (107, 3)
        for pos in range(config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"

            reactivity = sample_pred[pos, 0]
            deg_Mg_pH10 = sample_pred[pos, 1]
            deg_pH10 = 0.0  # Not predicted
            deg_Mg_50C = sample_pred[pos, 2]
            deg_50C = 0.0  # Not predicted

            submission_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = pd.DataFrame(submission_rows, columns=columns)

    df_sub.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")


def run_training_pipeline(config=Config):
    """
    Main function to run the training loop and generate submission.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load Data
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    # Initialize Model
    model = RNAModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_score = float("inf")

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, config.CLIP_GRAD
        )
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.MODEL_PATH)

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # Generate Submission
    generate_submission(model, config, device)
