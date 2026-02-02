import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import os
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import mcrmse_metric, seed_everything, format_submission
from library.dataset import RNADataset


class RNATransformer(nn.Module):
    def __init__(self):
        super(RNATransformer, self).__init__()

        # Hyperparameters
        self.seq_len = Config.SEQ_LEN
        self.embed_dim = Config.EMBED_DIM

        # Calculate sub-embedding dimension for concatenation
        # We have 3 inputs: sequence, structure, loop_type
        # We want total dim to be EMBED_DIM (192). So each gets 192 // 3 = 64.
        self.sub_embed_dim = self.embed_dim // 3
        assert (
            self.sub_embed_dim * 3 == self.embed_dim
        ), "EMBED_DIM must be divisible by 3"

        # 1. Embeddings
        self.embed_seq = nn.Embedding(Config.VOCAB_SIZE_SEQ, self.sub_embed_dim)
        self.embed_struct = nn.Embedding(Config.VOCAB_SIZE_STRUCT, self.sub_embed_dim)
        self.embed_loop = nn.Embedding(Config.VOCAB_SIZE_LOOP, self.sub_embed_dim)

        # 2. Positional Encoding (Learnable)
        self.pos_enc = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.N_HEADS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.N_LAYERS
        )

        # 4. Output Head
        self.output_head = nn.Linear(self.embed_dim, Config.NUM_TARGETS)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings and linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sequence, structure, loop):
        # sequence, structure, loop: (Batch, Seq_Len)

        # Embed inputs
        x_seq = self.embed_seq(sequence)  # (B, L, d/3)
        x_struct = self.embed_struct(structure)  # (B, L, d/3)
        x_loop = self.embed_loop(loop)  # (B, L, d/3)

        # Concatenate
        x = torch.cat([x_seq, x_struct, x_loop], dim=-1)  # (B, L, d)

        # Add Positional Encoding
        x = x + self.pos_enc

        # Transformer Encoder
        # No mask needed as sequence length is fixed and full context is used
        x = self.transformer_encoder(x)  # (B, L, d)

        # Output Projection
        out = self.output_head(x)  # (B, L, 5)

        return out


def masked_mse_loss(preds, targets, mask):
    """
    Computes MSE loss only on valid positions defined by the mask.

    Args:
        preds: (B, L, 5)
        targets: (B, L, 5)
        mask: (B, L) - 1.0 for valid positions, 0.0 otherwise
    """
    # Squared Error
    loss = (preds - targets) ** 2

    # Apply mask
    # Mask shape is (B, L), need (B, L, 1) to broadcast over 5 targets
    mask_expanded = mask.unsqueeze(-1)
    loss = loss * mask_expanded

    # Average over non-masked elements
    # Sum of errors / Sum of mask elements (times 5 targets)
    # To avoid division by zero, add epsilon
    sum_loss = loss.sum()
    num_active_elements = mask_expanded.sum() * preds.shape[-1]

    return sum_loss / (num_active_elements + 1e-8)


def train():
    """
    Main training routine.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # --- Data Loading ---
    print("Initializing Datasets...")
    train_dataset = RNADataset(split="train", load_cached_data=True)
    val_dataset = RNADataset(split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    model = RNATransformer().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- Training Loop ---
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            # Move to device
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["loop"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            preds = model(seq, struct, loop)
            loss = masked_mse_loss(preds, targets, mask)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                struct = batch["structure"].to(device)
                loop = batch["loop"].to(device)
                targets = batch["targets"].to(device)
                mask = batch["mask"].to(device)

                preds = model(seq, struct, loop)
                loss = masked_mse_loss(preds, targets, mask)
                val_loss_accum += loss.item()

                # Store for MCRMSE calculation
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        avg_val_loss = val_loss_accum / len(val_loader)

        # Calculate Metric
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_mcrmse = mcrmse_metric(all_targets_tensor, all_preds_tensor)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  New best model saved! MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")


def inference():
    """
    Runs inference on the test set using the best saved model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    print("Loading Test Data...")
    test_dataset = RNADataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = RNATransformer().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(device)
            struct = batch["structure"].to(device)
            loop = batch["loop"].to(device)

            # Forward pass
            preds = model(seq, struct, loop)  # (B, 107, 5)

            # Clamp predictions if necessary (optional, but degradation shouldn't be extremely negative)
            # However, ground truth has negatives, so we leave raw.

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(batch["id"])

    # Concatenate predictions
    final_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # Format and Save
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    format_submission(all_ids, final_preds, Config.SUBMISSION_FILE)
    print("Done.")
