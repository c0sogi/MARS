import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import Config
from library.data_loader import get_dataloaders
from library.utils import compute_auc, seed_everything

# ------------------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------------------


class CategoricalTransformer(nn.Module):
    """
    Specialized Transformer Encoder for the categorical sequence feature.
    Includes Embedding, Explicit Positional Embedding, and Transformer Layers.
    Cite solution_lesson_node_00030: Explicit Positional Embeddings are critical for short sequences.
    """

    def __init__(self, seq_len, vocab_size, embed_dim, depth, heads, ff_dim, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Categorical embedding
        self.cat_embedding = nn.Embedding(vocab_size + 1, embed_dim)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x_cat):
        # x_cat: (Batch, seq_len)

        # Embed: (Batch, seq_len, embed_dim)
        x = self.cat_embedding(x_cat)

        # Add Positional Embeddings
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        return x


class ResGatedBlock(nn.Module):
    """
    Residual Block with Gated Linear Unit (GLU), BatchNorm, and Dropout.
    Handles dimension changes via projected residual connection.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        # Main projection
        self.linear = nn.Linear(in_dim, out_dim)
        # Gating projection
        self.gate = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        # x: (Batch, in_dim)

        # GLU Mechanism
        h = self.linear(x)
        g = torch.sigmoid(self.gate(x))
        out = h * g

        # Regularization
        out = self.bn(out)
        out = self.dropout(out)

        # Residual
        res = self.res_proj(x)

        return out + res


class TransformerResFunnel(nn.Module):
    """
    Hybrid architecture (Dual-Stream / Late Fusion):
    1. Stream A: Categorical Sequence -> Embedding + Positional -> Transformer -> Flatten.
    2. Stream B: Continuous Features -> Identity (Concatenated directly).
    3. Fusion -> Residual Funnel Backbone.

    Cite solution_lesson_node_00032: Late Fusion is superior to Unified Tokenization for hybrid data.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.num_cont = len(Config.CONT_FEATURES)
        self.cat_seq_len = Config.CAT_SEQ_LEN
        self.embed_dim = Config.EMBED_DIM

        # 1. Categorical Stream (Transformer)
        self.cat_encoder = CategoricalTransformer(
            seq_len=self.cat_seq_len,
            vocab_size=Config.VOCAB_SIZE,
            embed_dim=self.embed_dim,
            depth=Config.TRANSFORMER_DEPTH,
            heads=Config.TRANSFORMER_HEADS,
            ff_dim=Config.TRANSFORMER_FF_DIM,
            dropout=Config.TRANSFORMER_DROPOUT,
        )

        # 2. Residual Funnel Backbone
        # Input dimension: Flattened Transformer Output + Continuous Features
        # (10 * 64) + 30 = 640 + 30 = 670
        in_dim = (self.cat_seq_len * self.embed_dim) + self.num_cont

        layers = []
        current_dim = in_dim

        # Build stages
        for next_dim in Config.FUNNEL_STAGES:
            layers.append(
                ResGatedBlock(
                    in_dim=current_dim, out_dim=next_dim, dropout=Config.FUNNEL_DROPOUT
                )
            )
            current_dim = next_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x_cont, x_cat):
        # Stream A: Categorical
        x_cat_enc = self.cat_encoder(x_cat)  # (Batch, 10, 64)
        x_cat_flat = x_cat_enc.flatten(start_dim=1)  # (Batch, 640)

        # Stream B: Continuous
        # x_cont is (Batch, 30) - we use it directly (Late Fusion)

        # Fusion
        x = torch.cat([x_cat_flat, x_cont], dim=1)  # (Batch, 670)

        # Backbone
        x = self.backbone(x)

        # Head
        logits = self.head(x)
        return logits


# ------------------------------------------------------------------------------
# Training & Execution Logic
# ------------------------------------------------------------------------------


def train():
    """
    Executes the training pipeline:
    1. Loads data.
    2. Initializes model, optimizer, loss.
    3. Runs training loop with early stopping.
    4. Generates submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing training on {device}...")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # Initialize Model
    model = TransformerResFunnel().to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training State
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print("Starting training loop...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_cont = batch["cont"].to(device)
            x_cat = batch["cat"].to(device)
            y = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_cont = batch["cont"].to(device)
                x_cat = batch["cat"].to(device)
                y = batch["target"].to(device).unsqueeze(1)

                logits = model(x_cont, x_cat)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_targets.append(y.cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()

        current_auc = compute_auc(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val AUC: {current_auc:.10f}"
        )

        # --- Early Stopping & Saving ---
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # --------------------------------------------------------------------------
    # Inference & Submission
    # --------------------------------------------------------------------------
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_preds = []
    # Note: Test loader order is preserved and matches test_metadata / test_ids

    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["cont"].to(device)
            x_cat = batch["cat"].to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Load test metadata to get IDs
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    submission = pd.DataFrame({"id": test_meta["id"], "target": test_preds})

    submission.to_csv(Config.SUBMISSION_SAVE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_SAVE_PATH}")
