import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, get_device, compute_auc, print_metric
from library.data import get_dataloaders

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class GLUBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Structure: x_out = Shortcut(x) + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Pre-activation Batch Norm
        self.bn = nn.BatchNorm1d(in_dim)

        # Linear layer mapping to 2 * out_dim for GLU
        self.linear = nn.Linear(in_dim, out_dim * 2)

        # Gated Linear Unit
        self.glu = nn.GLU(dim=1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Shortcut projection if dimensions change
        if in_dim != out_dim:
            self.shortcut_proj = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut_proj = nn.Identity()

    def forward(self, x):
        # Main Branch
        # Pre-activation: BN -> Linear -> GLU -> Dropout
        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        # Shortcut Branch
        # Note: In Pre-Act blocks, the shortcut operates on the un-normalized input
        shortcut = self.shortcut_proj(x)

        return shortcut + out


class HybridTransformerResFunnel(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # Stream 1: Categorical Sequence (Transformer)
        # ----------------------------------------------------------------------
        self.char_embed = nn.Embedding(Config.VOCAB_SIZE, Config.TRANSFORMER_EMBED_DIM)

        # Learnable Positional Encoding: (1, SeqLen, Dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, Config.SEQUENCE_LENGTH, Config.TRANSFORMER_EMBED_DIM)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.TRANSFORMER_EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.TRANSFORMER_EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Flattened dimension for transformer output
        self.trans_flat_dim = Config.SEQUENCE_LENGTH * Config.TRANSFORMER_EMBED_DIM

        # ----------------------------------------------------------------------
        # Stream 2: Continuous
        # ----------------------------------------------------------------------
        # Path A: Raw Continuous (30 dims) - Passed through directly

        # ----------------------------------------------------------------------
        # Fusion Layer
        # ----------------------------------------------------------------------
        # Total input dimension to backbone
        # Transformer Flat + Raw Continuous
        self.fusion_input_dim = self.trans_flat_dim + Config.NUM_CONTINUOUS_FEATURES

        # Project to Stage 1 width
        stage_dims = Config.BACKBONE_STAGES
        self.fusion_proj = nn.Linear(self.fusion_input_dim, stage_dims[0])
        self.fusion_bn = nn.BatchNorm1d(stage_dims[0])

        # ----------------------------------------------------------------------
        # Backbone: ResFunnel-GLU
        # ----------------------------------------------------------------------
        # Stage 1 (512)
        self.stage1 = nn.Sequential(
            GLUBlock(stage_dims[0], stage_dims[0], Config.BACKBONE_DROPOUT),
            GLUBlock(stage_dims[0], stage_dims[0], Config.BACKBONE_DROPOUT),
        )

        # Transition 1 (512 -> 256)
        self.trans1 = GLUBlock(stage_dims[0], stage_dims[1], Config.BACKBONE_DROPOUT)

        # Stage 2 (256)
        self.stage2 = nn.Sequential(
            GLUBlock(stage_dims[1], stage_dims[1], Config.BACKBONE_DROPOUT),
            GLUBlock(stage_dims[1], stage_dims[1], Config.BACKBONE_DROPOUT),
        )

        # Transition 2 (256 -> 128)
        self.trans2 = GLUBlock(stage_dims[1], stage_dims[2], Config.BACKBONE_DROPOUT)

        # Stage 3 (128)
        self.stage3 = nn.Sequential(
            GLUBlock(stage_dims[2], stage_dims[2], Config.BACKBONE_DROPOUT),
            GLUBlock(stage_dims[2], stage_dims[2], Config.BACKBONE_DROPOUT),
        )

        # ----------------------------------------------------------------------
        # Output Head
        # ----------------------------------------------------------------------
        self.head = nn.Linear(stage_dims[2], 1)

    def forward(self, x_seq, x_raw):
        # --- Stream 1: Transformer ---
        # x_seq: (B, 10)
        emb = self.char_embed(x_seq)  # (B, 10, 32)
        emb = emb + self.pos_embed  # Add positional encoding
        trans_out = self.transformer(emb)  # (B, 10, 32)
        trans_flat = trans_out.reshape(trans_out.size(0), -1)  # (B, 320)

        # --- Stream 2: Continuous ---
        # Path A: x_raw (B, 30) - No processing needed

        # --- Fusion ---
        concat = torch.cat([trans_flat, x_raw], dim=1)
        x = self.fusion_proj(concat)
        x = self.fusion_bn(x)

        # --- Backbone ---
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)

        # --- Head ---
        logits = self.head(x)
        return logits


# ------------------------------------------------------------------------------
# Training & Inference Logic
# ------------------------------------------------------------------------------


def train_model():
    seed_everything(Config.RANDOM_STATE)
    device = get_device()

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Model
    model = HybridTransformerResFunnel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_seq = batch["x_seq"].to(device)
            x_raw = batch["x_raw"].to(device)
            y = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x_seq, x_raw)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x_seq = batch["x_seq"].to(device)
                x_raw = batch["x_raw"].to(device)
                y = batch["target"].to(device)

                logits = model(x_seq, x_raw)
                probs = torch.sigmoid(logits).squeeze(1)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = compute_auc(val_targets, val_preds)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def generate_submission():
    device = get_device()

    # Load Data (Test loader)
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = HybridTransformerResFunnel().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No model checkpoint found. Using untrained model.")

    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch["x_seq"].to(device)
            x_raw = batch["x_raw"].to(device)

            logits = model(x_seq, x_raw)
            probs = torch.sigmoid(logits).squeeze(1)
            predictions.append(probs.cpu().numpy())

    predictions = np.concatenate(predictions)

    # Load Sample Submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    sample_sub["target"] = predictions

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    train_model()
    generate_submission()


# Execute main logic
if __name__ == "__main__":
    main()
