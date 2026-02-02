import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_datasets

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) with Sigmoid activation for the gate.
    Structure:
        x -> Linear(in, out * 2) -> split -> a, b
        output = a * sigmoid(b)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        # Project to double dimension
        proj = self.linear(x)
        # Split into value (u) and gate (v) branches
        u, v = proj.chunk(2, dim=-1)
        # Apply Gating
        return u * torch.sigmoid(v)


class ResidualGatedBlock(nn.Module):
    """
    Residual Block with GLU, BatchNorm, and Dropout.
    Structure:
        x_out = x_in + Dropout(BatchNorm(GLU(x_in)))
    Operates at constant width.
    """

    def __init__(self, width, dropout_rate):
        super().__init__()
        self.glu = GatedLinearUnit(width, width)
        self.bn = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.glu(x)
        out = self.bn(out)
        out = self.dropout(out)
        return out + residual


class DownsampleBlock(nn.Module):
    """
    Downsampling Block to reduce dimension between stages.
    Structure:
        Main Path: GLU(in -> out) -> BN -> Dropout
        Skip Path: Linear(in -> out)
        x_out = Main(x) + Skip(x)
    """

    def __init__(self, in_width, out_width, dropout_rate):
        super().__init__()
        # Main path: Project and Gate
        self.glu = GatedLinearUnit(in_width, out_width)
        self.bn = nn.BatchNorm1d(out_width)
        self.dropout = nn.Dropout(dropout_rate)

        # Skip path: Linear Projection to match dimensions
        self.proj = nn.Linear(in_width, out_width)

    def forward(self, x):
        # Skip connection with projection
        residual = self.proj(x)

        # Main path
        out = self.glu(x)
        out = self.bn(out)
        out = self.dropout(out)

        return out + residual


class ResFunnelGLU(nn.Module):
    """
    Residual Funnel Gated Network.
    Features:
        - Categorical Embedding
        - Continuous Feature Fusion
        - 3-Stage Funnel Architecture (512 -> 256 -> 128)
        - Residual Gated Blocks
        - Downsampling with Projections
    """

    def __init__(self):
        super().__init__()

        # 1. Input Processing
        # Embedding for f_27 characters
        self.emb = nn.Embedding(Config.VOCAB_SIZE, Config.EMBED_DIM)

        # Calculate fusion dimension
        # Continuous features + Flattened embeddings (10 chars * 32 dim)
        fusion_input_dim = Config.NUM_CONT_FEATURES + (
            Config.F27_SEQ_LEN * Config.EMBED_DIM
        )

        # Initial projection to Stage 1 width
        self.fusion = nn.Linear(fusion_input_dim, Config.INIT_WIDTH)

        # 2. Backbone Stages
        # Config.STAGES = [512, 256, 128]

        # Stage 1: Width 512
        # Multiple identity blocks (3 blocks)
        self.stage1 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[0], Config.DROPOUT_RATE),
        )

        # Downsample 1: 512 -> 256
        self.down1 = DownsampleBlock(
            Config.STAGES[0], Config.STAGES[1], Config.DROPOUT_RATE
        )

        # Stage 2: Width 256
        # Multiple identity blocks (3 blocks)
        self.stage2 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[1], Config.DROPOUT_RATE),
        )

        # Downsample 2: 256 -> 128
        self.down2 = DownsampleBlock(
            Config.STAGES[1], Config.STAGES[2], Config.DROPOUT_RATE
        )

        # Stage 3: Width 128
        # Multiple identity blocks (3 blocks)
        self.stage3 = nn.Sequential(
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
            ResidualGatedBlock(Config.STAGES[2], Config.DROPOUT_RATE),
        )

        # 3. Output Head
        self.head = nn.Linear(Config.STAGES[2], 1)

    def forward(self, cont_data, cat_data):
        # cont_data: (B, 30)
        # cat_data: (B, 10)

        # Embed and flatten categorical features
        emb = self.emb(cat_data)  # (B, 10, 32)
        emb_flat = emb.view(emb.size(0), -1)  # (B, 320)

        # Concatenate
        x = torch.cat([cont_data, emb_flat], dim=1)

        # Fusion
        x = self.fusion(x)

        # Stage 1
        x = self.stage1(x)

        # Downsample
        x = self.down1(x)

        # Stage 2
        x = self.stage2(x)

        # Downsample
        x = self.down2(x)

        # Stage 3
        x = self.stage3(x)

        # Head (Sigmoid for binary classification)
        return torch.sigmoid(self.head(x))


# ------------------------------------------------------------------------------
# Training & Inference Logic
# ------------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        cont = batch["cont"].to(device)
        cat = batch["cat"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        output = model(cont, cat)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * cont.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)
            target = batch["target"].to(device)

            output = model(cont, cat)
            loss = criterion(output, target)

            total_loss += loss.item() * cont.size(0)
            all_targets.append(target.cpu().numpy())
            all_preds.append(output.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc


def run_training():
    print(f"Device: {Config.DEVICE}")

    # Load Data
    train_ds, val_ds, _ = get_datasets(load_cached_data=True, debug=Config.DEBUG)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = ResFunnelGLU().to(Config.DEVICE)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCELoss()

    # Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def generate_submission():
    print("Generating submission...")

    # Load Data
    _, _, test_ds = get_datasets(load_cached_data=True, debug=Config.DEBUG)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = ResFunnelGLU().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        print(
            "Warning: Best model not found. Using initialized weights (this is likely an error)."
        )

    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            cont = batch["cont"].to(Config.DEVICE)
            cat = batch["cat"].to(Config.DEVICE)

            output = model(cont, cat)
            all_preds.append(output.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Create Submission DataFrame
    # Need to load test IDs. Using dataset.py logic, test_ds is aligned with test_metadata.csv
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    submission = pd.DataFrame({"id": test_meta["id"], "target": all_preds})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    run_training()
    generate_submission()


# Execute main function
main()
