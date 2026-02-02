import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import VentilatorDataset

# ==========================================
# Model Architecture
# ==========================================


class Stem(nn.Module):
    """
    Mixed Multi-Scale Initialization Stem.
    Processes input sequence with multiple kernel sizes to capture short and medium term dynamics,
    then projects to a compressed bottleneck dimension.
    """

    def __init__(self, input_dim, stem_dim):
        super().__init__()
        # Multi-scale Inception-style convolutions
        # We project input to 64 channels per branch
        self.conv3 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)

        # Concatenated dimension: 64 * 3 = 192
        self.proj = nn.Linear(192, stem_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> Transpose for Conv1d: (Batch, Feat, Seq)
        x_t = x.transpose(1, 2)

        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        c7 = self.conv7(x_t)

        # Concatenate along channel dimension
        out = torch.cat([c3, c5, c7], dim=1)  # (Batch, 192, Seq)
        out = out.transpose(1, 2)  # (Batch, Seq, 192)

        return self.act(self.proj(out))


class WideIdentityBlock(nn.Module):
    """
    Wide-State Identity Block.
    Features:
    - Curated Context Injection (Static + Physics)
    - Wide-State Bi-LSTM (Output dim matches Model dim)
    - Strict Identity Residuals (No weights, no Norm)
    - Pointwise Channel Mixing
    """

    def __init__(self, model_dim, context_dim, lstm_hidden, dropout=0.1):
        super().__init__()

        # Context Injection: Input to LSTM is model_dim + context_dim
        # Bi-LSTM output is lstm_hidden * 2.
        # We set lstm_hidden = model_dim // 2 so output matches model_dim exactly.
        self.lstm = nn.LSTM(
            input_size=model_dim + context_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Pointwise FFN (Expansion factor 2)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (Batch, Seq, model_dim)
        # context: (Batch, Seq, context_dim)

        # 1. Curated Context Injection
        # Concatenate context to the latent state before entering the recurrent layer
        lstm_input = torch.cat([x, context], dim=-1)

        # 2. Wide-State Bi-LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # 3. Strict Identity Residual 1
        # Direct addition creates a gradient highway
        x = x + self.dropout(lstm_out)

        # 4. Pointwise Channel Mixing (FFN) & Residual 2
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class WideProjectedNet(nn.Module):
    """
    Wide-Projected Deeply-Supervised Physics-Identity Network.
    Decouples feature extraction (Stem) from latent modeling (Backbone).
    """

    def __init__(self, input_dim=14):
        super().__init__()

        stem_dim = Config.stem_dim
        model_dim = Config.model_dim
        lstm_hidden = Config.lstm_hidden
        dropout = Config.dropout

        # 1. Stem
        self.stem = Stem(input_dim, stem_dim)

        # 2. Wide Projection Adapter
        # Projects compressed stem features to high-capacity model dimension
        self.adapter = nn.Linear(stem_dim, model_dim)

        # 3. Backbone
        # Context features indices: R(3), C(4), R*u_in(6), Vol/C(7) -> 4 dimensions
        self.context_dim = 4

        self.block1 = WideIdentityBlock(
            model_dim, self.context_dim, lstm_hidden, dropout
        )
        self.block2 = WideIdentityBlock(
            model_dim, self.context_dim, lstm_hidden, dropout
        )
        self.block3 = WideIdentityBlock(
            model_dim, self.context_dim, lstm_hidden, dropout
        )
        self.block4 = WideIdentityBlock(
            model_dim, self.context_dim, lstm_hidden, dropout
        )

        # 4. Heads
        self.aux_head = nn.Linear(model_dim, 1)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat)

        # Extract Context: R, C, R_u_in, vol_C
        # We rely on the specific column ordering from FeatureEngineer in dataset.py
        # Indices: 3, 4, 6, 7
        context = x[:, :, [3, 4, 6, 7]]

        # Stem
        h = self.stem(x)  # -> (Batch, Seq, 512)

        # Adapter
        h = self.adapter(h)  # -> (Batch, Seq, 1024)

        # Blocks
        h = self.block1(h, context)
        h = self.block2(h, context)

        # Deep Supervision (Auxiliary Head)
        aux_out = self.aux_head(h)

        h = self.block3(h, context)
        h = self.block4(h, context)

        # Final Head
        final_out = self.head(h)

        return final_out, aux_out


# ==========================================
# Training & Inference Logic
# ==========================================


def masked_mae_loss(pred, target, mask):
    """
    Computes L1 loss only on inspiratory phase.
    mask input (u_out) is 1 for expiratory, 0 for inspiratory.
    We want to weight inspiratory samples with 1.
    """
    w = 1 - mask
    loss = torch.abs(pred - target)
    loss = loss * w

    sum_w = w.sum()
    if sum_w < 1e-6:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return loss.sum() / sum_w


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for x, u_out, y in loader:
        x, u_out, y = x.to(device), u_out.to(device), y.to(device)

        optimizer.zero_grad()

        pred, aux_pred = model(x)

        # Squeeze last dim to match target shape
        pred = pred.squeeze(-1)
        aux_pred = aux_pred.squeeze(-1)

        # Loss calculation
        loss_final = masked_mae_loss(pred, y, u_out)
        loss_aux = masked_mae_loss(aux_pred, y, u_out)

        # Weighted sum
        loss = loss_final + Config.aux_weight * loss_aux

        loss.backward()

        # Strict Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for x, u_out, y in loader:
            x, u_out, y = x.to(device), u_out.to(device), y.to(device)

            pred, _ = model(x)
            pred = pred.squeeze(-1)

            mae = masked_mae_loss(pred, y, u_out)
            total_mae += mae.item()

    return total_mae / len(loader)


def predict_test(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for x, _, _ in loader:  # Test loader returns dummy y
            x = x.to(device)
            pred, _ = model(x)
            preds.append(pred.squeeze(-1).cpu().numpy())

    return np.concatenate(preds)


def main():
    """
    Main execution pipeline:
    1. Setup and Data Loading
    2. Model Initialization
    3. Training Loop with Validation and Checkpointing
    4. Inference on Test Set
    5. Submission Generation
    """
    # Setup
    Config.setup()
    device = torch.device(Config.device)
    print(f"Using device: {device}")

    # Data Loading
    print("Loading datasets...")
    # VentilatorDataset handles feature engineering and caching internally
    train_ds = VentilatorDataset(split="train")
    val_ds = VentilatorDataset(split="val")
    test_ds = VentilatorDataset(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model Initialization
    input_dim = train_ds.x.shape[2]
    print(f"Input feature dimension: {input_dim}")
    model = WideProjectedNet(input_dim=input_dim).to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr_max, weight_decay=Config.weight_decay
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr_max,
        epochs=Config.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.pct_start,
    )

    # Training Loop
    best_mae = float("inf")
    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_mae = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.model_path)
            print(f"  New best model saved! MAE: {best_mae:.6f}")

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.model_path, map_location=device))

    print("Predicting on test set...")
    flat_preds = predict_test(model, test_loader, device).flatten()

    # Submission Generation
    print("Generating submission file...")
    # Load raw test file to get IDs
    test_df = pd.read_csv(Config.test_file)

    # Ensure sorting matches FeatureEngineer logic (breath_id, time_step)
    # This guarantees that the flat predictions align with the correct IDs
    test_df = test_df.sort_values([Config.breath_id_col, "time_step"]).reset_index(
        drop=True
    )

    submission = pd.DataFrame({"id": test_df["id"], "pressure": flat_preds})

    submission.to_csv(Config.output_submission_path, index=False)
    print(f"Submission saved to {Config.output_submission_path}")
