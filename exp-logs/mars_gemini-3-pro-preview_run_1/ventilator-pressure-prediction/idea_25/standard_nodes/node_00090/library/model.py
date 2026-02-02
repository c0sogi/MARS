import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from library.config import Config
from library.utils import (
    AverageMeter,
    compute_mae,
    save_checkpoint,
    load_checkpoint,
    get_device,
)
from library.data import get_dataloaders

# =============================================================================
# Model Architecture
# =============================================================================


class MultiScaleStem(nn.Module):
    """
    Multi-Scale 1D Convolutional Block (Inception-style).
    Processes input sequence with kernel sizes [3, 5, 7] to capture short
    and medium term temporal dependencies immediately.
    """

    def __init__(self, input_dim, hidden_dim, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_dim // len(kernel_sizes),
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        # Calculate the concatenated dimension
        concat_dim = (hidden_dim // len(kernel_sizes)) * len(kernel_sizes)

        # Projection to mix multi-scale features into unified representation
        self.project = nn.Linear(concat_dim, hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq) for Conv1d
        x_in = x.transpose(1, 2)

        outs = []
        for conv in self.convs:
            outs.append(conv(x_in))

        # Concatenate along channel dimension: (Batch, C_total, Seq)
        x_cat = torch.cat(outs, dim=1)

        # Transpose back: (Batch, Seq, C_total)
        x_cat = x_cat.transpose(1, 2)

        # Linear projection
        out = self.project(x_cat)
        return out


class CompositeBlock(nn.Module):
    """
    Uniform Composite Block with Curated Context Injection and Strict Identity Residuals.
    Maintains uniform capacity (Hidden Dim 512) throughout.
    """

    def __init__(self, hidden_dim, context_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Aligned Bi-LSTM
        # Input is hidden_state + context
        # Hidden dim is hidden_dim // 2 per direction -> output is hidden_dim
        # This prevents the information bottleneck seen in standard Bi-LSTMs
        self.lstm = nn.LSTM(
            input_size=hidden_dim + context_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Pointwise FFN (Expansion 2x)
        # We avoid 4x expansion to maintain stability without normalization
        ffn_dim = hidden_dim * Config.FFN_EXPANSION_FACTOR
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # x: (Batch, Seq, Hidden)
        # context: (Batch, Seq, Context)

        # 1. Curated Context Injection
        # We re-inject the static/physics context at every block
        lstm_input = torch.cat([x, context], dim=-1)

        # 2. Aligned Bi-LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # 3. Strict Identity Residual 1
        # Dimensions match (512 -> 512), no linear projection on skip
        # Dropout applied to residual branch only
        x = x + self.dropout(lstm_out)

        # 4. Pointwise FFN
        ffn_out = self.ffn(x)

        # 5. Strict Identity Residual 2
        # No LayerNorm, preserving absolute pressure magnitude
        x = x + self.dropout(ffn_out)

        return x


class VentilatorModel(nn.Module):
    """
    Feature-Complete Uniform-Capacity Physics-Composite Network.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM

        # Context Projection
        # Learns a compact representation of the input (physics & static features)
        # to serve as the "Curated Context" injected into each block.
        self.context_dim = 64
        self.context_proj = nn.Linear(input_dim, self.context_dim)

        # Stem
        self.stem = MultiScaleStem(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            kernel_sizes=Config.STEM_KERNEL_SIZES,
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [
                CompositeBlock(self.hidden_dim, self.context_dim, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # Heads
        self.head = nn.Linear(self.hidden_dim, 1)
        self.aux_head = nn.Linear(self.hidden_dim, 1)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)
                        # Initialize forget gate bias to 1 for long-term memory
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)

    def forward(self, x):
        # x: (Batch, Seq, Feat)

        # Generate Context Vector
        context = self.context_proj(x)

        # Stem
        h = self.stem(x)

        aux_out = None

        # Backbone
        for i, block in enumerate(self.blocks):
            h = block(h, context)

            # Deep Supervision after Block 2 (index 1)
            # Provides gradient short-circuit
            if i == 1:
                aux_out = self.aux_head(h)

        # Final Head
        final_out = self.head(h)

        return final_out, aux_out


# =============================================================================
# Training Functions
# =============================================================================


def loss_fn(pred, target, u_out, aux_pred=None):
    """
    Weighted Masked L1 Loss.
    """
    # Create mask: 1 where u_out == 0 (inspiratory), 0 otherwise (expiratory)
    # The metric only evaluates the inspiratory phase.
    mask = 1 - u_out

    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    mask = mask.view(-1)

    # Main Loss
    loss_mae = torch.abs(pred - target) * mask
    loss_final = loss_mae.sum() / (mask.sum() + 1e-8)

    total_loss = loss_final

    # Aux Loss (Deep Supervision)
    if aux_pred is not None:
        aux_pred = aux_pred.view(-1)
        loss_aux_mae = torch.abs(aux_pred - target) * mask
        loss_aux = loss_aux_mae.sum() / (mask.sum() + 1e-8)
        total_loss += Config.AUX_WEIGHT * loss_aux

    return total_loss


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    for batch_idx, (X, y, u_out) in enumerate(loader):
        X, y, u_out = X.to(device), y.to(device), u_out.to(device)

        optimizer.zero_grad()

        pred, aux_pred = model(X)

        loss = loss_fn(pred, y, u_out, aux_pred)

        loss.backward()

        # Strict Gradient Clipping (Critical for LSTM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()
        scheduler.step()

        # Metrics
        batch_size = X.size(0)
        loss_meter.update(loss.item(), batch_size)

        # Calculate MAE for reporting (only inspiratory)
        with torch.no_grad():
            mae = compute_mae(pred, y, u_out)
            mae_meter.update(mae, batch_size)

    return loss_meter.avg, mae_meter.avg


def validate(model, loader, device):
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for X, y, u_out in loader:
            X, y, u_out = X.to(device), y.to(device), u_out.to(device)

            pred, aux_pred = model(X)

            loss = loss_fn(pred, y, u_out, aux_pred)

            batch_size = X.size(0)
            loss_meter.update(loss.item(), batch_size)

            mae = compute_mae(pred, y, u_out)
            mae_meter.update(mae, batch_size)

    return loss_meter.avg, mae_meter.avg


def generate_submission(model, test_loader, device):
    print("Generating submission...")

    # Load Best Model
    _, best_mae = load_checkpoint(model, filename="model_best.pth")
    print(f"Loaded best model with MAE: {best_mae:.6f}")
    model.eval()

    predictions = []

    with torch.no_grad():
        for X, _, _ in tqdm(test_loader, desc="Inference"):
            X = X.to(device)
            pred, _ = model(X)  # Ignore aux head
            predictions.append(pred.cpu().numpy().flatten())

    predictions = np.concatenate(predictions)

    # Load Sample Submission to get IDs
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Ensure lengths match
    if len(predictions) != len(sub_df):
        print(
            f"Warning: Prediction length {len(predictions)} != Submission length {len(sub_df)}"
        )

    sub_df["pressure"] = predictions

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    print(f"Starting training for {Config.EXPERIMENT_ID}...")
    Config.print_config()
    device = get_device()

    # Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine Input Dimension
    # X shape: (Batch, Seq, Feat)
    sample_X = train_loader.dataset.X
    input_dim = sample_X.shape[-1]
    print(f"Input Dimension: {input_dim}")

    # Model
    model = VentilatorModel(input_dim=input_dim).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    best_mae = float("inf")

    for epoch in range(Config.EPOCHS):
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_mae = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        # Save Best
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_mae,
                },
                is_best=True,
                filename="checkpoint.pth",
            )
            print(f"  >>> New Best MAE: {best_mae:.6f} saved.")

    print(f"Training complete. Best Val MAE: {best_mae:.6f}")

    # Generate Submission
    generate_submission(model, test_loader, device)
