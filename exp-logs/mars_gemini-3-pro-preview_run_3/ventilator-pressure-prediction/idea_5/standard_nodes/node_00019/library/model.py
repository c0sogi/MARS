import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from torch.nn.utils import weight_norm

from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data import get_dataloaders

# ==========================================
# TCN Components
# ==========================================


class Chomp1d(nn.Module):
    """
    Removes the last 'chomp_size' elements from the input.
    Used to ensure causality in TCN by removing padding from the future.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """
    A single residual block for the TCN.
    Contains two dilated convolutions with causal padding, ReLU, and Dropout.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # First dilated convolution
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection: 1x1 conv if dimensions change
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBranch(nn.Module):
    """
    The TCN Branch of CAP-Net.
    Stacks TemporalBlocks with increasing dilation.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Features, Length)
        return self.network(x)


# ==========================================
# CAP-Net Model
# ==========================================


class CAPNet(nn.Module):
    """
    Context-Aware Parallel Hybrid Network.
    Combines a TCN branch (Resistive dynamics) and an LSTM branch (Elastic dynamics).
    """

    def __init__(self):
        super(CAPNet, self).__init__()

        # --- TCN Branch ---
        # Input to TCN: (Batch, Input_Dim, Seq_Len)
        self.tcn = TCNBranch(
            num_inputs=Config.INPUT_DIM,
            num_channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )
        tcn_out_dim = Config.TCN_CHANNELS[-1]

        # --- LSTM Branch ---
        # Input to LSTM: (Batch, Seq_Len, Input_Dim)
        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )
        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.LSTM_BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )

        # --- Fusion Head ---
        fusion_dim = tcn_out_dim + lstm_out_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, Config.FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FC_HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)

        # 1. TCN Path
        # Permute for Conv1d: (Batch, Features, Seq_Len)
        x_tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(x_tcn_in)
        # Permute back: (Batch, Seq_Len, TCN_Channels)
        tcn_out = tcn_out.permute(0, 2, 1)

        # 2. LSTM Path
        # Standard LSTM input: (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm(x)

        # 3. Fusion
        # Concatenate along feature dimension
        combined = torch.cat([tcn_out, lstm_out], dim=2)

        # Prediction
        # Output: (Batch, Seq_Len, 1)
        pred = self.head(combined)

        # Remove last dimension: (Batch, Seq_Len)
        return pred.squeeze(-1)


# ==========================================
# Training & Inference Pipeline
# ==========================================


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Computes L1 Loss only for the inspiratory phase (u_out == 0).
    """
    mask = (u_out == 0).float()
    loss = torch.abs(y_pred - y_true) * mask
    # Avoid division by zero
    sum_mask = mask.sum()
    if sum_mask < 1e-6:
        return loss.sum() * 0.0
    return loss.sum() / sum_mask


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, u_out, y in loader:
        x, u_out, y = x.to(device), u_out.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = masked_mae_loss(y_pred, y, u_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    u_outs = []

    with torch.no_grad():
        for x, u_out, y in loader:
            x, u_out, y = x.to(device), u_out.to(device), y.to(device)
            y_pred = model(x)

            preds.append(y_pred.cpu())
            targets.append(y.cpu())
            u_outs.append(u_out.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    u_outs = torch.cat(u_outs)

    mae = compute_metric(preds, targets, u_outs)
    return mae


def generate_submission(model, loader, device, output_path):
    model.eval()
    preds = []

    with torch.no_grad():
        for x, u_out in loader:
            x = x.to(device)
            y_pred = model(x)
            preds.append(y_pred.cpu().numpy().flatten())

    all_preds = np.concatenate(preds)

    # Create DataFrame
    # Note: id starts from 1
    ids = np.arange(1, len(all_preds) + 1)
    sub_df = pd.DataFrame({"id": ids, "pressure": all_preds})

    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main entry point for training the CAP-Net model.
    Handles data loading, training loop, early stopping, and submission generation.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    # Clean start logic handled by Config.CLEAN_START in get_dataloaders via Config
    if Config.CLEAN_START:
        print("Cleaning cache for fresh start...")
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".npy") or f.endswith(".pth"):
                os.remove(os.path.join(Config.WORKING_DIR, f))

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHE
    )

    # 3. Model Initialization
    model = CAPNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # 4. Training Loop
    best_mae = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae = validate(model, val_loader, device)

        scheduler.step(val_mae)

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae} | Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! MAE: {best_mae}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 5. Final Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Pipeline complete.")


# Maintain backward compatibility via aliasing (Cite debug_lesson_3)
BiLSTMModel = CAPNet
