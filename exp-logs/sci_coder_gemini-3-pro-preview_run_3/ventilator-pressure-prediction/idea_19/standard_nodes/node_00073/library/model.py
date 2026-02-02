import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.dataset import get_data_loaders
from library.utils import seed_everything, get_device, MaskedL1Loss


class DenseTCNBlock(nn.Module):
    """
    A single dense convolutional block for the TCN encoder.
    Consists of Conv1d -> BatchNorm -> GELU -> Dropout.
    Strictly uses dilation=1 (Dense) to preserve local fidelity.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(DenseTCNBlock, self).__init__()
        # Calculate padding to maintain sequence length with kernel_size
        # padding = (kernel_size - 1) // 2 for dilation=1
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class DenseTCNEncoder(nn.Module):
    """
    The Resistive Stream: Stack of DenseTCNBlocks.
    Models high-frequency, derivative-dependent dynamics.
    """

    def __init__(self, input_dim):
        super(DenseTCNEncoder, self).__init__()
        layers = []
        current_dim = input_dim

        # Build layers based on Config
        for out_dim in Config.TCN_CHANNELS:
            layers.append(
                DenseTCNBlock(
                    in_channels=current_dim,
                    out_channels=out_dim,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dropout=Config.TCN_DROPOUT,
                )
            )
            current_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        return self.net(x)


class RecurrentEncoder(nn.Module):
    """
    The Elastic Stream: High-Capacity Bidirectional LSTM.
    Models low-frequency, integral-dependent dynamics.
    """

    def __init__(self, input_dim):
        super(RecurrentEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_dim = Config.LSTM_HIDDEN_SIZE * 2

    def forward(self, x):
        # x shape: [Batch, Length, Features]
        output, _ = self.lstm(x)
        return output


class WideFusionHead(nn.Module):
    """
    Wide-Latent Integration Head.
    Concatenates branches and projects through a wide hidden layer.
    """

    def __init__(self, tcn_dim, lstm_dim):
        super(WideFusionHead, self).__init__()
        input_dim = tcn_dim + lstm_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.FUSION_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Config.FUSION_HIDDEN_DIM, 1),
        )

    def forward(self, tcn_out, lstm_out):
        # tcn_out: [Batch, Channels, Length] -> Permute to [Batch, Length, Channels]
        tcn_out = tcn_out.permute(0, 2, 1)

        # Concatenate along feature dimension
        # lstm_out: [Batch, Length, Features]
        fused = torch.cat([tcn_out, lstm_out], dim=2)

        # Project
        return self.net(fused)


class WSDHNet(nn.Module):
    """
    Wide-Fusion Stabilized Dense-Hybrid Network (WSDH-Net).
    Combines Dense TCN and Bi-LSTM with a wide fusion head.
    """

    def __init__(self, input_dim):
        super(WSDHNet, self).__init__()

        self.tcn_encoder = DenseTCNEncoder(input_dim)
        self.lstm_encoder = RecurrentEncoder(input_dim)

        self.head = WideFusionHead(
            tcn_dim=self.tcn_encoder.output_dim, lstm_dim=self.lstm_encoder.output_dim
        )

    def forward(self, x):
        # x shape: [Batch, Length, Features]

        # TCN Branch (Requires [Batch, Channels, Length])
        x_tcn = x.permute(0, 2, 1)
        tcn_out = self.tcn_encoder(x_tcn)

        # LSTM Branch (Requires [Batch, Length, Features])
        lstm_out = self.lstm_encoder(x)

        # Fusion
        out = self.head(tcn_out, lstm_out)

        # Remove last dimension [Batch, Length, 1] -> [Batch, Length]
        return out.squeeze(-1)


def train(load_cached_data=True):
    """
    Executes the training pipeline for WSDH-Net.
    """
    Config.initialize()
    seed_everything(Config.SEED)
    device = get_device()

    # 1. Load Data
    train_loader, val_loader, test_loader = get_data_loaders(
        load_cached_data=load_cached_data
    )

    # Determine input dimension from a batch
    sample_batch, _, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[2]
    print(f"Detected Input Dimension: {input_dim}")

    # 2. Initialize Model
    model = WSDHNet(input_dim=input_dim).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    criterion = MaskedL1Loss()

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0
        train_steps = 0

        # Training Step
        for x, u_out, y in train_loader:
            x, u_out, y = x.to(device), u_out.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)

            # Masked Loss
            loss = criterion(preds, y, u_out)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

            optimizer.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / train_steps

        # Validation Step
        model.eval()
        val_loss_sum = 0
        val_steps = 0

        with torch.no_grad():
            for x, u_out, y in val_loader:
                x, u_out, y = x.to(device), u_out.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y, u_out)
                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / val_steps

        # Scheduler Step
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.8f} - Val Loss: {avg_val_loss:.8f}"
        )

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! (Val Loss: {best_val_loss:.8f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.8f}")

    # 4. Inference and Submission
    predict(model, test_loader, device)


def predict(model, test_loader, device):
    """
    Generates predictions and creates the submission file.
    """
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            preds = model(x)
            # Flatten predictions (Batch, Length) -> (Batch * Length)
            predictions.extend(preds.view(-1).cpu().numpy())

    # Load sample submission to get IDs
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Ensure lengths match
    if len(predictions) != len(sub_df):
        print(
            f"Warning: Prediction count {len(predictions)} does not match submission rows {len(sub_df)}."
        )
        # Truncate or pad if necessary (though strictly shouldn't happen with correct data)
        if len(predictions) > len(sub_df):
            predictions = predictions[: len(sub_df)]

    sub_df["pressure"] = predictions
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
