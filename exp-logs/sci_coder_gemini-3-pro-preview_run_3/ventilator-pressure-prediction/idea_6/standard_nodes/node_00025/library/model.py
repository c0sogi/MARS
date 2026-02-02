import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from library.config import Config
from library.dataset import get_dataloaders


class TemporalBlock(nn.Module):
    """
    A single block for the TCN branch.
    Consists of two dilated convolutions with ReLU and Dropout.
    Includes a residual connection with an optional 1x1 convolution for channel matching.
    """

    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()

        # First Conv-ReLU-Dropout
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second Conv-ReLU-Dropout
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )

        # Downsample/Project for residual connection if dimensions change
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


class PyramidalTCN(nn.Module):
    """
    TCN Backbone with Pyramidal Channel Scaling.
    Channels increase as dilation increases.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(PyramidalTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Calculate padding to maintain sequence length ('same' padding logic for dilated conv)
            # padding = (kernel_size - 1) * dilation // 2
            padding = (kernel_size - 1) * dilation_size // 2

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Input_Channels, Seq_Len)
        return self.network(x)


class PCANet(nn.Module):
    """
    Pyramidal Context-Aware Hybrid Network (PCA-Net).
    Combines a Pyramidal TCN branch and a BiLSTM branch with a Physics Skip Connection.
    """

    def __init__(self, config=Config):
        super(PCANet, self).__init__()

        self.input_dim = config.INPUT_DIM
        self.seq_len = config.SEQ_LEN

        # --- Branch 1: Pyramidal TCN (Resistive Stream) ---
        self.tcn = PyramidalTCN(
            num_inputs=self.input_dim,
            num_channels=config.TCN_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dropout=config.TCN_DROPOUT,
        )
        self.tcn_out_dim = config.TCN_CHANNELS[-1]

        # --- Branch 2: BiLSTM (Elastic Stream) ---
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_LAYERS > 1 else 0,
        )
        self.lstm_out_dim = config.LSTM_HIDDEN * 2

        # --- Fusion & Head ---
        # Input to head is: TCN_Out + LSTM_Out + Raw_Input (Skip Connection)
        self.fusion_dim = self.tcn_out_dim + self.lstm_out_dim + self.input_dim

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)

        # 1. TCN Path
        # TCN expects (Batch, Channels, Seq_Len)
        x_tcn_in = x.transpose(1, 2)
        tcn_out = self.tcn(x_tcn_in)
        # Transpose back to (Batch, Seq_Len, Channels)
        tcn_out = tcn_out.transpose(1, 2)

        # 2. LSTM Path
        # LSTM expects (Batch, Seq_Len, Features)
        lstm_out, _ = self.lstm(x)

        # 3. Fusion with Physics Skip Connection
        # Concatenate: TCN features, LSTM features, and original Input (Context/Physics)
        fused = torch.cat([tcn_out, lstm_out, x], dim=2)

        # 4. Regression Head
        output = self.head(fused)

        # Output shape: (Batch, Seq_Len, 1) -> Squeeze to (Batch, Seq_Len)
        return output.squeeze(-1)


class MaskedMAELoss(nn.Module):
    """
    Computes L1 Loss strictly during the inspiratory phase (u_out == 0).
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        # Create mask: 1 where u_out == 0 (Inspiratory), 0 otherwise
        mask = 1 - u_out

        loss = self.l1(pred, target)

        # Apply mask
        masked_loss = loss * mask

        # Average over the number of valid elements (sum(mask))
        # Add epsilon to avoid division by zero
        return masked_loss.sum() / (mask.sum() + 1e-8)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        u_out = batch["u_out"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, targets, u_out)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            u_out = batch["u_out"].to(device)
            targets = batch["target"].to(device)

            preds = model(inputs)
            loss = criterion(preds, targets, u_out)

            total_loss += loss.item()

    return total_loss / len(loader)


def predict_and_submit(model, test_loader, device, output_path):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            preds = model(inputs)
            predictions.append(preds.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.INPUT_DIR + "/sample_submission.csv")

    # Ensure lengths match
    if len(all_preds) != len(sample_sub):
        print(
            f"Warning: Prediction length {len(all_preds)} != Submission length {len(sample_sub)}"
        )
        # Truncate or pad if necessary, though strict adherence to pipeline should prevent this
        if len(all_preds) > len(sample_sub):
            all_preds = all_preds[: len(sample_sub)]

    sample_sub["pressure"] = all_preds
    sample_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function to train the PCA-Net and generate submission.
    """
    print("Initializing PCA-Net Training Pipeline...")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 2. Model Setup
    device = torch.device(Config.DEVICE)
    model = PCANet(Config).to(device)

    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train MAE: {train_loss:.6f} | Val MAE: {val_loss:.6f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"  -> Model saved to {Config.MODEL_PATH}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 4. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating submission...")
    predict_and_submit(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Pipeline complete.")
