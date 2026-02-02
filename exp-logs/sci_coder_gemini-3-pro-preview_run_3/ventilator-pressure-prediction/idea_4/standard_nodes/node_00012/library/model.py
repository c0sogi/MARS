import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config, set_seed
from library.dataset import DataManager
from library.utils import get_device

# ==========================================
# Architecture Components
# ==========================================


class TCNBranch(nn.Module):
    """
    Temporal Convolutional Network Branch for modeling fast resistive dynamics.
    Uses dilated convolutions to capture instantaneous flow and acceleration features.
    """

    def __init__(self, input_dim, channels, kernel_size, dropout):
        super().__init__()
        layers = []
        in_channels = input_dim
        dilation = 1

        for out_channels in channels:
            # Calculate padding to maintain sequence length for stride=1
            # Padding = (dilation * (kernel_size - 1)) // 2
            padding = (dilation * (kernel_size - 1)) // 2

            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(out_channels))

            in_channels = out_channels
            dilation *= 2  # Exponential dilation schedule

        self.net = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        return self.net(x)


class LSTMBranch(nn.Module):
    """
    Bidirectional LSTM Branch for modeling slow elastic state dynamics.
    Captures volume accumulation and history.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # x shape: (Batch, Length, Channels)
        # Output shape: (Batch, Length, Hidden*2)
        out, _ = self.lstm(x)
        return out


class DisentangledTCNLSTM(nn.Module):
    """
    Hybrid architecture that splits inputs into physical streams (Resistive vs Elastic),
    processes them in parallel, and fuses them with static lung attributes.
    """

    def __init__(self):
        super().__init__()

        input_dims = Config.get_input_dims()

        # 1. TCN Branch (Resistive Stream)
        self.tcn = TCNBranch(
            input_dim=input_dims["tcn"],
            channels=Config.TCN_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # 2. LSTM Branch (Elastic Stream)
        self.lstm = LSTMBranch(
            input_dim=input_dims["lstm"],
            hidden_dim=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            dropout=Config.LSTM_DROPOUT,
        )

        # 3. Fusion & Head
        # TCN output (C_tcn) + LSTM output (H*2) + Skip features (C_skip)
        fusion_dim = self.tcn.output_dim + self.lstm.output_dim + input_dims["skip"]

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, Config.FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.FC_HIDDEN_DIM, 1),
        )

    def forward(self, inputs):
        """
        Args:
            inputs (dict): Dictionary containing 'tcn', 'lstm', and 'skip' tensors.
        """
        # TCN Branch
        # Input: (B, C_in, L) -> Output: (B, C_out, L)
        tcn_out = self.tcn(inputs["tcn"])
        # Transpose to (B, L, C_out) for concatenation
        tcn_out = tcn_out.transpose(1, 2)

        # LSTM Branch
        # Input: (B, L, C_in) -> Output: (B, L, H*2)
        lstm_out = self.lstm(inputs["lstm"])

        # Skip Connection (Physics Injection)
        # Input: (B, L, C_skip)
        skip_out = inputs["skip"]

        # Concatenate all streams
        # Shape: (B, L, Total_Features)
        combined = torch.cat([tcn_out, lstm_out, skip_out], dim=2)

        # Regression Head
        # Shape: (B, L, 1)
        pred = self.head(combined)

        # Remove last dim -> (B, L)
        return pred.squeeze(-1)


# ==========================================
# Training & Evaluation Logic
# ==========================================


def masked_mae_loss(pred, target, u_out):
    """
    Computes Mean Absolute Error only for the inspiratory phase (u_out == 0).
    The expiratory phase (u_out == 1) is ignored in the loss.
    """
    # Create mask: 1 for inspiratory, 0 for expiratory
    mask = 1 - u_out

    loss = F.l1_loss(pred, target, reduction="none")
    loss = loss * mask

    # Normalize by the number of valid time steps
    sum_mask = mask.sum()
    if sum_mask > 0:
        return loss.sum() / sum_mask
    else:
        return loss.sum()


def get_u_out_from_inputs(inputs):
    """
    Extracts the 'u_out' tensor from the skip connection inputs for masking.
    """
    try:
        idx = Config.SKIP_FEATURES.index("u_out")
    except ValueError:
        return None

    # inputs['skip'] is (B, L, C_skip)
    return inputs["skip"][:, :, idx]


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for inputs, targets in loader:
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        # Extract u_out for masking
        u_out = get_u_out_from_inputs(inputs)

        loss = masked_mae_loss(preds, targets, u_out)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = targets.to(device)

            preds = model(inputs)
            u_out = get_u_out_from_inputs(inputs)

            loss = masked_mae_loss(preds, targets, u_out)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def predict_test(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())

    # Concatenate: (N_batches, B, L) -> (N_total, L)
    return np.concatenate(all_preds, axis=0)


def run_training():
    print("Initializing Experiment Idea 4: Disentangled TCN-LSTM...")
    set_seed(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # Data Management
    dm = DataManager()

    # Loaders
    print("Preparing DataLoaders...")
    train_loader = dm.get_dataloader("train", shuffle=True)
    val_loader = dm.get_dataloader("validation", shuffle=False)

    # Model
    print("Initializing Model...")
    model = DisentangledTCNLSTM().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! (Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating predictions on Test set...")
    test_loader = dm.get_dataloader("test", shuffle=False)
    predictions = predict_test(model, test_loader, device)

    # Flatten predictions (N_breaths, 80) -> (N_rows,)
    flat_preds = predictions.flatten()

    # Create submission dataframe
    # We must ensure alignment with the sample submission ID order
    print("Aligning predictions with test IDs...")
    test_df = pd.read_csv(Config.TEST_PATH)
    test_df.sort_values(["breath_id", "time_step"], inplace=True)

    submission = pd.DataFrame({"id": test_df["id"], "pressure": flat_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


# Run the experiment
if __name__ == "__main__":
    run_training()
