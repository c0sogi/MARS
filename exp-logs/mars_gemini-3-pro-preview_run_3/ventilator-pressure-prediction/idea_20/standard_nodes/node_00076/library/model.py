import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import prepare_datasets, VentilatorDataset

# =========================================================================
# 0. Utils
# =========================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================================
# 1. Model Architecture: CWDH-Net
# =========================================================================


class DenseTCNBlock(nn.Module):
    """
    A single block of the Dense Large-Kernel TCN branch.
    Uses large kernels (9) with dense dilation (1) to capture high-fidelity local dynamics.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        # Calculate padding for 'same' output with dilation=1
        # padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # Projection for residual connection if dimensions change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.act2(out + residual)


class CWDHNet(nn.Module):
    """
    Corrected Wide-Context Dense-Hybrid Network.
    Combines a Dense TCN (Resistive Stream) and a BiLSTM (Elastic Stream).
    """

    def __init__(self):
        super().__init__()

        # --- Branch 1: Dense Large-Kernel TCN (Resistive) ---
        self.tcn_blocks = nn.ModuleList()
        in_c = Config.INPUT_DIM

        for out_c in Config.TCN_FILTERS:
            self.tcn_blocks.append(
                DenseTCNBlock(in_c, out_c, Config.TCN_KERNEL_SIZE, Config.TCN_DROPOUT)
            )
            in_c = out_c

        self.tcn_out_dim = Config.TCN_FILTERS[-1]

        # --- Branch 2: Bidirectional LSTM (Elastic) ---
        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT,
        )
        self.lstm_out_dim = Config.LSTM_HIDDEN_DIM * 2

        # --- Fusion Head ---
        self.fusion_in_dim = self.tcn_out_dim + self.lstm_out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(self.fusion_in_dim, Config.FUSION_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Config.FUSION_DIM, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # 1. TCN Path
        # TCN expects (batch, channels, seq_len)
        x_tcn = x.transpose(1, 2)
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        # Transpose back: (batch, seq_len, channels)
        x_tcn = x_tcn.transpose(1, 2)

        # 2. LSTM Path
        # LSTM expects (batch, seq_len, features)
        x_lstm, _ = self.lstm(x)

        # 3. Fusion
        # Concatenate along feature dimension
        x_cat = torch.cat([x_tcn, x_lstm], dim=2)

        # Project
        out = self.fusion_head(x_cat)

        # Remove last dim -> (batch, seq_len)
        return out.squeeze(-1)


# =========================================================================
# 2. Training Components
# =========================================================================


class MaskedMAELoss(nn.Module):
    """
    Computes L1 Loss only for the inspiratory phase (u_out == 0).
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")
        # Index of u_out in Config.CONT_FEATURES
        try:
            self.u_out_idx = Config.CONT_FEATURES.index("u_out")
        except ValueError:
            raise ValueError(
                "'u_out' must be present in Config.CONT_FEATURES for Masked Loss."
            )

    def forward(self, preds, targets, inputs):
        """
        preds: (batch, seq_len)
        targets: (batch, seq_len)
        inputs: (batch, seq_len, features)
        """
        # Extract u_out
        u_out = inputs[:, :, self.u_out_idx]

        # Create mask: 1 where u_out == 0 (Inspiratory), 0 otherwise
        mask = (u_out == 0).float()

        loss = self.l1(preds, targets)
        masked_loss = loss * mask

        # Avoid division by zero
        return masked_loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        preds = model(inputs)
        loss = criterion(preds, targets, inputs)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets, inputs)

            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


# =========================================================================
# 3. Main Execution Functions
# =========================================================================


def train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_x, train_y, val_x, val_y, _ = prepare_datasets(load_cached_data=True)

    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = CWDHNet().to(device)
    criterion = MaskedMAELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 3. Training Loop
    best_loss = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  -> New best model saved! Loss: {best_loss:.6f}")
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Loss: {best_loss:.6f}")


def generate_submission(batch_size=Config.BATCH_SIZE):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    # We only need test_x here, but prepare_datasets returns tuple
    _, _, _, _, test_x = prepare_datasets(load_cached_data=True)

    # Load test_ids from cache (created by prepare_datasets)
    test_ids_path = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    if not os.path.exists(test_ids_path):
        raise FileNotFoundError(
            "test_ids.npy not found. Run training/preparation first."
        )
    test_ids = np.load(test_ids_path)

    test_dataset = VentilatorDataset(test_x, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Load Model
    model = CWDHNet().to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            # Flatten predictions to 1D array
            predictions.append(preds.cpu().numpy().flatten())

    predictions = np.concatenate(predictions)

    # 3. Create Submission DataFrame
    # test_ids is (N_breaths, 80), flatten it to match predictions
    flat_ids = test_ids.flatten()

    # Safety check
    if len(flat_ids) != len(predictions):
        print(
            f"Warning: Length mismatch! IDs: {len(flat_ids)}, Preds: {len(predictions)}"
        )
        # Truncate to match (should not happen if logic is correct)
        min_len = min(len(flat_ids), len(predictions))
        flat_ids = flat_ids[:min_len]
        predictions = predictions[:min_len]

    submission_df = pd.DataFrame({"id": flat_ids, "pressure": predictions})

    # 4. Save
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run_pipeline():
    """
    Executes the full pipeline: Training -> Inference.
    """
    train_model()
    generate_submission()
