import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, WeightedL1Loss, compute_metric
from library.dataset import get_dataloaders


class DualPathInjectionBlock(nn.Module):
    """
    Generates an 'Injection Payload' via two parallel paths:
    1. Identity Path: Preserves raw physical magnitudes.
    2. Context Path: Gated Linear Unit (GLU) for non-linear feature extraction.
    """

    def __init__(self, input_dim, glu_dim):
        super(DualPathInjectionBlock, self).__init__()
        self.identity_path = nn.Identity()

        # GLU expects input size 2 * dim to split in half
        self.context_projection = nn.Linear(input_dim, glu_dim * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x):
        # Path A: Identity (B, L, Input_Dim)
        out_identity = self.identity_path(x)

        # Path B: Context (B, L, Glu_Dim)
        out_context = self.context_projection(x)
        out_context = self.glu(out_context)

        # Concatenate (B, L, Input_Dim + Glu_Dim)
        injection_payload = torch.cat([out_identity, out_context], dim=-1)
        return injection_payload


class DP_GI_BiLSTM(nn.Module):
    """
    Dual-Path Gated-Injection BiLSTM.
    Uses Deep Injection to feed the payload into every recurrent layer.
    """

    def __init__(self, input_dim):
        super(DP_GI_BiLSTM, self).__init__()

        self.hidden_size = Config.LSTM_HIDDEN_SIZE
        self.num_layers = Config.LSTM_LAYERS
        self.glu_dim = Config.GLU_PROJECTION_DIM

        # Injection Block
        self.injection_block = DualPathInjectionBlock(input_dim, self.glu_dim)
        self.injection_dim = input_dim + self.glu_dim

        # Recurrent Backbone (ModuleList for Deep Injection)
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # Input to layer 0 is just the injection payload
            # Input to layer k > 0 is (Previous_Output + Injection_Payload)
            if i == 0:
                layer_input_dim = self.injection_dim
            else:
                layer_input_dim = (2 * self.hidden_size) + self.injection_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Regularization applied to recurrent output only
            self.layer_norms.append(nn.LayerNorm(2 * self.hidden_size))
            self.dropouts.append(nn.Dropout(Config.DROPOUT))

        # Head
        self.head = nn.Linear(2 * self.hidden_size, 1)

    def forward(self, x):
        # Generate Injection Payload (B, L, Injection_Dim)
        injection_payload = self.injection_block(x)

        current_input = injection_payload
        prev_lstm_out = None  # Track output for residual connection

        for i in range(self.num_layers):
            # LSTM Forward
            # self.lstm_layers[i] returns (output, (h_n, c_n))
            lstm_out, _ = self.lstm_layers[i](current_input)

            # Add Residual Connection if not the first layer (Cite Lesson 21, 23)
            if prev_lstm_out is not None:
                lstm_out = lstm_out + prev_lstm_out

            # Apply Regularization to LSTM output
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.dropouts[i](lstm_out)

            # Store for next layer's residual
            prev_lstm_out = lstm_out

            # Prepare input for next layer (Concatenate Output + Injection)
            if i < self.num_layers - 1:
                current_input = torch.cat([lstm_out, injection_payload], dim=-1)
            else:
                current_input = lstm_out

        # Final Prediction (B, L, 1)
        # We use the output of the last LSTM layer (which is in current_input)
        preds = self.head(current_input)

        # Squeeze last dim to match target shape (B, L)
        return preds.squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets, u_out) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        u_out = u_out.to(device)

        optimizer.zero_grad()

        preds = model(inputs)

        loss = criterion(preds, targets, u_out)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets, u_out in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            u_out = u_out.to(device)

            preds = model(inputs)

            loss = criterion(preds, targets, u_out)
            mae = compute_metric(preds, targets, u_out)

            total_loss += loss.item()
            total_mae += mae
            count += 1

    return total_loss / count, total_mae / count


def train_model():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing DP-GI-BiLSTM on {device}...")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Model
    model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Stretched Horizon Scheduler
    # T_max matches total epochs to keep LR higher for longer
    epochs = Config.get_epochs()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)

    # Loss
    criterion = WeightedL1Loss(
        inspiratory_weight=Config.LOSS_INSPIRATORY_WEIGHT,
        expiratory_weight=Config.LOSS_EXPIRATORY_WEIGHT,
    )

    best_mae = float("inf")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.10f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! MAE: {best_mae:.10f}")

    print(f"Training complete. Best Validation MAE: {best_mae:.10f}")
    return model


def generate_submission():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print("Loading best model for inference...")
    model = DP_GI_BiLSTM(input_dim=Config.INPUT_DIM).to(device)

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Get test loader
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, _, _ in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)

            # Flatten predictions (B, L) -> (B*L)
            preds_flat = preds.view(-1).cpu().numpy()
            predictions.extend(preds_flat)

    predictions = np.array(predictions)

    # Load test metadata to map to IDs
    print("Mapping predictions to IDs...")
    test_meta = pd.read_csv(Config.TEST_META)

    # Ensure lengths match
    if len(predictions) != len(test_meta):
        # This might happen if the DataLoader dropped the last batch or padding occurred.
        # However, VentilatorDataset logic and drop_last=False for test should prevent this.
        # If there's a mismatch, we truncate or pad, but strictly we expect exact match.
        print(
            f"Warning: Prediction count {len(predictions)} != Metadata count {len(test_meta)}"
        )
        # In this specific dataset, id is continuous 1..N.

    submission = pd.DataFrame({"id": test_meta["id"], "pressure": predictions})

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission generated successfully.")


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing if needed.
    # The prompt asks to implement the module functions.
    pass


def run():
    """
    Main entry point for the execution script.
    """
    train_model()
    generate_submission()
