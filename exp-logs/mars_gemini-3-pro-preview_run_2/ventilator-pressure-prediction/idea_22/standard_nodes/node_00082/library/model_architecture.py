import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MetricMonitor
from library.data_factory import get_dataloaders


class InjectionBlock(nn.Module):
    """
    Physics-Augmented Injection Block.
    Constructs a payload from two parallel paths:
    Path A: Identity (Raw features + Explicit Physics terms)
    Path B: Wide Monolithic GLU projected to a Compressed Bottleneck.
    """

    def __init__(self, input_dim, bottleneck_dim, hidden_dim=512):
        super().__init__()
        # Path A is Identity (handled in forward pass via concatenation)

        # Path B: Wide GLU -> Bottleneck
        # "Wide" implies expanding to the hidden dimension size (or similar width)
        # GLU halves the dimension, so we project to hidden_dim * 2 first.
        self.fc_glu = nn.Linear(input_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.fc_bottleneck = nn.Linear(hidden_dim, bottleneck_dim)

    def forward(self, x):
        # x: (batch, seq, input_dim)

        # Path B Processing
        path_b = self.fc_glu(x)
        path_b = self.glu(path_b)
        path_b = self.fc_bottleneck(path_b)

        # Fusion: Concatenate Path A (x) and Path B (bottleneck)
        # No dropout here as per design (stable ground truth)
        return torch.cat([x, path_b], dim=-1)


class PADIBiLSTM(nn.Module):
    """
    Physics-Augmented Dual-Injection BiLSTM.
    Features:
    - Deep Injection: Payload fed to every LSTM layer.
    - Wide Deep Backbone: 4 layers, 512 hidden units.
    - Inter-Layer Connectivity: LayerNorm + Dropout between layers.
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.INPUT_DIM
        hidden_size = Config.LSTM_HIDDEN_SIZE
        num_layers = Config.LSTM_NUM_LAYERS
        bottleneck_dim = Config.INJECTION_BOTTLENECK_DIM
        dropout_p = Config.LSTM_DROPOUT

        # Injection Block
        self.injection = InjectionBlock(
            input_dim, bottleneck_dim, hidden_dim=hidden_size
        )

        # The payload dimension is Input + Bottleneck
        self.payload_dim = input_dim + bottleneck_dim

        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            # Determine input size for this layer
            # Layer 0: Receives just the Injection Payload
            # Layer >0: Receives (Previous Layer Output) + (Injection Payload)
            if i == 0:
                layer_input_dim = self.payload_dim
            else:
                # BiLSTM output is hidden_size * 2
                layer_input_dim = (hidden_size * 2) + self.payload_dim

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Add Norm and Dropout between layers (not after the last one)
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
                self.dropouts.append(nn.Dropout(dropout_p))

        # Regression Head
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (batch, seq, input_dim)

        # 1. Generate Injection Payload
        payload = self.injection(x)  # (batch, seq, payload_dim)

        current_input = payload

        # 2. Iterate through Deep Recurrent Backbone
        for i, lstm in enumerate(self.lstm_layers):
            # Forward pass through LSTM layer
            lstm_out, _ = lstm(current_input)

            if i < len(self.lstm_layers) - 1:
                # Apply Inter-Layer Connectivity
                lstm_out = self.layer_norms[i](lstm_out)
                lstm_out = self.dropouts[i](lstm_out)

                # Deep Injection: Concatenate payload to current output for the next layer
                current_input = torch.cat([lstm_out, payload], dim=-1)
            else:
                # Final layer output
                current_input = lstm_out

        # 3. Projection
        pred = self.head(current_input)  # (batch, seq, 1)
        return pred.squeeze(-1)


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss.
    Assigns weight 1.0 to Inspiratory phase (u_out=0) and 0.1 to Expiratory phase (u_out=1).
    """

    def __init__(self):
        super().__init__()
        self.w_insp = Config.LOSS_WEIGHT_INSPIRATORY
        self.w_exp = Config.LOSS_WEIGHT_EXPIRATORY
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        raw_loss = self.l1(pred, target)

        # u_out is 0 (Inspiratory) or 1 (Expiratory)
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        weighted_loss = raw_loss * weights
        return weighted_loss.mean()


def train_and_predict(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Executes the training pipeline and generates submission.

    Args:
        debug (bool): If True, uses a small subset of data.
        epochs (int): Number of training epochs.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model Initialization
    model = PADIBiLSTM().to(device)

    # 4. Optimizer & Scheduler (Stretched-Horizon Protocol)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)
    criterion = WeightedL1Loss()

    # 5. Training Loop
    best_mae = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_monitor = MetricMonitor()

        for batch in train_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y, u_out)
            loss.backward()
            optimizer.step()

            train_monitor.update("Loss", loss.item(), X.size(0))

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_monitor = MetricMonitor()

        with torch.no_grad():
            for batch in val_loader:
                X = batch["X"].to(device)
                y = batch["y"].to(device)
                u_out = batch["u_out"].to(device)

                pred = model(X)

                # Metric: MAE on Inspiratory Phase (u_out == 0)
                mask = u_out == 0
                if mask.sum() > 0:
                    mae = torch.abs(pred[mask] - y[mask]).mean()
                    val_monitor.update("MAE_Insp", mae.item(), mask.sum().item())

                val_loss = criterion(pred, y, u_out)
                val_monitor.update("Loss", val_loss.item(), X.size(0))

        current_mae = val_monitor.get_avg("MAE_Insp")
        print(
            f"Epoch {epoch}: Train Loss: {train_monitor.get_avg('Loss'):.6f} | Val Loss: {val_monitor.get_avg('Loss'):.6f} | Val MAE Insp: {current_mae:.6f}"
        )

        # Save Best Model
        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MAE: {best_mae:.6f}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy().flatten())

    all_preds = np.concatenate(predictions)

    # 7. Submission Generation
    # Load metadata to map predictions to IDs
    test_meta = pd.read_csv(Config.TEST_METADATA)

    # Safety check
    if len(all_preds) != len(test_meta):
        print(
            f"Warning: Prediction length {len(all_preds)} does not match Metadata length {len(test_meta)}"
        )

    test_meta["pressure"] = all_preds

    # Format: id, pressure
    submission = test_meta[["id", "pressure"]]
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
