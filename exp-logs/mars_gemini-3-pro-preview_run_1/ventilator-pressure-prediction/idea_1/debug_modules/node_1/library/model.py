import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import MaskedL1Loss, seed_everything
from library.dataset import prepare_data


class BidirectionalLSTM(nn.Module):
    """
    Masked Bidirectional LSTM Model for Ventilator Pressure Prediction.

    Architecture:
    1. Input Projection: Linear -> LayerNorm -> ReLU
    2. Backbone: Stacked Bidirectional LSTM
    3. Head: Linear Projection to scalar pressure
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
    ):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Project input features to hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )

        # Recurrent Backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Regression Head
        # Input size is hidden_dim * 2 for bidirectional
        self.head = nn.Linear(hidden_dim * self.num_directions, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Seq_Len, Input_Dim)
        Returns:
            torch.Tensor: Predicted pressure of shape (Batch, Seq_Len)
        """
        # Project inputs
        # x shape: (B, L, input_dim) -> (B, L, hidden_dim)
        x = self.projection(x)

        # Pass through LSTM
        # lstm_out shape: (B, L, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)

        # Project to scalar output
        # pred shape: (B, L, 1)
        pred = self.head(lstm_out)

        # Squeeze last dimension to match target shape (B, L)
        return pred.squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Calculate masked loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate loss (weighted by batch size)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)
            loss = criterion(preds, y, u_out)

            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def fit(model, train_loader, val_loader, config):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    device = config.DEVICE
    model.to(device)

    criterion = MaskedL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print(f"  Model saved to {config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        print("Loaded best model weights.")

    return model


def predict_and_submit(model, test_loader, config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = config.DEVICE
    model.to(device)
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            # Forward pass
            preds = model(x)
            # Flatten predictions to match sample submission format
            all_preds.append(preds.cpu().numpy().flatten())

    # Concatenate all batches
    flat_preds = np.concatenate(all_preds)

    # Load sample submission to ensure correct IDs
    if not os.path.exists(config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Sample submission not found at {config.SAMPLE_SUBMISSION_PATH}"
        )

    submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)

    # Safety check for length
    if len(flat_preds) != len(submission):
        print(
            f"Warning: Prediction length {len(flat_preds)} does not match submission length {len(submission)}."
        )

    submission["pressure"] = flat_preds

    # Save submission
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run():
    """
    Orchestration function to run the full pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Data Preparation
    train_loader, val_loader, test_loader = prepare_data(load_cached_data=True)

    # 3. Model Initialization
    model = BidirectionalLSTM(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        bidirectional=Config.BIDIRECTIONAL,
    )

    # 4. Training
    model = fit(model, train_loader, val_loader, Config)

    # 5. Inference & Submission
    predict_and_submit(model, test_loader, Config)
