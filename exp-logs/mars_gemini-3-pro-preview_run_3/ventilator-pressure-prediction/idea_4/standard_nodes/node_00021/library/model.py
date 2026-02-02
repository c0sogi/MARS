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


class RecurrentModel(nn.Module):
    """
    Monolithic Bidirectional LSTM (Cite solution_lesson_node_00014).
    Processes all features (dynamic + static) in a unified stream.
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.get_input_dim()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT,
        )

        # Head
        self.head = nn.Sequential(
            nn.Linear(Config.LSTM_HIDDEN_DIM * 2, Config.FC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.FC_HIDDEN_DIM, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Length, Features)

        # LSTM
        # Output: (Batch, Length, Hidden*2)
        out, _ = self.lstm(x)

        # Head
        pred = self.head(out)

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
