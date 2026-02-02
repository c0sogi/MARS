import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import BraTS25DEfficientNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        # Inputs are already (B, 64, H, W) due to dynamic sub-sampling in Dataset
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using a deterministic view (Even slices).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            # Input shape from val_loader is (B, 128, H, W) -> 4 modalities * 32 slices
            # We need to extract 16 slices (Even indices) to match model input (64 channels)
            B, C, H, W = inputs.shape
            # Reshape to (B, 4, 32, H, W)
            inputs = inputs.view(B, 4, 32, H, W)
            # Select even slices: indices 0, 2, ..., 30
            inputs = inputs[:, :, 0::2, :, :]
            # Flatten back to (B, 64, H, W)
            inputs = inputs.reshape(B, -1, H, W)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def predict_and_submit(model, loader, ids, device, output_path):
    """
    Generates predictions using Multi-View Ensemble (Even + Odd slices) and saves submission.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)

            # Input shape: (B, 128, H, W)
            B, C, H, W = inputs.shape
            inputs = inputs.view(B, 4, 32, H, W)

            # View A: Even slices (0, 2, ..., 30)
            view_a = inputs[:, :, 0::2, :, :].reshape(B, -1, H, W)

            # View B: Odd slices (1, 3, ..., 31)
            view_b = inputs[:, :, 1::2, :, :].reshape(B, -1, H, W)

            # Inference
            logits_a = model(view_a)
            logits_b = model(view_b)

            probs_a = torch.sigmoid(logits_a)
            probs_b = torch.sigmoid(logits_b)

            # Ensemble Average
            avg_probs = (probs_a + probs_b) / 2.0
            predictions.extend(avg_probs.cpu().numpy().flatten())

    # Create Submission DataFrame
    # ids is a numpy array of strings (e.g., "00013")
    # Sample submission expects BraTS21ID as int (e.g., 13)
    submission_df = pd.DataFrame(
        {"BraTS21ID": [int(pid) for pid in ids], "MGMT_value": predictions}
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    num_epochs=30,
    batch_size=8,
    patience=5,
    load_cached_data=True,
    learning_rate=1e-4,
    save_dir="./working/idea_3",
):
    """
    Main execution function for training, validation, and submission generation.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 2. Initialize Model & Optimizer
    model = BraTS25DEfficientNet(pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_auc = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best AUC! Model saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # 4. Generate Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    submission_path = "./submission/submission.csv"
    predict_and_submit(model, test_loader, test_ids, device, submission_path)

    print("Run complete.")
