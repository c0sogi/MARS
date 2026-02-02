import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BACKGROUND_CLASS_WEIGHT,
    SMOOTHING_THRESHOLD,
    SMOOTHING_LOSS_WEIGHT,
    NUM_CLASSES,
    SEED,
)
from library.dataset import GestureDataset
from library.model import NRGSNet

# Ensure deterministic behavior
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class CustomLoss(nn.Module):
    """
    Cascaded Loss Function:
    1. Weighted NLL Loss for each stage (Deep Supervision).
    2. Log-Space Smoothing Loss (Truncated MSE) for temporal consistency.
    """

    def __init__(self, device):
        super(CustomLoss, self).__init__()

        # Class Weights: Down-weight background (index 0)
        weights = torch.ones(NUM_CLASSES, device=device)
        weights[0] = BACKGROUND_CLASS_WEIGHT
        self.nll_loss = nn.NLLLoss(weight=weights)

        self.smoothing_threshold = SMOOTHING_THRESHOLD
        self.smoothing_weight = SMOOTHING_LOSS_WEIGHT

    def smoothing_loss(self, log_probs):
        """
        Calculates Truncated MSE between adjacent frames in log-space.
        Input: (Batch, Time, Classes)
        """
        # Diff between t and t-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared error
        sq_diff = diff**2

        # Truncate (clamp) the error
        # We clamp the squared error to threshold^2
        truncated_sq_diff = torch.clamp(sq_diff, max=self.smoothing_threshold**2)

        return torch.mean(truncated_sq_diff)

    def forward(self, outputs, targets):
        """
        outputs: tuple (out_1, out_2, out_3) each of shape (B, T, C)
        targets: (B, T)
        """
        total_loss = 0.0

        # Flatten targets for NLLLoss: (B*T)
        targets_flat = targets.view(-1)

        for stage_out in outputs:
            # 1. Classification Loss
            # Reshape (B, T, C) -> (B*T, C)
            stage_out_flat = stage_out.reshape(-1, NUM_CLASSES)
            cls_loss = self.nll_loss(stage_out_flat, targets_flat)

            # 2. Smoothing Loss
            smooth_loss = self.smoothing_loss(stage_out)

            # Sum
            total_loss += cls_loss + (self.smoothing_weight * smooth_loss)

        return total_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: returns tuple (out1, out2, out3)
        outputs = model(inputs)

        # Calculate loss (Deep Supervision handled inside criterion)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy on the final stage (out3)
        final_output = outputs[-1]  # (B, T, C)
        _, predicted = torch.max(final_output, 2)

        # Mask out padding if necessary?
        # Dataset produces fixed windows, so we evaluate on the whole window.
        # However, accuracy on background class dominates.
        # We calculate global accuracy here.
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            final_output = outputs[-1]
            _, predicted = torch.max(final_output, 2)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc


def train_model(limit_data=None, load_cached_data=True):
    """
    Main training function.
    Args:
        limit_data (int): Optional limit for debugging.
        load_cached_data (bool): Whether to use cached features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Loading Training Data...")
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH,
        is_train=True,
        load_cached_data=load_cached_data,
        limit=limit_data,
    )

    print("Loading Validation Data...")
    val_dataset = GestureDataset(
        VAL_METADATA_PATH,
        is_train=False,
        load_cached_data=load_cached_data,
        limit=limit_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Initialize Model
    model = NRGSNet().to(device)

    # 3. Setup Optimizer and Loss
    # Using Adam as per plan (no AdamW)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CustomLoss(device)

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
            f"Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  Validation loss improved. Saved model.")
        else:
            patience_counter += 1
            print(
                f"  EarlyStopping counter: {patience_counter} out of {EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss}")

    return best_model_path
