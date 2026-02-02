import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.loss import MCRMSELoss
from library.model import DARDN
from library.data import get_loader


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch using the iterative refinement strategy.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass returns predictions from both passes
        # y_1: First pass (Zero Feedback)
        # y_2: Second pass (Feedback from y_1)
        y_1, y_2 = model(inputs, partner_indices)

        # Calculate weighted loss
        # MCRMSELoss handles the column selection and sequence slicing internally
        loss_final = criterion(y_2, targets)
        loss_aux = criterion(y_1, targets)

        loss = (Config.LOSS_WEIGHT_FINAL * loss_final) + (
            Config.LOSS_WEIGHT_AUX * loss_aux
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch} Training Loss: {avg_loss}")
    return avg_loss


def validate(model, loader, device):
    """
    Validates the model. Calculates Correct Global RMSE by accumulating SSE
    across the entire dataset to avoid batch-averaging bias.
    """
    model.eval()

    # Indices for scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # The target array has 5 columns, but we only score these 3.
    scored_indices = [0, 1, 3]

    # Accumulators
    total_sse = torch.zeros(len(scored_indices), device=device)
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass: We only care about the final refined prediction y_2
            _, y_2 = model(inputs, partner_indices)

            # 1. Slice Sequence Length (First 68 positions)
            pred_scored = y_2[:, : Config.SEQ_SCORED, :]
            true_scored = targets[:, : Config.SEQ_SCORED, :]

            # 2. Select Scored Columns
            pred_scored = pred_scored[:, :, scored_indices]
            true_scored = true_scored[:, :, scored_indices]

            # 3. Accumulate Squared Errors
            diff = pred_scored - true_scored
            # Sum over Batch (0) and Sequence (1) dimensions
            batch_sse = torch.sum(diff**2, dim=(0, 1))

            total_sse += batch_sse
            # Count total elements per column (BatchSize * SeqScored)
            total_count += pred_scored.shape[0] * pred_scored.shape[1]

    # Compute RMSE per column
    if total_count == 0:
        return float("inf")

    mse_per_col = total_sse / total_count
    rmse_per_col = torch.sqrt(mse_per_col)

    # MCRMSE is the mean of the column RMSEs
    global_mcrmse = torch.mean(rmse_per_col).item()

    print(f"Validation MCRMSE: {global_mcrmse}")
    return global_mcrmse


def train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Main execution function for training the DA-RDN model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Data (get_loader handles caching)
    train_loader = get_loader("train", batch_size=batch_size, shuffle=True)
    val_loader = get_loader("val", batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = DARDN().to(device)

    # Loss, Optimizer, Scheduler
    criterion = MCRMSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with MCRMSE: {best_score}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_score}")
    return best_score
