import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed, FocalLoss, optimize_mcc_threshold
from library.data_processing import process_data
from library.dataset import get_dataloader
from library.models import SRVNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x_kin, x_vis, y in dataloader:
        x_kin = x_kin.to(device)
        x_vis = x_vis.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_kin, x_vis)

        # Loss calculation
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, best MCC score, and the threshold used.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x_kin, x_vis, y in dataloader:
            x_kin = x_kin.to(device)
            x_vis = x_vis.to(device)
            y = y.to(device)

            logits = model(x_kin, x_vis)
            loss = criterion(logits, y)

            running_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    all_logits = torch.cat(all_logits).numpy()
    all_targets = (
        torch.cat(all_targets).numpy().flatten()
    )  # Targets are (N, 1), flatten to (N,)

    # Convert logits to probabilities for MCC optimization
    # Sigmoid: 1 / (1 + exp(-x))
    all_probs = 1.0 / (1.0 + np.exp(-all_logits)).flatten()

    # Calculate metrics
    best_threshold, best_mcc = optimize_mcc_threshold(all_targets, all_probs)

    return avg_loss, best_mcc, best_threshold


def train_model(load_cached_data=True):
    """
    Main function to train the SRV-Net model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Training Data...")
    X_kin_train, X_vis_train, y_train, _ = process_data(
        "train", load_cached_data=load_cached_data
    )

    print("Loading Validation Data...")
    X_kin_val, X_vis_val, y_val, _ = process_data(
        "validation", load_cached_data=load_cached_data
    )

    # Create Dataloaders
    train_loader = get_dataloader(
        X_kin_train, X_vis_train, y_train, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        X_kin_val, X_vis_val, y_val, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # 3. Model Initialization
    model = SRVNet(
        input_dim_kin=Config.INPUT_DIM_KINEMATIC,
        input_dim_vis=Config.INPUT_DIM_VISUAL,
        kinematic_hidden_dims=Config.KINEMATIC_HIDDEN_DIMS,
        visual_hidden_dims=Config.VISUAL_HIDDEN_DIMS,
        dropout_rate=Config.DROPOUT_RATE,
        lambda_visual=Config.LAMBDA_VISUAL,
    ).to(device)

    # 4. Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Using Focal Loss as per design
    criterion = FocalLoss(gamma=Config.FOCAL_LOSS_GAMMA)

    # 5. Training Loop
    best_val_mcc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_mcc, val_thresh = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCC: {val_mcc} (at threshold {val_thresh})"
        )

        # Early Stopping Check
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MCC: {best_val_mcc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    return best_val_mcc
