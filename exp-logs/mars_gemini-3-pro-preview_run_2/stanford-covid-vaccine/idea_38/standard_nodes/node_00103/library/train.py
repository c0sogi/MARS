import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import process_data, RNADataset
from library.model import DF_DCN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training using the Iterative Refinement strategy.

    Strategy:
    1. Compute static backbone features (Z).
    2. Pass 1: Predict Y_1 using Z and zero feedback.
    3. Pass 2: Predict Y_2 using Z and feedback from Y_1 (detached).
    4. Loss = Loss(Y_2) + 0.5 * Loss(Y_1).
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 1. Static Backbone (computed once per batch)
        # Shape: (N, L, LatentDim)
        z = model.forward_backbone(inputs)

        # 2. Pass 1: No Feedback (prev_preds=None implies zeros)
        preds_1 = model.forward_head(z, partner_indices, prev_preds=None)

        # 3. Pass 2: With Feedback
        # We detach preds_1 to stop gradients flowing back through the feedback generation process
        # This treats preds_1 as a fixed input for the second pass.
        preds_1_detached = preds_1.detach()
        preds_2 = model.forward_head(z, partner_indices, prev_preds=preds_1_detached)

        # 4. Loss Calculation
        # Calculate MCRMSE for both passes on scored targets
        loss_2 = criterion(preds_2, targets)
        loss_1 = criterion(preds_1, targets)

        # Weighted sum
        loss = (loss_2 * Config.LOSS_WEIGHT_PASS_2) + (
            loss_1 * Config.LOSS_WEIGHT_PASS_1
        )

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate loss (batch_mean * batch_size)
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Performs the full inference steps (Pass 1 -> Pass 2) and scores the final output.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            # Inference Logic
            # 1. Backbone
            z = model.forward_backbone(inputs)

            # 2. Pass 1
            preds_1 = model.forward_head(z, partner_indices, prev_preds=None)

            # 3. Pass 2 (Final Prediction)
            preds_2 = model.forward_head(z, partner_indices, prev_preds=preds_1)

            # Validation metric is based on the final output
            loss = criterion(preds_2, targets)

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_training():
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data Loading
    print("Loading and processing data...")
    # process_data handles caching internally
    train_data = process_data("train", load_cached_data=True)
    val_data = process_data("val", load_cached_data=True)

    # Debugging: Subset data if enabled
    if Config.DEBUG:
        print(f"DEBUG MODE: Trimming datasets to {Config.DEBUG_SUBSET_SIZE} samples.")
        for k in ["inputs", "partner_indices", "targets", "ids"]:
            train_data[k] = train_data[k][: Config.DEBUG_SUBSET_SIZE]
            val_data[k] = val_data[k][: Config.DEBUG_SUBSET_SIZE]

    train_dataset = RNADataset(train_data, mode="train")
    val_dataset = RNADataset(val_data, mode="val")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing DF-DCN Model...")
    model = DF_DCN().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.FACTOR,
        patience=Config.PATIENCE,
        min_lr=Config.MIN_LR,
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Training Loop...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print metrics (Full precision for Val Loss as requested)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New Best Model Saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.ES_PATIENCE}"
            )

        if patience_counter >= Config.ES_PATIENCE:
            print("Early Stopping Triggered.")
            break

    print(f"Training Complete. Best Val Loss: {best_val_loss}")
