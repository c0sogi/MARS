import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data, model


def train_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, partner_indices, targets, _) in enumerate(loader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass returns predictions from both passes
        # y1: Pass 1 (Zero Feedback)
        # y2: Pass 2 (Feedback from Pass 1)
        y1, y2 = model(inputs, partner_indices)

        # Calculate losses over full sequence length (anchoring)
        loss1 = utils.mcrmse_loss(y1, targets)
        loss2 = utils.mcrmse_loss(y2, targets)

        # Weighted total loss
        loss = loss2 + config.AUX_WEIGHT * loss1

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the global MCRMSE metric on scored positions.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets, _ in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            _, y2 = model(inputs, partner_indices)

            # Collect predictions and targets (move to CPU numpy)
            all_preds.append(y2.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric using the official global method
    metric = utils.calculate_global_mcrmse(all_preds, all_targets)

    return metric


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for inputs, partner_indices, _, sample_ids in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass - use y2 (refined prediction)
            _, y2 = model(inputs, partner_indices)

            all_preds.append(y2.cpu().numpy())
            all_ids.extend(sample_ids)

    # Shape: (N_samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare data for DataFrame
    submission_data = []

    for i, sample_id in enumerate(all_ids):
        pred_matrix = all_preds[i]  # (107, 5)

        for seq_pos in range(config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = pred_matrix[seq_pos]

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for t_idx, col_name in enumerate(config.TARGET_COLS):
                row_dict[col_name] = float(row_values[t_idx])

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + config.TARGET_COLS
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def train_model():
    """
    Main training routine.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing HC_HIDN model...")
    net = model.HC_HIDN().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(net, train_loader, optimizer, device)

        # Validate
        val_metric = validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step(val_metric)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_metric} | "  # Printing full precision
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Model Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(net.state_dict(), config.MODEL_PATH)
            print(f"  New best model saved! (MCRMSE: {best_metric})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Inference
    print("Loading best model for inference...")
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    generate_submission(net, test_loader, device)
    print("Done.")
