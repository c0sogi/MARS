import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library import config, utils, data, model


def get_scoring_mask(batch_size, device):
    """
    Generates a mask of shape (Batch, SeqLen) where the first 68 positions are 1.0
    and the rest are 0.0.
    """
    mask = torch.zeros((batch_size, config.SEQ_LEN), device=device)
    mask[:, : config.PRED_LEN] = 1.0
    return mask


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for x_static, partner_idx, y in loader:
        x_static = x_static.to(device)
        partner_idx = partner_idx.to(device)
        y = y.to(device)
        B = x_static.shape[0]

        # Pass 1: Initial guess with zero recycling
        x_recycled_1 = torch.zeros((B, config.SEQ_LEN, 5), device=device)
        # Note: We don't need gradients for Pass 1 input, but we need the graph for Pass 2
        pred_1 = model(x_static, x_recycled_1, partner_idx)

        # Pass 2: Refinement using Pass 1 output
        # We allow gradients to flow back through pred_1 to learn how to produce good initial guesses
        x_recycled_2 = pred_1
        pred_2 = model(x_static, x_recycled_2, partner_idx)

        # Calculate loss on the refined prediction
        mask = get_scoring_mask(B, device)
        loss = criterion(pred_2, y, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_static, partner_idx, y in loader:
            x_static = x_static.to(device)
            partner_idx = partner_idx.to(device)
            y = y.to(device)
            B = x_static.shape[0]

            # Pass 1: Initial guess
            x_recycled_1 = torch.zeros((B, config.SEQ_LEN, 5), device=device)
            pred_1 = model(x_static, x_recycled_1, partner_idx)

            # Pass 2: Refinement
            x_recycled_2 = pred_1
            pred_2 = model(x_static, x_recycled_2, partner_idx)

            mask = get_scoring_mask(B, device)
            loss = criterion(pred_2, y, mask)

            running_loss += loss.item()

    return running_loss / len(loader)


def run_training(debug=False, epochs=None):
    # Set seeds for reproducibility
    utils.set_seed(42)

    # Configuration
    device = config.DEVICE
    if epochs is None:
        epochs = config.EPOCHS

    # Load Data
    # If debug is True, we might want to limit the data size, but get_loaders doesn't natively support slicing
    # inside the function. We rely on the speed of the model or manual interruption if needed,
    # or we can assume the caller handles config changes.
    # For this implementation, we load standard data.
    train_loader, val_loader = data.get_loaders(load_cached_data=True)

    # Initialize Model, Optimizer, Loss
    net = model.RecurrentDenseNet().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = utils.MCRMSELoss()

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    save_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(net, train_loader, optimizer, criterion, device)
        val_loss = validate(net, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")


def generate_submission():
    utils.set_seed(42)
    device = config.DEVICE

    # Load Model
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    net = model.RecurrentDenseNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Load Test Data
    test_loader = data.get_test_loader(load_cached_data=True)

    all_preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for x_static, partner_idx, _ in test_loader:
            x_static = x_static.to(device)
            partner_idx = partner_idx.to(device)
            B = x_static.shape[0]

            # Pass 1
            x_recycled_1 = torch.zeros((B, config.SEQ_LEN, 5), device=device)
            pred_1 = net(x_static, x_recycled_1, partner_idx)

            # Pass 2
            x_recycled_2 = pred_1
            pred_2 = net(x_static, x_recycled_2, partner_idx)

            all_preds.append(pred_2.cpu().numpy())

    # Concatenate all batches: [N_samples, SeqLen, 5]
    all_preds = np.concatenate(all_preds, axis=0)

    # Load original test IDs to map back
    # We can reload the test csv or the cached data dict to get IDs
    test_data = data.process_data("test", load_cached_data=True)
    ids = test_data["ids"]

    submission_rows = []
    target_cols = (
        config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_pred = all_preds[i]  # [107, 5]

        for seqpos in range(config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    # Save
    out_path = config.SUBMISSION_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
