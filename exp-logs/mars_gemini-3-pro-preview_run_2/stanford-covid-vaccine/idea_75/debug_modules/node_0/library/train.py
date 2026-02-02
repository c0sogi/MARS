import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library import config, utils, loss, data, model


def train_one_epoch(model_instance, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using the iterative refinement strategy.

    Strategy:
    1. Pass 1: Forward pass with zero feedback.
    2. Pass 2: Forward pass using detached predictions from Pass 1 as feedback.
    3. Loss: Weighted sum of Pass 2 loss (1.0) and Pass 1 loss (0.5).
    """
    model_instance.train()
    running_loss = 0.0
    num_samples = 0

    for x, p_idx, y in loader:
        x = x.to(device)
        p_idx = p_idx.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()

        # --- Pass 1 (Static / Zero Feedback) ---
        # y_prev=None implies zero feedback
        pred1 = model_instance(x, p_idx, y_prev=None)
        loss1 = criterion(pred1, y)

        # --- Pass 2 (Iterative Refinement) ---
        # Use detached predictions from Pass 1 as feedback
        # The FeedbackModule inside the model handles channel masking
        pred2 = model_instance(x, p_idx, y_prev=pred1.detach())
        loss2 = criterion(pred2, y)

        # --- Combined Loss ---
        total_loss = (config.LOSS_PASS_2_WEIGHT * loss2) + (
            config.LOSS_PASS_1_WEIGHT * loss1
        )

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * batch_size
        num_samples += batch_size

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return epoch_loss


def validate(model_instance, loader, device):
    """
    Evaluates the model on the validation set using the two-pass inference strategy.
    Computes the Global MCRMSE (accumulating all predictions first).
    """
    model_instance.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, p_idx, y in loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            # y is kept on CPU for metric calculation to save GPU memory

            # Pass 1
            pred1 = model_instance(x, p_idx, y_prev=None)

            # Pass 2 (Final Prediction)
            pred2 = model_instance(x, p_idx, y_prev=pred1)

            all_preds.append(pred2.cpu().numpy())
            all_targets.append(y.numpy())

    # Concatenate all batches
    if not all_preds:
        return 0.0

    preds_concat = np.concatenate(all_preds, axis=0)
    targets_concat = np.concatenate(all_targets, axis=0)

    # Compute Global MCRMSE
    score = utils.compute_mcrmse(preds_concat, targets_concat)
    return score


def generate_submission(model_instance, loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    print("Generating submission...")
    model_instance.eval()
    all_preds = []

    with torch.no_grad():
        for x, p_idx, _ in loader:
            x = x.to(device)
            p_idx = p_idx.to(device)

            # Pass 1
            pred1 = model_instance(x, p_idx, y_prev=None)

            # Pass 2
            pred2 = model_instance(x, p_idx, y_prev=pred1)

            all_preds.append(pred2.cpu().numpy())

    # Shape: (N_samples, Seq_Len, 5)
    preds_concat = np.concatenate(all_preds, axis=0)

    # Load test metadata to get IDs
    test_csv_path = os.path.join(config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test metadata not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    ids = df_test["id"].values

    # Prepare data for submission DataFrame
    submission_data = []
    target_cols = (
        config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Iterate through each sample and each sequence position
    # preds_concat shape: (240, 107, 5)
    for i, sample_id in enumerate(ids):
        sample_preds = preds_concat[i]  # (107, 5)
        for seq_pos in range(config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_preds = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[col_idx]

            submission_data.append(row_dict)

    df_sub = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model():
    """
    Main function to orchestrate training, validation, and submission generation.
    """
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print(f"Using device: {device}")

    # 2. Data
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing HC-HSGFN Model...")
    net = model.HCHSGFN().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = loss.MaskedMCRMSELoss().to(device)

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(net, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(net, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(net.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Score: {best_score:.10f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  >>> No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Generate Submission
    print("Loading best model for submission...")
    net.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(net, test_loader, device, config.SUBMISSION_PATH)


def run():
    """
    Entry point function.
    """
    train_model()
