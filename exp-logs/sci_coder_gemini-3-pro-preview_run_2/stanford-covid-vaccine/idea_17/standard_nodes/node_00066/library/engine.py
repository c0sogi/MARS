import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import MCRMSELoss, MetricTracker


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    criterion = MCRMSELoss()

    # Iterate over the loader
    # Loader yields: inputs, partner_indices, targets, ids (optional)
    for i, batch in enumerate(loader):
        # Unpack batch (handle potential variable length if ids are/aren't present)
        inputs = batch[0].to(device)
        partner_indices = batch[1].to(device)
        targets = batch[2].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, partner_indices)

        # Compute Loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using global MCRMSE.
    """
    model.eval()
    criterion = MCRMSELoss()
    tracker = MetricTracker()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            partner_indices = batch[1].to(device)
            targets = batch[2].to(device)

            outputs = model(inputs, partner_indices)

            # Compute batch loss for logging purposes
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Update global metric tracker
            tracker.update(outputs, targets)

    avg_loss = running_loss / len(loader)
    global_mcrmse = tracker.compute()

    return avg_loss, global_mcrmse


def predict(model, loader, device):
    """
    Generates predictions for the entire dataset in the loader.
    Returns:
        preds: Numpy array of shape (N_samples, Seq_Len, 5)
        ids: List of sample IDs
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            partner_indices = batch[1].to(device)
            # batch[2] is targets (dummy or real), batch[3] is ids
            ids = batch[3]

            outputs = model(inputs, partner_indices)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0)
    return preds, all_ids


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Scheduler Step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mcrmse)
            else:
                scheduler.step()

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model. MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")
    return best_mcrmse


def generate_submission_csv(preds, ids, output_path):
    """
    Formats predictions into the competition submission CSV format.

    Args:
        preds: Numpy array (N_samples, Seq_Len, 5)
        ids: List of N_samples IDs
        output_path: Path to save the CSV
    """
    # Target columns in order
    target_cols = Config.TARGET_COLS

    data_rows = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (Seq_Len, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos].tolist()

            # Create row dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            data_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(data_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
