import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import calculate_rmse


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run training on.
        criterion (Loss): The loss function.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Model returns the denoised image (Input - Noise_Pred)
        outputs = model(inputs)

        # Compute loss between Denoised Output and Clean Target
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

    Returns:
        float: Global RMSE over the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs = inputs.to(device)
            # Targets are clean images

            # Inference on full images
            outputs = model(inputs)

            # Clamp outputs to valid range [0, 1]
            outputs = torch.clamp(outputs, 0, 1)

            # Move to CPU and flatten for global metric calculation
            # We accumulate all pixels to compute one global RMSE score
            all_preds.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.numpy().flatten())

    # Concatenate all pixels from all images
    global_preds = np.concatenate(all_preds)
    global_targets = np.concatenate(all_targets)

    # Calculate RMSE using the provided utility
    rmse = calculate_rmse(global_targets, global_preds)
    return rmse


def generate_submission(model, dataloader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The test data loader.
        device (torch.device): The device to run inference on.
        output_path (str): Path to save the submission CSV.
    """
    model.eval()
    results = []

    print("Generating submission...")

    with torch.no_grad():
        for inputs, _, img_ids in dataloader:
            inputs = inputs.to(device)

            # Inference
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            outputs_np = outputs.cpu().numpy()

            # Process each image in the batch
            for i, img_id in enumerate(img_ids):
                # Get the single channel image (H, W)
                img_pred = outputs_np[i, 0, :, :]
                h, w = img_pred.shape

                # Create coordinate grids (1-based indexing for submission)
                # np.indices returns (2, H, W) where [0] is rows, [1] is cols
                grid = np.indices((h, w))
                rows = grid[0].flatten() + 1
                cols = grid[1].flatten() + 1
                vals = img_pred.flatten()

                # Create IDs: {img_id}_{row}_{col}
                # Using list comprehension is efficient enough for this scale
                ids = [f"{img_id}_{r}_{c}" for r, c in zip(rows, cols)]

                # Create DataFrame for this image
                df_img = pd.DataFrame({"id": ids, "value": vals})

                results.append(df_img)

    # Concatenate all image dataframes
    if results:
        final_df = pd.concat(results, ignore_index=True)
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
    else:
        print("No results generated.")


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, patience
):
    """
    Main training loop with early stopping and model checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
    """
    criterion = nn.MSELoss()
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.10f} - Val RMSE: {val_rmse:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"New best model saved with RMSE: {best_rmse:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"Training complete. Best Val RMSE: {best_rmse:.10f}")
