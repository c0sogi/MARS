import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import library.config as cfg
import library.utils as utils


def train_fn(model, data_loader, optimizer, device, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        data_loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device to run on (CPU/GPU).
        criterion (nn.Module): The loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        # Regression target shape match: (B, 1) vs (B,)
        loss = criterion(outputs, labels.view(-1, 1))

        loss.backward()

        if cfg.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    return running_loss / dataset_size


def eval_fn(model, data_loader, device, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        data_loader (DataLoader): The validation data loader.
        device (torch.device): The device to run on.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (average_loss, qwk_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            # Collect targets and outputs for metric calculation
            final_targets.extend(labels.cpu().numpy().tolist())
            final_outputs.extend(outputs.detach().cpu().numpy().flatten().tolist())

    avg_loss = running_loss / dataset_size

    # Compute Quadratic Weighted Kappa
    score = utils.compute_score(final_targets, final_outputs)

    return avg_loss, score


def run_training(
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
    Runs the full training loop with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.

    Returns:
        float: Best validation score achieved.
    """
    # MSE Loss for regression
    criterion = nn.MSELoss()

    best_score = -np.inf
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_fn(model, train_loader, optimizer, device, criterion)
        val_loss, val_score = eval_fn(model, val_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs} - Time: {elapsed}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_score}")

        # Save best model based on QWK (maximize)
        if val_score > best_score:
            print(f"Validation Score Improved ({best_score} ---> {val_score})")
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            print(f"Model Saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation Score. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_score


def make_submission(model, test_loader, device, save_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader (returns image, id_code).
        device (torch.device): Device.
        save_path (str): Path to save the submission CSV.
    """
    model.eval()
    ids = []
    predictions = []

    with torch.no_grad():
        for images, id_codes in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Flatten outputs
            preds = outputs.detach().cpu().numpy().flatten()

            ids.extend(id_codes)
            predictions.extend(preds)

    # Post-process predictions: Clip to [0, 4] and round to nearest integer
    predictions = np.array(predictions)
    predictions = np.clip(predictions, 0, 4)
    predictions = np.round(predictions).astype(int)

    # Create DataFrame
    submission_df = pd.DataFrame({"id_code": ids, "diagnosis": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
