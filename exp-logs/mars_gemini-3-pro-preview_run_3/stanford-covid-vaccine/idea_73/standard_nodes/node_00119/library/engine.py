import torch
import numpy as np
import os
from library.config import Hyperparameters
from library.utils import MCRMSELoss, metric_mcrmse_scored


def train_fn(model, data_loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    Iterates through the training DataLoader, computes loss, and updates weights.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in data_loader:
        # Move inputs and targets to device
        inputs = batch["inputs"].to(device)
        adjacency = batch["adjacency"].to(device)
        targets = batch["targets"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, adjacency)

        # Compute loss
        # MCRMSELoss handles slicing internally
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), Hyperparameters.GRADIENT_CLIP
        )

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    return avg_loss


def eval_fn(model, data_loader, device):
    """
    Evaluation loop.
    Iterates through the validation DataLoader, aggregates predictions,
    and computes the scored MCRMSE metric.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["inputs"].to(device)
            adjacency = batch["adjacency"].to(device)
            targets = batch["targets"]  # Keep targets on CPU for aggregation

            # Forward pass
            outputs = model(inputs, adjacency)

            # Move outputs to CPU
            outputs = outputs.cpu()

            all_preds.append(outputs)
            all_targets.append(targets)

    # Concatenate all batches
    # Preds: (Total_Samples, 107, 5)
    # Targets: (Total_Samples, 68, 5)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metric
    # metric_mcrmse_scored handles slicing internally
    score = metric_mcrmse_scored(all_preds, all_targets)

    return score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    model_save_path,
):
    """
    Orchestrates the training process with Early Stopping.
    """
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        # Training Step
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_score = eval_fn(model, val_loader, device)

        # Print metrics (Full precision)
        print(f"Epoch {epoch}: Train Loss = {train_loss}, Val MCRMSE = {val_score}")

        # Early Stopping & Model Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation score improved. Model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_score
