import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNARegressor


class MCRMSELoss(nn.Module):
    """
    Differentiable MCRMSE Loss.
    Calculates the Mean Columnwise Root Mean Squared Error.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (Batch, Seq_Len, 5)
        # Calculate MSE per column (averaging over Batch and Seq_Len)
        mse = torch.mean((y_pred - y_true) ** 2, dim=(0, 1))
        # Add epsilon for numerical stability
        rmse = torch.sqrt(mse + 1e-8)
        # Average RMSE across the 5 targets
        return torch.mean(rmse)


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        bpp_mask = batch["bpp_mask"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        outputs = model(inputs, bpp_indices, bpp_mask)

        # Slice to scored sequence length (68) for loss calculation
        # This prevents the model from learning the zero-padding in the tails
        outputs_sliced = outputs[:, : config.seq_scored, :]
        targets_sliced = targets[:, : config.seq_scored, :]

        loss = criterion(outputs_sliced, targets_sliced)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, bpp_indices, bpp_mask)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate metric using the utility function which handles slicing
    score = mcrmse_metric(y_true, y_pred, seq_scored=config.seq_scored)
    return score


def generate_submission(model, loader, device, config):
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, bpp_indices, bpp_mask)  # (B, 107, 5)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(ids)

    preds_arr = np.concatenate(preds_list, axis=0)  # (N, 107, 5)

    # Prepare submission dataframe
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_arr[i]  # (107, 5)
        for seqpos in range(config.seq_length):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()
            submission_data.append([row_id] + row_values)

    submission_df = pd.DataFrame(submission_data, columns=["id_seqpos"] + target_cols)
    return submission_df


def run_training(config=None):
    if config is None:
        config = Config()

    set_seed(config.seed)

    # DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model
    model = RNARegressor(config).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Loss
    criterion = MCRMSELoss()

    best_val_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.device}...")

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device, config
        )
        val_score = validate(model, val_loader, config.device, config)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.num_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Checkpoint and Early Stopping
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Score: {best_val_score}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))

    submission_df = generate_submission(model, test_loader, config.device, config)
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
