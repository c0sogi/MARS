import os
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import get_dataloader
from library.model import WideResBiGRU
from library.loss import MaskedMSELoss
from library.utils import seed_everything


def train_one_epoch(model, loader, optimizer, loss_fn, config):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    # Create mask for first 68 positions (pred_len)
    mask = torch.zeros(config.seq_len, device=config.device)
    mask[: config.pred_len] = 1.0

    for batch in loader:
        seq = batch["sequence"].to(config.device)
        loop = batch["loop"].to(config.device)
        dist = batch["pair_dist"].to(config.device)
        target = batch["target"].to(config.device)
        # Error targets are ignored

        optimizer.zero_grad()

        # Forward pass
        pred = model(seq, loop, dist)

        # Create batch mask
        batch_mask = mask.unsqueeze(0).expand(seq.size(0), -1)

        # Calculate Loss
        loss = loss_fn(pred, target, batch_mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, config):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["sequence"].to(config.device)
            loop = batch["loop"].to(config.device)
            dist = batch["pair_dist"].to(config.device)
            target = batch["target"].to(config.device)

            # Forward pass
            pred_val = model(seq, loop, dist)

            # Extract valid positions (0 to 68) for scoring
            valid_len = config.pred_len

            # Slice: [Batch, 68, 3]
            p = pred_val[:, :valid_len, :].cpu().numpy()
            t = target[:, :valid_len, :].cpu().numpy()

            all_preds.append(p)
            all_targets.append(t)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)  # [N_samples, 68, 3]
    all_targets = np.concatenate(all_targets, axis=0)

    # MCRMSE Calculation: Average of RMSEs per column
    # Flatten spatial dimensions to treat all valid positions as samples
    flat_preds = all_preds.reshape(-1, 3)
    flat_targets = all_targets.reshape(-1, 3)

    rmse_list = []
    for i in range(3):
        # Calculate MSE for this column
        mse = np.mean((flat_preds[:, i] - flat_targets[:, i]) ** 2)
        rmse = np.sqrt(mse)
        rmse_list.append(rmse)

    # Final metric is the mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_list)
    return mcrmse


def generate_submission(model, config):
    """
    Generates predictions for the test set and saves to CSV.
    """
    # Load test data
    test_loader = get_dataloader(mode="test", config=config, shuffle=False)

    # Retrieve IDs from the dataset
    dataset_ids = test_loader.dataset.ids

    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["sequence"].to(config.device)
            loop = batch["loop"].to(config.device)
            dist = batch["pair_dist"].to(config.device)

            # Forward pass
            pred_val = model(seq, loop, dist)
            preds_list.append(pred_val.cpu().numpy())

    # Concatenate: [N_test_samples, 107, 3]
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission rows
    submission_rows = []

    for i, sample_id in enumerate(dataset_ids):
        sample_pred = all_preds[i]  # Shape [107, 3]

        for j in range(config.seq_len):
            row_id = f"{sample_id}_{j}"

            # Model outputs: 0: reactivity, 1: deg_Mg_pH10, 2: deg_Mg_50C
            reactivity = float(sample_pred[j, 0])
            deg_Mg_pH10 = float(sample_pred[j, 1])
            deg_Mg_50C = float(sample_pred[j, 2])

            # Fill unscored columns with 0.0
            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": 0.0,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": 0.0,
                }
            )

    sub_df = pd.DataFrame(submission_rows)

    # Save to file
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)
    sub_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def run_training():
    """
    Main execution function.
    """
    # 1. Setup
    seed_everything(42)
    config = Config()
    os.makedirs(config.cache_dir, exist_ok=True)

    print(f"Starting training on {config.device}...")

    # 2. Data
    train_loader = get_dataloader(mode="train", config=config, shuffle=True)
    val_loader = get_dataloader(mode="val", config=config, shuffle=False)

    # 3. Model
    model = WideResBiGRU(config).to(config.device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = MaskedMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.cache_dir, "best_model.pth")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config)
        val_score = validate(model, val_loader, config)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Early Stopping / Save Best
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Val MCRMSE: {best_score}")

    # 6. Submission
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    generate_submission(model, config)
