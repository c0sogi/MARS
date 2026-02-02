import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import ModelConfig
from library.utils import set_seed, get_device, mcrmse_loss, save_checkpoint
from library.dataset import load_or_process_data, RNADataset
from library.model import RNARegressor


def run_training():
    """
    Executes the full training, validation, and inference pipeline.
    """
    # 1. Setup
    set_seed()
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(ModelConfig.output_dir, exist_ok=True)

    # 2. Data Loading
    print("Loading data...")
    train_data, val_data, test_data = load_or_process_data(load_cached_data=True)

    train_ds = RNADataset(train_data, mode="train")
    val_ds = RNADataset(val_data, mode="val")
    test_ds = RNADataset(test_data, mode="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNARegressor(config=ModelConfig).to(device)

    optimizer = AdamW(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=ModelConfig.num_epochs)

    # Loss function: MSE Loss (L2)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(ModelConfig.output_dir, "best_model.pth")

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(ModelConfig.num_epochs):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)  # (B, 68, 3)

            optimizer.zero_grad()

            # Forward pass
            # Output shape: (B, 107, 3)
            preds = model(seq, loop, dist, mask)

            # Masking: Only the first 68 positions are scored and have ground truth
            preds_scored = preds[:, :68, :]

            # Compute Loss
            loss = criterion(preds_scored, target)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        # Update scheduler
        scheduler.step()

        avg_train_loss = train_loss_accum / len(train_loader)

        # 5. Validation Loop
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)

                preds = model(seq, loop, dist, mask)
                preds_scored = preds[:, :68, :]

                val_preds_list.append(preds_scored)
                val_targets_list.append(target)

        # Concatenate for metric calculation
        val_preds_tensor = torch.cat(val_preds_list, dim=0)
        val_targets_tensor = torch.cat(val_targets_list, dim=0)

        # Calculate MCRMSE
        # Using the utility function which averages RMSE per column
        val_mcrmse = mcrmse_loss(val_targets_tensor, val_preds_tensor).item()

        print(
            f"Epoch {epoch+1}/{ModelConfig.num_epochs} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            save_checkpoint(model, optimizer, epoch, val_mcrmse, best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse}")

    # 6. Inference
    print("Generating submission...")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["id"]  # List of strings

            # Predict full length (107)
            preds = model(seq, loop, dist, mask)  # (B, 107, 3)
            preds_np = preds.cpu().numpy()

            batch_size = preds_np.shape[0]
            seq_len = preds_np.shape[1]

            for i in range(batch_size):
                sample_id = ids[i]
                sample_preds = preds_np[i]  # (107, 3)

                for pos in range(seq_len):
                    row_id = f"{sample_id}_{pos}"

                    # Map predictions to columns
                    # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
                    reactivity = sample_preds[pos, 0]
                    deg_Mg_pH10 = sample_preds[pos, 1]
                    deg_Mg_50C = sample_preds[pos, 2]

                    # Unscored columns filled with 0.0
                    deg_pH10 = 0.0
                    deg_50C = 0.0

                    submission_rows.append(
                        [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
                    )

    # 7. Save Submission
    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    os.makedirs(os.path.dirname(ModelConfig.submission_file), exist_ok=True)
    sub_df.to_csv(ModelConfig.submission_file, index=False)
    print(f"Submission saved to {ModelConfig.submission_file}")
