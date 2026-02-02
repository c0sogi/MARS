import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import RNADataset
from library.model import RNAMultiTaskBiGRU
from library.loss import JointAlignedLoss
from library.utils import set_seed, mcrmse, build_submission_df


def train_model(config=None):
    """
    Trains the RNA Multi-Task BiGRU model.

    Args:
        config (Config, optional): Configuration object. If None, uses default.

    Returns:
        float: The best validation MCRMSE score achieved.
    """
    if config is None:
        config = Config()

    # Ensure reproducibility
    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize Data Loaders
    print("Initializing datasets...")
    train_dataset = RNADataset("train", config=config)
    val_dataset = RNADataset("val", config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize Model, Loss, Optimizer
    model = RNAMultiTaskBiGRU(config).to(device)
    criterion = JointAlignedLoss(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    best_mcrmse = float("inf")
    model_save_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training loop...")

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        mse_loss_accum = 0.0
        ce_loss_accum = 0.0

        for batch in train_loader:
            # Move data to device
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_idx = batch["pair_idx"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            mask_labels = batch["mask_labels"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Forward pass
            reg_out, recon_out = model(seq, loop, pair_idx, pair_dist)

            # Calculate loss
            loss, mse, ce = criterion(reg_out, recon_out, targets, mask_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            mse_loss_accum += mse.item()
            ce_loss_accum += ce.item()

        scheduler.step()

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_mse = mse_loss_accum / len(train_loader)
        avg_ce = ce_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                pair_idx = batch["pair_idx"].to(device)
                pair_dist = batch["pair_dist"].to(device)
                targets = batch["targets"].to(device)

                # Forward pass (only regression output needed for val metric)
                reg_out, _ = model(seq, loop, pair_idx, pair_dist)

                # Extract scored columns and positions
                # targets shape: (B, 68, 5) -> select scored indices -> (B, 68, 3)
                scored_targets = targets[:, : config.SCORED_LEN, config.SCORED_INDICES]

                # preds shape: (B, 107, 3) -> select scored len -> (B, 68, 3)
                scored_preds = reg_out[:, : config.SCORED_LEN, :]

                all_preds.append(scored_preds.cpu().numpy())
                all_targets.append(scored_targets.cpu().numpy())

        # Concatenate for metric calculation
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Reshape to (N_samples * 68, 3) or keep as (N, 68, 3)
        # The mcrmse util expects (N, 3), so we flatten the sequence dimension
        flat_preds = all_preds.reshape(-1, 3)
        flat_targets = all_targets.reshape(-1, 3)

        val_mcrmse = mcrmse(flat_targets, flat_preds)

        # Print metrics
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"  Train Loss: {avg_train_loss} (MSE: {avg_mse}, CE: {avg_ce})")
        print(f"  Val MCRMSE: {val_mcrmse}")

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), model_save_path)
            print(f"  New best model saved to {model_save_path}")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")
    return best_mcrmse


def predict_and_submit(config=None):
    """
    Runs inference on the test set and generates a submission file.

    Args:
        config (Config, optional): Configuration object. If None, uses default.
    """
    if config is None:
        config = Config()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    print("Loading test data...")
    test_dataset = RNADataset("test", config=config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load Model
    print("Loading model...")
    model = RNAMultiTaskBiGRU(config).to(device)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using random weights.")

    model.eval()
    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            pair_idx = batch["pair_idx"].to(device)
            pair_dist = batch["pair_dist"].to(device)

            # Forward pass
            reg_out, _ = model(seq, loop, pair_idx, pair_dist)

            # Store predictions (keep on CPU)
            all_preds.append(reg_out.cpu().numpy())

    # Concatenate all batches: (N_test, 107, 3)
    all_preds = np.concatenate(all_preds, axis=0)

    # Generate Submission DataFrame
    print("Generating submission file...")
    ids = test_dataset.ids
    df_sub = build_submission_df(ids, all_preds, config)

    # Save
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
