import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import seed_everything, get_device, WeightedL1Loss, compute_metric
from library.dataset import get_dataloaders
from library.model import HFSI_BiLSTM


def train_model(config: Config):
    """
    Trains the HFSI-BiLSTM model, performs validation, handles early stopping,
    and generates the submission file using the best checkpoint.
    """
    # 1. Setup
    seed_everything(config.SEED)
    device = get_device()

    # 2. Data Loading
    # get_dataloaders handles caching and processing internally
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(config)

    # Determine input dimension from a sample batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["input"].shape[-1]

    # 3. Model Initialization
    model = HFSI_BiLSTM(config, input_dim).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Stretched Horizon Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=1e-6
    )

    # Loss Function (Focus on Inspiratory Phase)
    criterion = WeightedL1Loss(
        w_insp=config.LOSS_WEIGHT_INSPIRATORY, w_exp=config.LOSS_WEIGHT_EXPIRATORY
    )

    # 5. Training Loop
    best_val_mae = float("inf")
    patience_counter = 0
    patience_limit = 50  # Generous patience for long-tail convergence

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            u_out = batch["u_out"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(inputs)

            # Squeeze last dim (B, L, 1) -> (B, L) to match targets
            preds_squeezed = preds.squeeze(-1)

            loss = criterion(preds_squeezed, targets, u_out)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds_list = []
        val_targets_list = []
        val_u_out_list = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
                u_out = batch["u_out"].to(device)

                preds = model(inputs)

                val_preds_list.append(preds.squeeze(-1).cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())
                val_u_out_list.append(u_out.cpu().numpy())

        # Concatenate for metric computation
        val_preds_arr = np.concatenate(val_preds_list)
        val_targets_arr = np.concatenate(val_targets_list)
        val_u_out_arr = np.concatenate(val_u_out_list)

        # Compute Inspiratory MAE
        val_mae = compute_metric(val_preds_arr, val_targets_arr, val_u_out_arr)

        # Step Scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Print Metrics (Full Precision for Val MAE)
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.6f} | Val MAE: {val_mae}"
        )

        # --- Checkpointing & Early Stopping ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation MAE: {best_val_mae}")

    # 6. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    test_preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            preds = model(inputs)
            test_preds_list.append(preds.squeeze(-1).cpu().numpy())

    # Flatten predictions and IDs for CSV format
    test_preds_flat = np.concatenate(test_preds_list).flatten()
    test_ids_flat = test_ids.flatten()

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids_flat, "pressure": test_preds_flat})

    # Sort by ID to ensure correct order
    submission_df.sort_values(by="id", inplace=True)

    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
