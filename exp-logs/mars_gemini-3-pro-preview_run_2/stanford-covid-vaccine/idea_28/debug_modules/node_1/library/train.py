import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config, set_seed, device
from library.data import get_dataset, RNADataset
from library.model import SRDN
from library.loss import MCRMSELoss


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    patience=Config.PATIENCE,
    load_cached_data=True,
    debug_sample_size=None,
):
    set_seed(42)

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {device}")

    # ==========================================================================
    # 1. Data Loading
    # ==========================================================================
    print("Loading datasets...")
    # Note: get_dataset handles caching internally as per library.data
    train_ids, train_inputs, train_pmaps, train_targets = get_dataset(
        "train", load_cached_data=load_cached_data
    )
    val_ids, val_inputs, val_pmaps, val_targets = get_dataset(
        "val", load_cached_data=load_cached_data
    )

    # Optional debugging
    if debug_sample_size:
        train_inputs = train_inputs[:debug_sample_size]
        train_pmaps = train_pmaps[:debug_sample_size]
        train_targets = train_targets[:debug_sample_size]
        val_inputs = val_inputs[:debug_sample_size]
        val_pmaps = val_pmaps[:debug_sample_size]
        val_targets = val_targets[:debug_sample_size]

    train_dataset = RNADataset(train_inputs, train_pmaps, train_targets)
    val_dataset = RNADataset(val_inputs, val_pmaps, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ==========================================================================
    # 2. Model Setup
    # ==========================================================================
    model = SRDN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )
    criterion = MCRMSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    early_stop_counter = 0

    # Mask for loss calculation (Scored positions only)
    # Shape: (1, L) -> broadcastable
    loss_mask = torch.zeros(Config.SEQ_LENGTH, device=device)
    loss_mask[: Config.SCORED_LENGTH] = 1.0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        for x, pmap, y in train_loader:
            x = x.to(device)  # (B, L, 19)
            pmap = pmap.to(device)  # (B, L)
            y = y.to(device)  # (B, L, 5)

            B, L, _ = x.shape

            # --- Pass 1: Cold Start ---
            # Initialize recycling channels to zero
            recycling_zero = torch.zeros(B, L, 5, device=device)
            x1 = torch.cat([x, recycling_zero], dim=2)  # (B, L, 24)

            pred1 = model(x1, pmap)

            # --- Pass 2: Refinement (Stabilized Recycling) ---
            # Detach pred1 to stop gradients flowing through the recycling loop
            recycling_detached = pred1.detach()
            x2 = torch.cat([x, recycling_detached], dim=2)  # (B, L, 24)

            pred2 = model(x2, pmap)

            # --- Loss Calculation ---
            # Expand mask for batch
            batch_mask = loss_mask.unsqueeze(0).expand(B, -1)

            loss_main = criterion(pred2, y, batch_mask)
            loss_aux = criterion(pred1, y, batch_mask)

            # Weighted sum: Main Loss + 0.5 * Auxiliary Loss
            loss = loss_main + 0.5 * loss_aux

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item() * B

        avg_train_loss = train_loss_accum / len(train_dataset)

        # ======================================================================
        # Validation (Global MCRMSE)
        # ======================================================================
        model.eval()
        total_sse = torch.zeros(
            3, device=device
        )  # For reactivity, deg_Mg_pH10, deg_Mg_50C
        total_count = 0

        # Scored indices defined in Config: [0, 1, 3]
        scored_indices = Config.SCORED_INDICES

        with torch.no_grad():
            for x, pmap, y in val_loader:
                x = x.to(device)
                pmap = pmap.to(device)
                y = y.to(device)

                B, L, _ = x.shape

                # Pass 1
                recycling = torch.zeros(B, L, 5, device=device)
                x_in = torch.cat([x, recycling], dim=2)
                pred1 = model(x_in, pmap)

                # Pass 2
                recycling = pred1.detach()
                x_in = torch.cat([x, recycling], dim=2)
                pred2 = model(x_in, pmap)

                # Select scored columns
                pred_scored = pred2[:, :, scored_indices]
                target_scored = y[:, :, scored_indices]

                # Calculate Squared Error
                sq_diff = (pred_scored - target_scored) ** 2

                # Apply Mask
                # batch_mask shape: (B, L). View as (B, L, 1) for broadcasting
                mask_expanded = loss_mask.view(1, -1, 1).expand(B, L, 3)
                sq_diff = sq_diff * mask_expanded

                # Sum errors over batch and sequence length
                total_sse += torch.sum(sq_diff, dim=(0, 1))

                # Count valid positions (same for all 3 columns)
                # sum(mask) gives valid positions per sequence (68). Multiply by Batch size.
                total_count += B * Config.SCORED_LENGTH

        # Compute RMSE per column: sqrt(SSE / N)
        rmse_per_col = torch.sqrt(total_sse / total_count)
        # Mean of RMSEs
        val_mcrmse = torch.mean(rmse_per_col).item()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse}"
        )

        scheduler.step(val_mcrmse)

        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! MCRMSE: {best_mcrmse}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Val MCRMSE: {best_mcrmse}")

    # ==========================================================================
    # 3. Submission Generation
    # ==========================================================================
    generate_submission(best_model_path, batch_size, load_cached_data)


def generate_submission(model_path, batch_size, load_cached_data):
    print("Generating submission...")

    # Load Test Data
    test_ids, test_inputs, test_pmaps = get_dataset(
        "test", load_cached_data=load_cached_data
    )
    test_dataset = RNADataset(test_inputs, test_pmaps)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load Model
    model = SRDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list = []

    with torch.no_grad():
        for x, pmap in test_loader:
            x = x.to(device)
            pmap = pmap.to(device)
            B, L, _ = x.shape

            # Pass 1
            recycling = torch.zeros(B, L, 5, device=device)
            x_in = torch.cat([x, recycling], dim=2)
            pred1 = model(x_in, pmap)

            # Pass 2
            recycling = pred1.detach()
            x_in = torch.cat([x, recycling], dim=2)
            pred2 = model(x_in, pmap)

            # Store predictions (B, L, 5)
            preds_list.append(pred2.cpu().numpy())

    all_preds = np.concatenate(preds_list, axis=0)  # (N_samples, 107, 5)

    # Format for submission
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    sub_ids = []
    sub_data = []

    # Target columns order in Config.TARGET_COLS matches the output of the model
    # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        for pos in range(Config.SEQ_LENGTH):
            sub_ids.append(f"{sample_id}_{pos}")
            sub_data.append(all_preds[i, pos])

    sub_data = np.array(sub_data)

    submission_df = pd.DataFrame(sub_data, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", sub_ids)

    save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    # This block is not required by the prompt but useful for local testing if run directly
    train_model()
