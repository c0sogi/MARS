import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import PATHS, MODEL_PARAMS, TRAIN_PARAMS, DATA_CONFIG
from library.utils import set_seed, get_device, MCRMSELoss
from library.data import get_dataloaders
from library.model import SR_DCN


def train_model():
    # 1. Setup
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Create working directory if it doesn't exist (handled by config, but double check)
    os.makedirs(os.path.dirname(PATHS["MODEL_SAVE"]), exist_ok=True)

    # 2. Data Loading
    # debug=TRAIN_PARAMS["debug"] controls if we use a small subset
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=TRAIN_PARAMS["debug"], load_cached_data=True
    )

    # 3. Model Initialization
    model = SR_DCN().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS["learning_rate"],
        weight_decay=TRAIN_PARAMS["weight_decay"],
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Loss function (scores specific columns: reactivity, deg_Mg_pH10, deg_Mg_50C)
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {TRAIN_PARAMS['num_epochs']} epochs...")

    for epoch in range(TRAIN_PARAMS["num_epochs"]):
        model.train()
        train_loss_accum = 0.0

        start_time = time.time()

        for batch_idx, (inputs, partner_indices, targets, masks, ids) in enumerate(
            train_loader
        ):
            inputs = inputs.to(device)  # (B, L, 18)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            batch_size, seq_len, _ = inputs.shape

            # --- Pass 1: Cold Start ---
            # Initialize recycling channels to zero
            recycling_zero = torch.zeros(
                batch_size, seq_len, MODEL_PARAMS["num_targets"]
            ).to(device)

            # Concatenate: (B, L, 18) + (B, L, 5) -> (B, L, 23)
            input_pass1 = torch.cat([inputs, recycling_zero], dim=2)

            # Forward Pass 1
            preds_1 = model(input_pass1, partner_indices)

            # --- Pass 2: Refinement ---
            # Detach predictions from pass 1 to stop gradients flowing back through the recycling loop
            # This implements the "Stabilized Recycling" strategy
            recycling_pass2 = preds_1.detach()

            # Concatenate inputs with detached predictions
            input_pass2 = torch.cat([inputs, recycling_pass2], dim=2)

            # Forward Pass 2
            preds_2 = model(input_pass2, partner_indices)

            # --- Loss Calculation ---
            # Primary loss on refined predictions
            loss_2 = criterion(preds_2, targets, masks)

            # Auxiliary loss on initial predictions (to guide the cold start)
            loss_1 = criterion(preds_1, targets, masks)

            # Total loss
            loss = loss_2 + 0.5 * loss_1

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        val_mcrmse = validate(model, val_loader, device, criterion)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Scheduler Step
        scheduler.step(val_mcrmse)

        # Early Stopping & Checkpointing
        if val_mcrmse < best_val_loss:
            best_val_loss = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), PATHS["MODEL_SAVE"])
            print(f"  New best model saved! (MCRMSE: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_PARAMS["patience"]:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation MCRMSE: {best_val_loss:.6f}")

    # 6. Generate Submission
    generate_submission(test_loader, device)


def validate(model, dataloader, device, criterion):
    """
    Validates the model using the two-stage inference process.
    Computes the correct Global MCRMSE.
    """
    model.eval()

    # Accumulators for global RMSE calculation
    # We track SSE (Sum Squared Error) and Count for each scored column
    # Scored columns indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    scored_indices = criterion.scored_indices
    num_scored = len(scored_indices)

    total_sse = torch.zeros(num_scored).to(device)
    total_count = torch.zeros(num_scored).to(device)

    with torch.no_grad():
        for inputs, partner_indices, targets, masks, ids in dataloader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            batch_size, seq_len, _ = inputs.shape

            # --- Pass 1: Cold Start ---
            recycling_zero = torch.zeros(
                batch_size, seq_len, MODEL_PARAMS["num_targets"]
            ).to(device)
            input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
            preds_1 = model(input_pass1, partner_indices)

            # --- Pass 2: Refinement ---
            recycling_pass2 = (
                preds_1  # No need to detach in no_grad mode, but conceptually same
            )
            input_pass2 = torch.cat([inputs, recycling_pass2], dim=2)
            preds_2 = model(input_pass2, partner_indices)

            # --- Metric Accumulation ---
            # Filter for scored columns
            preds_scored = preds_2[:, :, scored_indices]
            targets_scored = targets[:, :, scored_indices]

            # Apply mask
            mask_bool = masks.bool()  # (B, L)

            # We iterate over the scored columns to sum errors correctly
            for i in range(num_scored):
                p_col = preds_scored[:, :, i]
                t_col = targets_scored[:, :, i]

                # Select valid positions
                valid_p = p_col[mask_bool]
                valid_t = t_col[mask_bool]

                sse = torch.sum((valid_p - valid_t) ** 2)
                count = valid_p.numel()

                total_sse[i] += sse
                total_count[i] += count

    # Compute RMSE per column
    rmse_per_col = torch.sqrt(total_sse / (total_count + 1e-8))

    # Mean of RMSEs
    global_mcrmse = torch.mean(rmse_per_col).item()

    return global_mcrmse


def generate_submission(dataloader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Generating submission...")

    # Load best model
    model = SR_DCN().to(device)
    model.load_state_dict(torch.load(PATHS["MODEL_SAVE"], map_location=device))
    model.eval()

    results = []

    # Columns required in submission
    target_cols = DATA_CONFIG[
        "target_cols"
    ]  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    with torch.no_grad():
        for inputs, partner_indices, _, masks, ids in dataloader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            batch_size, seq_len, _ = inputs.shape

            # --- Pass 1 ---
            recycling_zero = torch.zeros(
                batch_size, seq_len, MODEL_PARAMS["num_targets"]
            ).to(device)
            input_pass1 = torch.cat([inputs, recycling_zero], dim=2)
            preds_1 = model(input_pass1, partner_indices)

            # --- Pass 2 ---
            recycling_pass2 = preds_1
            input_pass2 = torch.cat([inputs, recycling_pass2], dim=2)
            preds_2 = model(input_pass2, partner_indices)

            # preds_2 shape: (B, L, 5)
            preds_np = preds_2.cpu().numpy()

            for i in range(batch_size):
                sample_id = ids[i]
                sample_preds = preds_np[i]  # (L, 5)

                for seqpos in range(seq_len):
                    # Format: id_seqpos
                    row_id = f"{sample_id}_{seqpos}"

                    # Get values for the 5 columns
                    vals = sample_preds[seqpos]

                    row_dict = {"id_seqpos": row_id}
                    for col_idx, col_name in enumerate(target_cols):
                        row_dict[col_name] = float(vals[col_idx])

                    results.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure column order
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(PATHS["SUBMISSION"], index=False)
    print(f"Submission saved to {PATHS['SUBMISSION']}")
