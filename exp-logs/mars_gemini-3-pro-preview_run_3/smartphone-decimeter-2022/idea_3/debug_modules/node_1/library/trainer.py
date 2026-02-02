import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config, utils, model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["lengths"]

        # Forward pass
        # lengths must be on CPU for pack_padded_sequence if used inside model
        outputs = model(features, lengths)

        # Compute masked loss
        # outputs: (B, L, 2), targets: (B, L, 2), mask: (B, L)
        # We only care about valid time steps
        loss_unreduced = torch.abs(outputs - targets)  # L1 Loss

        # Expand mask to match output dimensions (B, L, 2)
        mask_expanded = mask.unsqueeze(-1).expand_as(loss_unreduced)

        # Apply mask
        masked_loss = loss_unreduced * mask_expanded.float()

        # Average over valid elements
        # Count of valid elements = sum(mask) * 2 (lat and lon)
        valid_elements = mask.sum() * 2
        loss = masked_loss.sum() / (valid_elements + 1e-8)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * mask.sum().item()  # track total loss sum
        total_samples += mask.sum().item()  # track total valid time steps

    epoch_loss = running_loss / total_samples
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set and computes the competition metric.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    # Containers for metric calculation
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            lengths = batch["lengths"]

            trip_ids = batch["tripIds"]
            timestamps_list = batch["timestamps"]
            wls_pos_list = batch["wls_pos"]

            outputs = model(features, lengths)

            # Compute masked loss for monitoring
            loss_unreduced = torch.abs(outputs - targets)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss_unreduced)
            masked_loss = loss_unreduced * mask_expanded.float()
            valid_elements = mask.sum() * 2
            loss = masked_loss.sum() / (valid_elements + 1e-8)

            running_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

            # Reconstruct predictions for metric calculation
            # Iterate over batch elements to handle variable lengths
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            for i in range(len(trip_ids)):
                length = lengths[i]
                trip_id = trip_ids[i]

                # Extract valid sequence
                pred_residuals = outputs_np[i, :length, :]  # (L, 2) -> dLat, dLon
                target_residuals = targets_np[i, :length, :]  # (L, 2)

                # Get metadata
                ts = timestamps_list[i]  # (L,)
                wls = wls_pos_list[i]  # (L, 2) -> lat_wls, lon_wls

                # Reconstruct Predictions: P_pred = P_wls + P_res
                lat_pred = wls[:, 0] + pred_residuals[:, 0]
                lon_pred = wls[:, 1] + pred_residuals[:, 1]

                # Reconstruct Ground Truth: P_gt = P_wls + T_res
                # (Or we could have passed GT directly, but this ensures alignment)
                lat_gt = wls[:, 0] + target_residuals[:, 0]
                lon_gt = wls[:, 1] + target_residuals[:, 1]

                # Create DataFrame chunks
                df_p = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": ts,
                        "LatitudeDegrees": lat_pred,
                        "LongitudeDegrees": lon_pred,
                    }
                )
                all_preds.append(df_p)

                df_g = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": ts,
                        "LatitudeDegrees": lat_gt,
                        "LongitudeDegrees": lon_gt,
                    }
                )
                all_gts.append(df_g)

    val_loss = running_loss / total_samples

    # Concatenate all chunks
    if all_preds:
        df_pred_all = pd.concat(all_preds, ignore_index=True)
        df_gt_all = pd.concat(all_gts, ignore_index=True)

        # Calculate Competition Score
        score = utils.calc_score(df_pred_all, df_gt_all)
    else:
        score = float("inf")

    return val_loss, score


def run_training(
    train_loader,
    val_loader,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    device_name="cuda",
):
    """
    Main training loop with Early Stopping.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Initialize Model
    net = model.ResidualBiLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
    ).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.L1Loss(reduction="none")  # We handle reduction manually with mask

    best_score = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.WORKING_DIR, "model_best.pth")

    for epoch in range(epochs):
        train_loss = train_epoch(net, train_loader, criterion, optimizer, device)
        val_loss, val_score = validate(net, val_loader, criterion, device)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Score: {val_score:.9f}"
        )

        # Save Best Model based on Competition Score
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Score: {best_score:.9f}")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return best_score


def generate_submission(test_loader, model_path, output_path, device_name="cuda"):
    """
    Generates submission file using the trained model.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Load Model
    net = model.ResidualBiLSTM(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=config.OUTPUT_SIZE,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
    ).to(device)

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    net.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"]

            trip_ids = batch["tripIds"]
            timestamps_list = batch["timestamps"]
            wls_pos_list = batch["wls_pos"]

            outputs = net(features, lengths)
            outputs_np = outputs.cpu().numpy()

            for i in range(len(trip_ids)):
                length = lengths[i]
                trip_id = trip_ids[i]

                # Extract valid sequence
                pred_residuals = outputs_np[i, :length, :]

                # Get metadata
                ts = timestamps_list[i]
                wls = wls_pos_list[i]

                # Reconstruct Predictions
                lat_pred = wls[:, 0] + pred_residuals[:, 0]
                lon_pred = wls[:, 1] + pred_residuals[:, 1]

                df_p = pd.DataFrame(
                    {
                        "tripId": trip_id,
                        "UnixTimeMillis": ts,
                        "LatitudeDegrees": lat_pred,
                        "LongitudeDegrees": lon_pred,
                    }
                )
                all_preds.append(df_p)

    if all_preds:
        submission_df = pd.concat(all_preds, ignore_index=True)
        # Ensure correct column order
        submission_df = submission_df[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}. Rows: {len(submission_df)}")
    else:
        print("No predictions generated.")
