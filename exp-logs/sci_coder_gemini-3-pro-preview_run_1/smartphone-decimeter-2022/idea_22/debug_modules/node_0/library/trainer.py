import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import StratifiedResUNet1D
from library.utils import save_checkpoint, load_checkpoint, cartesian_to_wgs84


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    # Weights for auxiliary losses from Config
    # Map stride string to weight
    aux_weights = {}
    for stride, weight in zip(Config.DEEP_SUPERVISION_STRIDES, Config.AUX_LOSS_WEIGHTS):
        aux_weights[str(stride)] = weight

    for batch_idx, (features, targets, mask, metadata) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs is a dict: {'main': ..., 'aux_2': ..., 'aux_4': ...}
        outputs = model(features)

        # 1. Main Loss (Full Resolution)
        # Apply mask to ignore padding
        main_pred = outputs["main"]

        # Ensure shapes match (sometimes padding might cause off-by-one in edge cases, though collate handles it)
        # We use the mask to select valid elements for loss calculation
        # Flatten tensors for masked selection
        # mask shape: (B, L) -> expand to (B, 2, L)
        mask_expanded = mask.unsqueeze(1).expand_as(main_pred)

        pred_masked = main_pred[mask_expanded]
        target_masked = targets[mask_expanded]

        loss = criterion(pred_masked, target_masked)

        # 2. Auxiliary Losses (Decimated Deep Supervision)
        for name, aux_pred in outputs.items():
            if name.startswith("aux_"):
                stride_str = name.split("_")[1]
                stride = int(stride_str)

                if stride_str in aux_weights:
                    # Decimate targets: Slice the ground truth to match aux resolution
                    # Target shape: (B, 2, L)
                    # Aux shape: (B, 2, L//stride)
                    target_decimated = targets[:, :, ::stride]

                    # Decimate mask similarly
                    mask_decimated = mask[:, ::stride]

                    # Handle potential length mismatch due to padding/pooling rounding
                    # Truncate to the shorter length
                    min_len = min(aux_pred.shape[2], target_decimated.shape[2])
                    aux_pred = aux_pred[:, :, :min_len]
                    target_decimated = target_decimated[:, :, :min_len]
                    mask_decimated = mask_decimated[:, :min_len]

                    # Apply mask
                    mask_dec_expanded = mask_decimated.unsqueeze(1).expand_as(aux_pred)
                    aux_pred_masked = aux_pred[mask_dec_expanded]
                    target_dec_masked = target_decimated[mask_dec_expanded]

                    if len(aux_pred_masked) > 0:
                        aux_loss = criterion(aux_pred_masked, target_dec_masked)
                        loss += aux_weights[stride_str] * aux_loss

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Validates the model.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for features, targets, mask, metadata in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            outputs = model(features)
            main_pred = outputs["main"]

            mask_expanded = mask.unsqueeze(1).expand_as(main_pred)
            pred_masked = main_pred[mask_expanded]
            target_masked = targets[mask_expanded]

            loss = criterion(pred_masked, target_masked)
            running_loss += loss.item()

    return running_loss / len(dataloader)


def train_model(load_cached_data=True):
    """
    Main training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and Loaders
    print("Initializing datasets...")
    train_dataset = GNSSSequenceDataset(
        split="train", load_cached_data=load_cached_data, debug=Config.DEBUG
    )
    val_dataset = GNSSSequenceDataset(
        split="val", load_cached_data=load_cached_data, debug=Config.DEBUG
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = StratifiedResUNet1D().to(device)

    # Optimization
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        duration = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {duration:.1f}s"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, best_model_path)
            print(f"  New best model saved! (Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    return best_model_path


def generate_submission(model_path, load_cached_data=True):
    """
    Generates submission file using the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = StratifiedResUNet1D().to(device)
    load_checkpoint(model, None, model_path, device=device)
    model.eval()

    # Load Test Data
    print("Loading test data...")
    test_dataset = GNSSSequenceDataset(
        split="test", load_cached_data=load_cached_data, debug=Config.DEBUG
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=4,
    )

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for features, mask, metadata_list in test_loader:
            features = features.to(device)

            # Forward pass
            outputs = model(features)
            preds = outputs["main"].cpu().numpy()  # (B, 2, L)

            # Iterate over batch
            for i in range(len(metadata_list)):
                meta = metadata_list[i]

                # Get valid length from mask or metadata
                # The mask in collate is (B, L_max).
                # We can also infer length from the timestamps in metadata which is (L,)
                seq_len = meta["timestamps"].shape[0]

                # Extract predictions for this sequence (2, L) -> (L, 2)
                # preds[i] is (2, L_max)
                local_preds = preds[i, :, :seq_len].transpose(1, 0)  # (L, 2)

                d_east = local_preds[:, 0]
                d_north = local_preds[:, 1]

                # Get Baseline WLS
                wls_pos = meta["wls_pos"].numpy()  # (L, 2) -> Lat, Lon
                wls_lat = wls_pos[:, 0]
                wls_lon = wls_pos[:, 1]

                # Convert offsets to WGS84
                pred_lat, pred_lon = cartesian_to_wgs84(
                    d_east, d_north, wls_lat, wls_lon
                )

                # Store results
                drive_id = meta["drive_id"]
                phone_name = meta["phone_name"]
                timestamps = meta["timestamps"].numpy()

                for t, lat, lon in zip(timestamps, pred_lat, pred_lon):
                    results.append(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "UnixTimeMillis": t,
                            "LatitudeDegrees": lat,
                            "LongitudeDegrees": lon,
                        }
                    )

    # Convert to DataFrame
    pred_df = pd.DataFrame(results)

    # Construct tripId
    pred_df["tripId"] = pred_df["drive_id"] + "-" + pred_df["phone_name"]

    # Load Sample Submission to ensure correct rows and order
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Merge predictions onto sample submission
    # We use left join on sample submission to ensure we output exactly what's requested
    submission = sample_sub[["tripId", "UnixTimeMillis"]].merge(
        pred_df[["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]],
        on=["tripId", "UnixTimeMillis"],
        how="left",
    )

    # Fill missing (if any) with WLS baseline or NaNs?
    # Ideally there shouldn't be missing if we processed everything.
    # If missing, we might want to fallback to sample submission values (which are usually baseline)
    # But here we assume coverage.

    # Save
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
