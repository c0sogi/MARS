import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, calculate_competition_metric
from library.preprocessing import PreProcessor
from library.dataset import GnssSequenceDataset
from library.model import SEResUNet1D
from library.loss import DeepSupervisionLoss


def train_model(debug=False, epochs=Config.EPOCHS):
    """
    Main training function for the 1D SE-ResUNet model.

    Args:
        debug (bool): If True, runs on a small subset of data.
        epochs (int): Number of training epochs.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # 2. Data Loading & Processing
    preprocessor = PreProcessor()
    # Load cached data or process from scratch
    train_df, val_df, _ = preprocessor.process_data(load_cached_data=True)

    if debug:
        print("Debug mode: Sampling data...")
        train_df = train_df.iloc[:1000]
        val_df = val_df.iloc[:500]

    # 3. Datasets & Loaders
    # Training: Sliding windows with stride
    train_dataset = GnssSequenceDataset(
        train_df,
        mode="train",
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE // 2,  # 50% overlap for training
    )

    # Validation: Full sequences (batch_size=1 required for variable lengths)
    val_dataset = GnssSequenceDataset(
        val_df,
        mode="val",
        window_size=Config.TRAIN_WINDOW_SIZE,  # Not used for splitting in val mode
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for variable sequence lengths in validation
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 4. Model, Optimizer, Loss
    model = SEResUNet1D(in_channels=Config.INPUT_CHANNELS, out_channels=2).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    criterion = DeepSupervisionLoss(weights=Config.DEEP_SUPERVISION_WEIGHTS)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (features, targets, meta) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            mask = meta["mask"].to(device)

            optimizer.zero_grad()

            # Forward pass: returns (final, aux1, aux2)
            preds = model(features)

            # Compute deep supervision loss
            loss = criterion(preds, targets, mask)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)

            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0

        # Store predictions for metric calculation
        all_preds_lat = []
        all_preds_lon = []
        all_gt_lat = []
        all_gt_lon = []
        all_phone_names = []

        with torch.no_grad():
            for features, targets, meta in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                # Forward pass (eval mode returns only final head)
                out_final = model(features)

                # Compute validation loss (MAE on final head)
                # Note: DeepSupervisionLoss handles list or tensor input
                val_loss = criterion(
                    out_final, targets, mask=None
                )  # No mask needed for full seq
                val_loss_accum += val_loss.item()

                # --- Reconstruct Coordinates for Metric ---
                # Output is (B, 2, T) -> (1, 2, T)
                # Convert to numpy (T, 2)
                pred_enu = out_final.cpu().numpy()[0].T
                pred_e = pred_enu[:, 0]
                pred_n = pred_enu[:, 1]

                # Get Baseline WLS
                base_lat = meta["baseline_lat"].numpy()[0]
                base_lon = meta["baseline_lon"].numpy()[0]

                # Approximate conversion from ENU offsets (meters) to Geodetic offsets (degrees)
                # 1 deg lat approx 111320m
                # 1 deg lon approx 111320m * cos(lat)
                lat_scale = 111320.0
                lon_scale = 111320.0 * np.cos(np.radians(base_lat))

                pred_lat = base_lat + (pred_n / lat_scale)
                pred_lon = base_lon + (pred_e / lon_scale)

                # Get Ground Truth (reconstruct from target offsets to be consistent,
                # or use original if available, but we only have offsets in dataset)
                # Target is ENU offset from baseline
                target_enu = targets.cpu().numpy()[0].T
                target_e = target_enu[:, 0]
                target_n = target_enu[:, 1]

                gt_lat = base_lat + (target_n / lat_scale)
                gt_lon = base_lon + (target_e / lon_scale)

                # Accumulate
                all_preds_lat.extend(pred_lat)
                all_preds_lon.extend(pred_lon)
                all_gt_lat.extend(gt_lat)
                all_gt_lon.extend(gt_lon)
                # Phone name is repeated for sequence length
                seq_len = len(pred_lat)
                all_phone_names.extend([meta["phone_name"][0]] * seq_len)

        avg_val_loss = val_loss_accum / len(val_loader)

        # Calculate Competition Metric
        df_pred = pd.DataFrame(
            {
                "phone_name": all_phone_names,
                "LatitudeDegrees": all_preds_lat,
                "LongitudeDegrees": all_preds_lon,
            }
        )

        df_gt = pd.DataFrame(
            {
                "phone_name": all_phone_names,
                "LatitudeDegrees": all_gt_lat,
                "LongitudeDegrees": all_gt_lon,
            }
        )

        val_score = calculate_competition_metric(df_pred, df_gt)

        # Step Scheduler
        scheduler.step()

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Val Score (50-95): {val_score:.9f}"
        )

        # Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            save_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Score! Model saved to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.9f}")
