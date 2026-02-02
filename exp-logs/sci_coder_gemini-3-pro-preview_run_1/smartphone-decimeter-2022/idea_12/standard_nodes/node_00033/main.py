import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_preprocessing import prepare_training_data
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import AtrousResUNet
from library.loss import DeepSupervisionMAELoss
from library.train import train_one_epoch, validate
from library.inference import predict_and_convert


def calculate_competition_metric(model, loader, device):
    """
    Calculates the competition metric: Mean of the 50th and 95th percentile distance errors,
    averaged across phones.
    Also returns feature values and errors for failure analysis.
    """
    model.eval()
    all_errors = []
    all_trip_ids = []

    # For failure analysis
    feature_vals = {col: [] for col in Config.INPUT_FEATURES}
    error_vals = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["masks"].to(device)
            trip_ids = batch["trip_ids"]

            # Predict
            outputs = model(features)
            final_out = outputs[0]  # (B, 2, L)

            # Iterate batch
            for i in range(features.shape[0]):
                valid_len = int(masks[i].sum().item())
                if valid_len == 0:
                    continue

                pred = final_out[i, :, :valid_len].cpu().numpy()  # (2, L)
                target = targets[i, :, :valid_len].cpu().numpy()  # (2, L)

                # Calculate Euclidean distance error in meters
                # pred/target are [North, East]
                # diff shape: (2, L)
                diff = pred - target
                dist_error = np.sqrt((diff**2).sum(axis=0))  # (L,)

                all_errors.append(dist_error)
                all_trip_ids.extend([trip_ids[i]] * valid_len)

                # Collect features for failure analysis (flattened)
                # features[i] is (C, L)
                feat_seq = features[i, :, :valid_len].cpu().numpy()
                for f_idx, f_name in enumerate(Config.INPUT_FEATURES):
                    feature_vals[f_name].extend(feat_seq[f_idx])
                error_vals.extend(dist_error)

    if not all_errors:
        return 0.0, feature_vals, error_vals

    # Combine errors into DataFrame
    flat_errors = np.concatenate(all_errors)
    df_metrics = pd.DataFrame({"tripId": all_trip_ids, "error": flat_errors})

    # Calculate metric per phone (tripId)
    score_per_phone = []
    for trip, group in df_metrics.groupby("tripId"):
        errors = group["error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        score_per_phone.append((p50 + p95) / 2)

    final_metric = np.mean(score_per_phone)

    return final_metric, feature_vals, error_vals


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Override epochs for fast baseline
    EPOCHS = 10

    # 2. Data Preparation
    print("\n[1/5] Preparing Data...")
    train_df, val_df = prepare_training_data(load_cached_data=True)

    train_dataset = GNSSSequenceDataset(train_df, mode="train")
    val_dataset = GNSSSequenceDataset(val_df, mode="train", scaler=train_dataset.scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=gnss_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Setup
    print("\n[2/5] Initializing Model...")
    model = AtrousResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_dim=Config.HIDDEN_DIM,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
    )

    criterion = DeepSupervisionMAELoss(weights=Config.LOSS_WEIGHTS).to(device)

    # 4. Training Loop
    print(f"\n[3/5] Training for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_n, val_mae_e = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE N/E: {val_mae_n:.2f}m / {val_mae_e:.2f}m"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # 5. Validation Assessment & Failure Analysis
    print("\n[4/5] Validation Assessment & Failure Analysis...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    metric, feat_vals, err_vals = calculate_competition_metric(
        model, val_loader, device
    )
    print(f"Final Validation Metric: {metric}")

    print("\nFailure Analysis (Pearson Correlation with Error Magnitude):")
    correlations = []
    # Convert error_vals to numpy array once
    err_arr = np.array(err_vals)

    for f_name, f_data in feat_vals.items():
        if len(f_data) == len(err_arr) and len(f_data) > 0:
            f_arr = np.array(f_data)
            # Handle potential constant arrays to avoid NaNs
            if np.std(f_arr) > 0 and np.std(err_arr) > 0:
                corr = np.corrcoef(f_arr, err_arr)[0, 1]
                correlations.append((f_name, corr))
            else:
                correlations.append((f_name, 0.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for f_name, corr in correlations:
        print(f"  {f_name}: {corr:.4f}")

    # 6. Submission
    print("\n[5/5] Checking Submission Threshold...")
    THRESHOLD = 3.802240262877392

    if metric < THRESHOLD:
        print(
            f"Metric {metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_convert(
            device=device,
            scaler=train_dataset.scaler,
            model_path=best_model_path,
            load_cached_data=True,
        )
    else:
        print(
            f"Metric {metric} is NOT below threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    run()
