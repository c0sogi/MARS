import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import sys

# Import from provided library
from library.config import Config
from library.model import RPCNet
from library.data_utils import get_dataloaders, set_seed
from library.train_utils import train_epoch, validate_epoch, MaskedL1Loss, predict


def main():
    # 1. Configuration & Setup
    # Limit epochs for fast baseline execution as per instructions
    Config.EPOCHS = 15

    # Ensure we use the device
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    set_seed(Config.SEED)

    # 2. Data Loading
    # Load cached data if available, otherwise process from scratch
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RPCNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    # Using verbose=False to keep output clean, logic handles updates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,  # Reduced patience for fewer epochs
    )

    criterion = MaskedL1Loss()

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)

    print("Training complete.")

    # 5. Final Evaluation & Failure Analysis
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    print("Computing validation metrics and performing failure analysis...")

    # Accumulate validation data for analysis
    all_preds = []
    all_targets = []
    all_u_out = []
    all_features = []

    with torch.no_grad():
        for batch in val_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(X)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())
            all_features.append(X.cpu().numpy())

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_u_out = np.concatenate(all_u_out, axis=0).flatten()
    # Features: (N, Seq, Feat) -> (N*Seq, Feat)
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(-1, all_features.shape[-1])

    # Filter for inspiratory phase (u_out == 0)
    mask = all_u_out == 0

    valid_preds = all_preds[mask]
    valid_targets = all_targets[mask]
    valid_features = all_features[mask]

    # Compute Metric
    abs_errors = np.abs(valid_preds - valid_targets)
    final_metric = np.mean(abs_errors)

    # Print Metric (Full Precision)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis: Correlation of Error with Features")
    feature_names = Config.FEATURE_LIST

    correlations = []
    for i, feat_name in enumerate(feature_names):
        feat_values = valid_features[:, i]
        # Pearson correlation
        if np.std(feat_values) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(feat_values, abs_errors)[0, 1]
        correlations.append((feat_name, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr in correlations:
        print(f"{name}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = 0.23978149890899658

    if final_metric < THRESHOLD:
        print(
            f"\nMetric condition met ({final_metric} < {THRESHOLD}). Generating submission..."
        )

        # Generate predictions on test set
        # predict() returns flattened array
        flat_test_preds = predict(model, test_loader, device)

        # Load test metadata to align IDs
        # Note: data_utils.compute_features sorts by [breath_id, time_step]
        # We must align our submission dataframe similarly
        test_df = pd.read_csv(Config.TEST_CSV)
        test_df = test_df.sort_values(["breath_id", "time_step"])

        if len(flat_test_preds) != len(test_df):
            print(
                f"Warning: Prediction length {len(flat_test_preds)} != Test DF length {len(test_df)}"
            )

        test_df["pressure"] = flat_test_preds

        # Prepare submission dataframe
        submission_df = test_df[["id", "pressure"]]

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric condition NOT met ({final_metric} >= {THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
