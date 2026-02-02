import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import DataProcessor, VentilatorDataset
from library.train import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------------------------
    # Using defaults from Config class (Full dataset, 50 epochs)
    # Cite solution_lesson_node_00018: Smaller batch size (128) with sufficient epochs
    # leads to better generalization than large batches or short runs.

    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(
        f"Configuration: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, BATCH_SIZE={Config.BATCH_SIZE}"
    )

    # --------------------------------------------------------------------------
    # 2. Data Preparation
    # --------------------------------------------------------------------------
    print("Initializing DataProcessor...")
    processor = DataProcessor(Config)

    # Load data (utilizes caching if available and matching)
    (train_data, val_data, test_data) = processor.prepare_data(load_cached_data=True)

    # Unpack data
    train_x, train_y, train_u_out = train_data
    val_x, val_y, val_u_out = val_data
    test_x, _, test_u_out = test_data

    # Create Datasets
    train_dataset = VentilatorDataset(train_x, train_y, train_u_out)
    val_dataset = VentilatorDataset(val_x, val_y, val_u_out)
    test_dataset = VentilatorDataset(test_x, None, test_u_out)

    # Create DataLoaders
    # Using pin_memory and num_workers for efficiency
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------------------------
    trainer = Trainer(Config)
    best_model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    best_val_mae = float("inf")

    print(f"Starting training on {trainer.device}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_mae = trainer.validate(val_loader)

        # Scheduler Step
        trainer.scheduler.step(val_mae)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(trainer.model.state_dict(), best_model_path)

    # --------------------------------------------------------------------------
    # 4. Final Validation & Metrics
    # --------------------------------------------------------------------------
    print("Loading best model for final evaluation...")
    if os.path.exists(best_model_path):
        trainer.model.load_state_dict(
            torch.load(best_model_path, map_location=trainer.device)
        )

    # Compute Final Metric
    final_metric = trainer.validate(val_loader)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 5. Failure Analysis
    # --------------------------------------------------------------------------
    print("\nRunning Failure Analysis...")
    trainer.model.eval()

    # Collect predictions, targets, and inputs
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(trainer.device)
            p = trainer.model(x)
            val_preds.append(p.cpu().numpy())

    val_preds = np.concatenate(val_preds).flatten()
    y_true = val_y.flatten()
    u_out_flat = val_u_out.flatten()

    # Reshape features for correlation analysis: (N_samples, N_features)
    x_flat = val_x.reshape(-1, val_x.shape[-1])

    # Mask for inspiratory phase (u_out == 0)
    mask = u_out_flat == 0

    if mask.sum() > 0:
        # Calculate Absolute Error
        errors = np.abs(val_preds[mask] - y_true[mask])
        x_masked = x_flat[mask]

        print("Correlation between Error Magnitude and Features:")
        feature_names = Config.FEATURE_COLS

        for i, feat_name in enumerate(feature_names):
            # Skip u_out as it is constant 0 in this mask
            if feat_name == "u_out":
                continue

            feat_vals = x_masked[:, i]

            # Calculate Pearson correlation
            # Handle constant features to avoid division by zero
            if np.std(feat_vals) > 1e-9:
                corr = np.corrcoef(feat_vals, errors)[0, 1]
                print(f"{feat_name}: {corr:.4f}")
            else:
                print(f"{feat_name}: 0.0000 (Constant)")
    else:
        print("No inspiratory phase samples found for analysis.")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.23978149890899658

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")

        # Generate predictions
        test_preds = trainer.predict(test_loader)

        # Load test metadata to map IDs
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply same filtering as DataProcessor if DEBUG is on
        if Config.DEBUG:
            test_ids = test_df["breath_id"].unique()[: Config.DEBUG_SAMPLES]
            test_df = test_df[test_df["breath_id"].isin(test_ids)].copy()

        # Sort to ensure alignment with predictions
        # DataProcessor sorts by [breath_id, time_step]
        test_df = test_df.sort_values(["breath_id", "time_step"])

        # Verify alignment
        if len(test_preds) != len(test_df):
            print(
                f"Error: Prediction count {len(test_preds)} != Metadata count {len(test_df)}"
            )
        else:
            # Create submission dataframe
            submission = pd.DataFrame({"id": test_df["id"], "pressure": test_preds})

            # Sort by ID for final format
            submission = submission.sort_values("id")

            # Save
            os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
            submission.to_csv(Config.SUBMISSION_PATH, index=False)
            print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
