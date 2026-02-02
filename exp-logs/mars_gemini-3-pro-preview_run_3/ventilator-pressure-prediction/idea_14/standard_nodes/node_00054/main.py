import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.data_utils import engineer_features, VentilatorDataset
from library.train_utils import run_training
from library.model import MSDHNet


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Baseline
    # -------------------------------------------------------------------------
    # Override Config to ensure execution finishes within time limits
    # 15 epochs on 15,000 breaths (approx 1.2M samples) is sufficient and fast.
    Config.EPOCHS = 15
    Config.BATCH_SIZE = 128

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Configuration: Epochs={Config.EPOCHS}, Working Dir={Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preprocessing & Caching (Subset Strategy)
    # -------------------------------------------------------------------------
    # We manually generate the cache to control the training set size.
    # run_training will pick up these files if load_cached_data=True.

    train_x_path = os.path.join(Config.WORKING_DIR, "train_x.npy")

    if not os.path.exists(train_x_path):
        print("Cache not found. Generating preprocessed data with subset strategy...")

        # Load Raw Metadata
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        # Subset Training Data (15,000 breaths)
        # This balances speed and performance.
        unique_breaths = train_df["breath_id"].unique()
        # Shuffle to ensure random subset if not already shuffled
        np.random.seed(Config.SEED)
        subset_breaths = np.random.choice(unique_breaths, size=15000, replace=False)
        train_df = train_df[train_df["breath_id"].isin(subset_breaths)].copy()

        print(
            f"Training subset size: {len(train_df)} samples ({len(subset_breaths)} breaths)"
        )

        # Save Test IDs for submission reconstruction
        test_ids = test_df[Config.ID_COL].values
        np.save(os.path.join(Config.WORKING_DIR, "test_ids.npy"), test_ids)

        # Extract Targets
        train_y_raw = train_df[Config.TARGET_COL].values
        val_y_raw = val_df[Config.TARGET_COL].values

        # Feature Engineering
        print("Engineering features...")
        train_df = engineer_features(train_df)
        val_df = engineer_features(val_df)
        test_df = engineer_features(test_df)

        # Select Features
        X_train = train_df[Config.FEATURE_COLS].values
        X_val = val_df[Config.FEATURE_COLS].values
        X_test = test_df[Config.FEATURE_COLS].values

        # Scaling
        print("Scaling features...")
        scaler = RobustScaler()
        # Fit on training subset only
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Reshaping
        print("Reshaping tensors...")
        seq_len = Config.SEQ_LEN

        # Helper for reshaping
        def reshape_seq(data):
            return data.reshape(-1, seq_len, data.shape[-1])

        def reshape_target(data):
            return data.reshape(-1, seq_len)

        train_x = reshape_seq(X_train)
        train_y = reshape_target(train_y_raw)
        val_x = reshape_seq(X_val)
        val_y = reshape_target(val_y_raw)
        test_x = reshape_seq(X_test)

        # Save to cache
        print("Saving to cache...")
        np.save(os.path.join(Config.WORKING_DIR, "train_x.npy"), train_x)
        np.save(os.path.join(Config.WORKING_DIR, "train_y.npy"), train_y)
        np.save(os.path.join(Config.WORKING_DIR, "val_x.npy"), val_x)
        np.save(os.path.join(Config.WORKING_DIR, "val_y.npy"), val_y)
        np.save(os.path.join(Config.WORKING_DIR, "test_x.npy"), test_x)
    else:
        print("Cached data found. Skipping preprocessing.")

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Starting training pipeline...")
    # This will load the cache we just created
    best_val_loss = run_training(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 4. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print(f"Final Validation Metric: {best_val_loss:.16f}")

    THRESHOLD = 0.1642141044139862

    if best_val_loss < THRESHOLD:
        print("\nThreshold met. Performing Failure Analysis...")

        # Load Validation Data
        val_x = np.load(os.path.join(Config.WORKING_DIR, "val_x.npy"))
        val_y = np.load(os.path.join(Config.WORKING_DIR, "val_y.npy"))

        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MSDHNet().to(device)
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        model.eval()

        # Create DataLoader for inference
        val_dataset = VentilatorDataset(val_x, val_y)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=512,  # Larger batch for faster inference
            shuffle=False,
            num_workers=2,
        )

        all_errors = []
        all_feats = []

        print("Running inference on validation set for analysis...")
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward
                preds = model(x_batch)

                # Get u_out for masking (index 1)
                u_out = x_batch[:, :, 1]

                # Calculate Absolute Error
                abs_err = torch.abs(preds - y_batch)

                # Mask: Only analyze inspiratory phase
                mask = u_out == 0

                # Flatten and filter
                # View as (-1) flattens the batch and sequence dims
                mask_flat = mask.view(-1)
                err_flat = abs_err.view(-1)
                # Flatten features: (B, L, D) -> (B*L, D)
                x_flat = x_batch.view(-1, x_batch.shape[-1])

                # Select valid points
                valid_err = err_flat[mask_flat].cpu().numpy()
                valid_feats = x_flat[mask_flat].cpu().numpy()

                all_errors.append(valid_err)
                all_feats.append(valid_feats)

        # Concatenate
        all_errors = np.concatenate(all_errors)
        all_feats = np.concatenate(all_feats)

        print(f"Analyzed {len(all_errors)} time steps.")
        print("\nCorrelation between Error Magnitude and Input Features:")
        print("-" * 50)
        print(f"{'Feature':<20} | {'Correlation':<10}")
        print("-" * 50)

        for i, feat_name in enumerate(Config.FEATURE_COLS):
            # Calculate Pearson correlation
            corr = np.corrcoef(all_feats[:, i], all_errors)[0, 1]
            print(f"{feat_name:<20} | {corr:.6f}")
        print("-" * 50)

        print(f"Submission file generated at: {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric {best_val_loss} did not meet threshold {THRESHOLD}."
        )
        print("Discarding submission file.")
        if os.path.exists(Config.SUBMISSION_PATH):
            os.remove(Config.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
