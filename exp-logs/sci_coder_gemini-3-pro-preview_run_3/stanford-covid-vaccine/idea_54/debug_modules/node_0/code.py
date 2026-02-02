import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import provided library modules
import library.config as config
import library.data as data_lib
import library.model as model_lib
import library.utils as utils_lib
import library.train as train_lib


def main():
    # =========================================================================
    # 1. Setup
    # =========================================================================
    print("Initializing demonstration...")
    config.set_seed(42)
    # Use CPU for demonstration to avoid overhead/compatibility issues on small data
    device = torch.device("cpu")

    # =========================================================================
    # 2. Data Processing Demonstration
    # =========================================================================
    print("\n--- Testing Data Processing ---")

    # Load a small subset of the training metadata to optimize for speed
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    train_df_subset = pd.read_parquet(config.TRAIN_METADATA_PATH).head(32)
    print(f"Loaded subset of {len(train_df_subset)} training samples.")

    # Process the dataframe into features and targets
    features, pair_indices, pair_masks, targets = data_lib.process_dataframe(
        train_df_subset, has_targets=True
    )

    # Verify shapes
    # Features: (N, 107, 14)
    expected_feat_shape = (32, config.SEQ_LEN, config.INPUT_DIM)
    assert (
        features.shape == expected_feat_shape
    ), f"Feature shape mismatch: {features.shape}"

    # Pair Indices: (N, 107)
    assert pair_indices.shape == (32, config.SEQ_LEN), "Pair indices shape mismatch"

    # Pair Masks: (N, 107, 1)
    assert pair_masks.shape == (32, config.SEQ_LEN, 1), "Pair masks shape mismatch"

    # Targets: (N, 107, 5)
    assert targets.shape == (
        32,
        config.SEQ_LEN,
        config.OUTPUT_DIM,
    ), "Targets shape mismatch"

    print("Data processing shapes verified.")

    # Create Dataset and DataLoader
    ids = train_df_subset["id"].values
    train_dataset = data_lib.RNADataset(
        features, pair_indices, pair_masks, targets, ids
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False
    )

    # =========================================================================
    # 3. Model Architecture Demonstration
    # =========================================================================
    print("\n--- Testing Model Architecture ---")

    model = model_lib.DeepStabilizedBiGRU().to(device)

    # Fetch a single batch
    batch_features, batch_indices, batch_masks, batch_targets = next(iter(train_loader))
    batch_features = batch_features.to(device)
    batch_indices = batch_indices.to(device)
    batch_masks = batch_masks.to(device)

    # Forward pass
    outputs = model(batch_features, batch_indices, batch_masks)

    # Verify output
    expected_out_shape = (4, config.SEQ_LEN, config.OUTPUT_DIM)
    assert (
        outputs.shape == expected_out_shape
    ), f"Model output shape mismatch: {outputs.shape}"
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    print("Model forward pass successful.")

    # =========================================================================
    # 4. Metric and Loss Demonstration
    # =========================================================================
    print("\n--- Testing Metrics and Loss ---")

    # Create dummy predictions and targets
    dummy_preds = torch.rand(4, 107, 5)
    dummy_targets = torch.rand(4, 107, 5)

    # Calculate MCRMSE
    metric_all = utils_lib.compute_mcrmse(
        dummy_preds, dummy_targets, scoring_only=False
    )
    metric_scored = utils_lib.compute_mcrmse(
        dummy_preds, dummy_targets, scoring_only=True
    )

    print(f"MCRMSE (All): {metric_all.item():.4f}")
    print(f"MCRMSE (Scored): {metric_scored.item():.4f}")

    assert metric_all > 0, "Metric should be positive"

    # Verify Loss Class
    criterion = utils_lib.MCRMSELoss()
    loss_val = criterion(dummy_preds, dummy_targets)

    # Loss class should wrap compute_mcrmse(scoring_only=False)
    assert torch.isclose(loss_val, metric_all), "Loss class output mismatch"
    print("Metric and Loss logic verified.")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n--- Testing Training Loop Components ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for one epoch on the subset
    train_loss = train_lib.train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train Loss (Subset): {train_loss:.4f}")

    # Validate on the same subset (just for demo purposes)
    val_loss, val_metric = train_lib.validate(model, train_loader, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Metric: {val_metric:.4f}")

    assert train_loss > 0, "Train loss should be positive"
    assert val_loss > 0, "Validation loss should be positive"

    # =========================================================================
    # 6. Submission Generation Demonstration
    # =========================================================================
    print("\n--- Testing Submission Generation ---")

    # Load small test subset
    test_df_subset = pd.read_parquet(config.TEST_METADATA_PATH).head(10)

    # Process test data (has_targets=False)
    t_features, t_indices, t_masks, t_targets = data_lib.process_dataframe(
        test_df_subset, has_targets=False
    )
    t_ids = test_df_subset["id"].values

    test_dataset = data_lib.RNADataset(t_features, t_indices, t_masks, t_targets, t_ids)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Generate submission
    # Note: This writes to ./submission/submission.csv
    train_lib.generate_submission(model, test_loader, device)

    # Verify file creation and content
    expected_path = config.SUBMISSION_PATH
    assert os.path.exists(expected_path), "Submission file was not created"

    sub_df = pd.read_csv(expected_path)
    # 10 samples * 107 positions = 1070 rows
    expected_rows = 10 * config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Verify columns
    expected_cols = ["id_seqpos"] + config.ALL_TARGETS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print(f"Submission verified. Rows: {len(sub_df)}")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
