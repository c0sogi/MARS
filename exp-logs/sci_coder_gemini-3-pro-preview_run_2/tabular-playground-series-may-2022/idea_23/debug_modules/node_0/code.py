import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the provided library modules
from library import config, utils, data, model, train


def main():
    print("=== Starting Demonstration of Manufacturing Control Library ===")

    # --------------------------------------------------------------------------
    # 1. Setup and Utils Verification
    # --------------------------------------------------------------------------
    print("\n[1] Verifying Utilities...")
    utils.set_seed(42)

    # Test AUC calculation with a known deterministic case
    # Case:
    # y_true = [0, 0, 1, 1]
    # y_pred = [0.1, 0.4, 0.35, 0.8]
    # Pairs:
    # (0, 0.1) vs (1, 0.35) -> Correct
    # (0, 0.1) vs (1, 0.8)  -> Correct
    # (0, 0.4) vs (1, 0.35) -> Incorrect
    # (0, 0.4) vs (1, 0.8)  -> Correct
    # Expected AUC = 3/4 = 0.75
    y_true_test = np.array([0, 0, 1, 1])
    y_pred_test = np.array([0.1, 0.4, 0.35, 0.8])
    auc = utils.calculate_auc(y_true_test, y_pred_test)
    print(f"   Calculated AUC: {auc:.4f}")
    assert np.isclose(auc, 0.75), f"AUC calculation incorrect. Expected 0.75, got {auc}"
    print("   Utils verification passed.")

    # --------------------------------------------------------------------------
    # 2. Data Processing Demonstration
    # --------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Processing...")

    # process_data handles loading raw CSVs, scaling continuous features, and encoding sequences.
    # It caches the result to a .npz file. We enable loading from cache for speed.
    processed_data = data.process_data(load_cached_data=True)

    # Verify integrity of the processed data dictionary
    expected_keys = [
        "train_cont",
        "train_seq",
        "train_target",
        "train_ids",
        "test_cont",
        "test_seq",
        "test_ids",
    ]
    for key in expected_keys:
        assert key in processed_data, f"Missing key {key} in processed data."

    print(
        f"   Data loaded. Train continuous shape: {processed_data['train_cont'].shape}"
    )
    print(f"   Data loaded. Train sequence shape: {processed_data['train_seq'].shape}")

    # --------------------------------------------------------------------------
    # 3. Creating Subset DataLoaders for Rapid Prototyping
    # --------------------------------------------------------------------------
    print("\n[3] Creating Subset DataLoaders for Rapid Prototyping...")

    # To optimize for speed, we create a tiny subset of the data (128 samples)
    # This allows us to run training and validation steps almost instantly.
    subset_size = 128
    batch_size = 32

    # Extract subset arrays
    train_cont_sub = processed_data["train_cont"][:subset_size]
    train_seq_sub = processed_data["train_seq"][:subset_size]
    train_target_sub = processed_data["train_target"][:subset_size]

    # Instantiate the Dataset class provided in library/data.py
    demo_train_dataset = data.ManufacturingDataset(
        cont_features=train_cont_sub,
        seq_features=train_seq_sub,
        targets=train_target_sub,
    )

    # Create a standard PyTorch DataLoader
    demo_train_loader = DataLoader(
        demo_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Ensure batch sizes are consistent for this demo
    )

    # Verify DataLoader yields correct structure
    batch = next(iter(demo_train_loader))
    assert "continuous" in batch
    assert "sequence" in batch
    assert "target" in batch
    assert batch["continuous"].shape == (batch_size, config.NUM_CONTINUOUS_FEATURES)
    assert batch["sequence"].shape == (batch_size, config.SEQUENCE_LENGTH)

    print(
        f"   Subset DataLoader created. Batch size: {batch_size}, Batches: {len(demo_train_loader)}"
    )

    # --------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # --------------------------------------------------------------------------
    print("\n[4] Initializing Model and Verifying Forward Pass...")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Instantiate the HybridNetwork from library/model.py
    net = model.HybridNetwork().to(device)
    print("   Model initialized successfully.")

    # Move the demo batch to the device
    cont_input = batch["continuous"].to(device)
    seq_input = batch["sequence"].to(device)

    # Perform a forward pass
    output = net(cont_input, seq_input)

    # Check output shape: Should be (Batch, 1) for binary classification logits
    assert output.shape == (
        batch_size,
        1,
    ), f"Output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
    print(f"   Forward pass successful. Output shape: {output.shape}")

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Step (One Epoch on Subset)...")

    # Setup standard training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    # Use the provided train_one_epoch function from library/train.py
    # This iterates over the entire provided loader (which is small in this demo)
    avg_loss = train.train_one_epoch(
        net, demo_train_loader, optimizer, criterion, device
    )

    print(f"   Training step complete. Average Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss resulted in NaN."

    # --------------------------------------------------------------------------
    # 6. Validation Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Demonstrating Validation...")

    # Create a validation loader (using the same subset for demo purposes)
    demo_val_loader = DataLoader(
        demo_train_dataset, batch_size=batch_size, shuffle=False
    )

    # Use the provided validate function from library/train.py
    val_auc = train.validate(net, demo_val_loader, device)
    print(f"   Validation complete. AUC: {val_auc:.4f}")

    # AUC must be between 0 and 1
    assert 0.0 <= val_auc <= 1.0, f"AUC score {val_auc} is out of valid range [0, 1]."

    # --------------------------------------------------------------------------
    # 7. Prediction Demonstration
    # --------------------------------------------------------------------------
    print("\n[7] Demonstrating Prediction...")

    # Create a test subset (Note: targets=None for test data)
    test_cont_sub = processed_data["test_cont"][:subset_size]
    test_seq_sub = processed_data["test_seq"][:subset_size]

    demo_test_dataset = data.ManufacturingDataset(
        cont_features=test_cont_sub, seq_features=test_seq_sub, targets=None
    )

    demo_test_loader = DataLoader(
        demo_test_dataset, batch_size=batch_size, shuffle=False
    )

    # Use the provided predict function from library/train.py
    predictions = train.predict(net, demo_test_loader, device)

    # Verify predictions
    assert (
        len(predictions) == subset_size
    ), f"Prediction count mismatch. Expected {subset_size}, got {len(predictions)}"
    # Predictions should be probabilities (sigmoid applied in predict function)
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Predictions contain values outside [0, 1]."

    print(f"   Prediction complete. Generated {len(predictions)} predictions.")
    print(f"   Sample predictions: {predictions[:5]}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
