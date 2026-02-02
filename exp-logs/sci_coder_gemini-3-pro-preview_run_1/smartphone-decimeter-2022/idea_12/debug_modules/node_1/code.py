import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from library modules
from library.config import Config
from library.utils import set_seed
from library.data_preprocessing import prepare_training_data
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import AtrousResUNet
from library.loss import DeepSupervisionMAELoss
from library.train import train_one_epoch, validate
from library.inference import predict_and_convert


def main():
    print("Initializing demonstration...")

    # 1. Configuration Override for Speed and Demonstration
    # We modify the Config class attributes directly to run a minimal version
    Config.DEBUG_SAMPLE_SIZE = 2  # Process only 2 drives to save time
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.HIDDEN_DIM = 32

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==========================================
    # 2. Data Preprocessing & Dataset Creation
    # ==========================================
    print("\n--- Testing Data Preprocessing & Dataset ---")

    # Prepare training data (this will process raw GNSS files for the debug drives)
    # load_cached_data=False ensures we actually run the processing logic on the subset
    train_df, val_df = prepare_training_data(load_cached_data=False)

    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Val DataFrame shape: {val_df.shape}")

    # Assertions to verify data structure
    assert not train_df.empty, "Training dataframe is empty."
    assert "delta_north" in train_df.columns, "Target column 'delta_north' missing."
    assert "delta_east" in train_df.columns, "Target column 'delta_east' missing."

    # Instantiate PyTorch Datasets
    # The scaler is computed from the training set and applied to validation
    train_dataset = GNSSSequenceDataset(train_df, mode="train")
    val_dataset = GNSSSequenceDataset(val_df, mode="train", scaler=train_dataset.scaler)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # Verify a single item from the dataset
    if len(train_dataset) > 0:
        item = train_dataset[0]
        print(f"Sample item keys: {list(item.keys())}")
        print(
            f"Feature shape: {item['features'].shape}"
        )  # Expected: (Channels, Length)
        print(f"Target shape: {item['targets'].shape}")  # Expected: (2, Length)

        # Logic checks
        assert item["features"].dim() == 2, "Features should be 2D (Channels, Length)"
        assert (
            item["features"].shape[0] == Config.IN_CHANNELS
        ), f"Expected {Config.IN_CHANNELS} input channels"
        assert (
            item["targets"].shape[0] == Config.OUT_CHANNELS
        ), f"Expected {Config.OUT_CHANNELS} target channels"
        assert (
            item["features"].shape[1] == item["targets"].shape[1]
        ), "Feature and target sequence lengths must match"

    # Test DataLoader and Collate Function
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
    )

    # Fetch one batch to verify collation (padding and masking)
    batch = next(iter(train_loader))
    print(f"Batch features shape: {batch['features'].shape}")  # (B, C, L_max)
    print(f"Batch masks shape: {batch['masks'].shape}")  # (B, L_max)

    assert batch["features"].dim() == 3, "Batch features should be 3D"
    assert batch["masks"].dim() == 2, "Batch masks should be 2D"
    assert batch["features"].shape[0] <= Config.BATCH_SIZE, "Batch size mismatch"

    # ==========================================
    # 3. Model Instantiation & Forward Pass
    # ==========================================
    print("\n--- Testing Model Architecture ---")

    # Instantiate the Atrous Residual U-Net
    # We use a smaller base_dim for the demo to save memory/time
    model = AtrousResUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_dim=Config.HIDDEN_DIM,
    ).to(device)

    features = batch["features"].to(device)
    targets = batch["targets"].to(device)
    masks = batch["masks"].to(device)

    # Forward pass
    # The model returns a list of outputs for deep supervision: [final_out, aux1, aux2]
    outputs = model(features)

    print(f"Model output count: {len(outputs)}")
    print(f"Final output shape: {outputs[0].shape}")

    assert (
        len(outputs) == 3
    ), "Model should return 3 outputs (Final + 2 Aux) for deep supervision"
    assert (
        outputs[0].shape == targets.shape
    ), f"Output shape {outputs[0].shape} mismatch with target {targets.shape}"

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n--- Testing Loss Function ---")

    criterion = DeepSupervisionMAELoss(weights=Config.LOSS_WEIGHTS).to(device)
    loss = criterion(outputs, targets, masks)

    print(f"Computed Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n--- Testing Training Loop (1 Epoch) ---")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Execute one training epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch 1 Train Loss: {train_loss:.6f}")

    # Execute validation
    # Using train_loader here just to verify the function works without loading more data
    val_loss, val_mae_n, val_mae_e = validate(model, train_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation MAE (North): {val_mae_n:.4f}m")
    print(f"Validation MAE (East): {val_mae_e:.4f}m")

    # Save the model state for the inference step
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ==========================================
    # 6. Inference Pipeline
    # ==========================================
    print("\n--- Testing Inference Pipeline ---")

    # The predict_and_convert function handles:
    # 1. Loading test metadata
    # 2. Preprocessing raw test GNSS data
    # 3. Creating a test dataset (using the training scaler)
    # 4. Running inference
    # 5. Converting predicted offsets (meters) back to WGS84 (Lat/Lon)
    # 6. Formatting and saving the submission CSV

    submission_df = predict_and_convert(
        device=device,
        scaler=train_dataset.scaler,
        model_path=model_path,
        load_cached_data=False,  # Force re-processing of the debug subset
        debug_size=Config.DEBUG_SAMPLE_SIZE,
    )

    print(f"Submission DataFrame shape: {submission_df.shape}")

    # Verification of submission format
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in submission_df.columns, f"Missing required column: {col}"

    # Check for NaNs in predictions
    if not submission_df.empty:
        nans = submission_df[["LatitudeDegrees", "LongitudeDegrees"]].isna().sum().sum()
        if nans > 0:
            print(
                f"Warning: {nans} NaNs found in submission (likely due to missing input data in sample)."
            )
        else:
            print("Submission contains valid coordinates (no NaNs).")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
