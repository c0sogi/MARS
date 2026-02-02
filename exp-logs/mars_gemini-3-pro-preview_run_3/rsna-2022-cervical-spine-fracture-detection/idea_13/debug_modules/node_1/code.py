import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import library modules
from library.config import Config
from library import utils
from library import dataset
from library import model
from library import engine


def main():
    # 1. Setup & Configuration Overrides
    # ---------------------------------------------------------
    print("--- Setting up Demonstration Configuration ---")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set reproducible seed
    utils.seed_everything(seed=42)

    # Override Config for speed and demonstration purposes
    # Reduce volume depth to speed up DICOM loading and processing
    Config.NUM_SLICES = 8
    # Use a tiny batch size suitable for the demo environment
    Config.BATCH_SIZE = 2
    # Run only 1 epoch
    Config.EPOCHS = 1
    # Limit dataset size to a few samples to ensure quick runtime
    Config.N_SAMPLES = 6
    # Ensure cache directory is clean or specific for demo (optional, but good for isolation)
    Config.CACHE_DIR = "./working/demo_cache"
    Config.create_directories()

    # Define device
    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Num Slices: {Config.NUM_SLICES}")
    print(f"N Samples: {Config.N_SAMPLES}")

    # 2. Dataset & DataLoader Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Dataset and DataLoader ---")

    # Initialize Train Dataset
    train_ds = dataset.RSNADataset(subset="train")

    # Verify length
    print(f"Train Dataset Length: {len(train_ds)}")
    assert len(train_ds) <= Config.N_SAMPLES, "Dataset did not respect N_SAMPLES limit."

    # Fetch one item to check shapes
    global_input, local_input, targets = train_ds[0]

    print(f"Global Input Shape: {global_input.shape}")
    print(f"Local Input Shape: {local_input.shape}")
    print(f"Targets Shape: {targets.shape}")

    # Assertions for shapes
    # Expected: (NUM_SLICES, 3, 256, 256)
    expected_shape = (Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        global_input.shape == expected_shape
    ), f"Global input shape mismatch. Got {global_input.shape}"
    assert (
        local_input.shape == expected_shape
    ), f"Local input shape mismatch. Got {local_input.shape}"
    # Expected: (8,) -> C1-C7 + Patient Overall
    assert targets.shape == (8,), f"Target shape mismatch. Got {targets.shape}"

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple demo debugging
        drop_last=True,
    )

    val_ds = dataset.RSNADataset(subset="val")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("DataLoaders created successfully.")

    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    net = model.DualStreamConvNeXt()
    net = net.to(device)

    # Create a dummy batch matching the DataLoader output
    # Shape: (Batch, Slices, Channels, H, W)
    dummy_global = torch.randn(
        Config.BATCH_SIZE, Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    dummy_local = torch.randn(
        Config.BATCH_SIZE, Config.NUM_SLICES, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)

    # Run forward pass
    with torch.no_grad():
        outputs = net(dummy_global, dummy_local)

    print(f"Model Output Shape: {outputs.shape}")

    # Assert output shape: (Batch, Num_Classes)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    print("Model forward pass successful.")

    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # We use the engine.fit function which handles the loop, validation, and saving
    engine.fit(net, train_loader, val_loader, device)

    # Verify that the model checkpoint was saved
    best_model_path = os.path.join("working", "best_model.pth")
    assert os.path.exists(best_model_path), "Training failed to save 'best_model.pth'."
    print("Training loop completed and model saved.")

    # 5. Inference & Submission
    # ---------------------------------------------------------
    print("\n--- Executing Inference ---")

    # Initialize Test Dataset
    # Note: We limit samples here too via Config.N_SAMPLES implicitly or we can force it
    test_ds = dataset.RSNADataset(subset="test")

    # Create Test Loader
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Run Inference
    engine.inference(net, test_loader, device)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Inference failed to create submission file."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(sub_df)}")
    print("Sample Submission Rows:")
    print(sub_df.head())

    # Logic Check: Number of rows should be Num_Test_Samples * 8
    # Note: test_ds length might be smaller than Config.N_SAMPLES if the CSV is smaller,
    # but here we capped it.
    expected_rows = len(test_ds) * 8
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    assert (
        "row_id" in sub_df.columns and "fractured" in sub_df.columns
    ), "Submission file missing required columns."

    # 6. Metric Validation Unit Test
    # ---------------------------------------------------------
    print("\n--- Verifying Competition Metric Logic ---")

    # Create synthetic ground truth and predictions
    # Case 1: Perfect predictions
    y_true = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]])
    # Predictions close to truth (clipped by metric function anyway)
    y_pred_perfect = np.array(
        [
            [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
            [0.999, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 0.999],
        ]
    )

    loss_perfect = utils.competition_metric(y_true, y_pred_perfect)
    print(f"Loss (Near Perfect): {loss_perfect:.6f}")
    assert loss_perfect < 0.01, "Metric should be very low for perfect predictions."

    # Case 2: Wrong predictions
    y_pred_wrong = 1.0 - y_pred_perfect
    loss_wrong = utils.competition_metric(y_true, y_pred_wrong)
    print(f"Loss (Wrong): {loss_wrong:.6f}")
    assert loss_wrong > 1.0, "Metric should be high for wrong predictions."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
