import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device, calculate_rmse
from library.dataset import DenoisingDataset
from library.model import AG_CAC_ResUNet
from library.engine import train_model
from library.inference import predict_tiled, predict_tta, generate_submission_inference


def run_demo():
    # --- 1. Setup & Configuration Overrides ---
    print(">>> Setting up demonstration environment...")

    # Define a specific directory for this demo run to avoid overwriting main work
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(demo_dir, "submission.csv")

    # Reduce computational load for demo
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.PATCHES_PER_IMAGE = 5  # Extract fewer patches per image
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # --- 2. Create Data Subsets for Speed ---
    print("\n>>> Creating data subsets...")

    # Load original metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    df_test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Take a tiny subset (e.g., 5 images each)
    subset_train_path = os.path.join(demo_dir, "train_subset.csv")
    subset_val_path = os.path.join(demo_dir, "val_subset.csv")
    subset_test_path = os.path.join(demo_dir, "test_subset.csv")

    df_train_full.head(5).to_csv(subset_train_path, index=False)
    df_val_full.head(5).to_csv(subset_val_path, index=False)
    df_test_full.head(2).to_csv(subset_test_path, index=False)

    print(
        f"Created subsets: Train={len(df_train_full.head(5))}, Val={len(df_val_full.head(5))}, Test={len(df_test_full.head(2))}"
    )

    # --- 3. Verify Dataset Logic ---
    print("\n>>> Verifying Dataset logic...")

    # Instantiate Training Dataset
    train_dataset = DenoisingDataset(
        subset_train_path, mode="train", load_cached_data=False
    )

    # Check length: num_images * patches_per_image
    expected_len = 5 * Config.PATCHES_PER_IMAGE
    assert (
        len(train_dataset) == expected_len
    ), f"Dataset length mismatch. Expected {expected_len}, got {len(train_dataset)}"

    # Check item structure
    input_tensor, target_tensor, img_id = train_dataset[0]

    # Check shapes (C, H, W) -> (1, 128, 128) based on Config.PATCH_SIZE
    assert input_tensor.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Input tensor shape mismatch. Got {input_tensor.shape}"
    assert target_tensor.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Target tensor shape mismatch. Got {target_tensor.shape}"

    # Check value range [0, 1]
    assert (
        input_tensor.min() >= 0 and input_tensor.max() <= 1
    ), "Input tensor values out of range [0, 1]"

    print("Dataset verification passed.")

    # --- 4. Verify Model Logic ---
    print("\n>>> Verifying Model logic...")

    model = AG_CAC_ResUNet().to(device)

    # Create a dummy batch (B, C, H, W)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Forward pass
    with torch.no_grad():
        dummy_output = model(dummy_input)

    # Check output shape (should match input for U-Net)
    assert (
        dummy_output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Input: {dummy_input.shape}, Output: {dummy_output.shape}"

    print("Model architecture verification passed.")

    # --- 5. Verify Metric Logic ---
    print("\n>>> Verifying Metric logic...")
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.0, 1.0, 0.5])
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 0.0, "RMSE should be 0 for identical arrays"

    y_pred_off = np.array([1.0, 0.0, 1.5])  # diffs: 1, 1, 1 -> mse=1 -> rmse=1
    rmse_off = calculate_rmse(y_true, y_pred_off)
    assert abs(rmse_off - 1.0) < 1e-6, "RMSE calculation incorrect"
    print("Metric verification passed.")

    # --- 6. Run Training Loop ---
    print("\n>>> Running Training Loop (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_dataset = DenoisingDataset(subset_val_path, mode="val", load_cached_data=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # Run Training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=1,
    )

    # Verify model was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training loop completed and model saved.")

    # --- 7. Demonstrate Inference (Tiled & TTA) ---
    print("\n>>> Demonstrating Inference (Tiled & TTA)...")

    # Load the best saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Get a sample image from val set (full image, not patch)
    sample_input, _, _ = val_dataset[0]  # (1, H, W)
    sample_input = sample_input.unsqueeze(0).to(device)  # (1, 1, H, W)

    # Run Tiled Inference
    output_tiled = predict_tiled(
        model, sample_input, patch_size=Config.PATCH_SIZE, overlap=0.25, device=device
    )
    assert (
        output_tiled.shape == sample_input.shape
    ), "Tiled inference output shape mismatch"

    # Run TTA Inference
    output_tta = predict_tta(
        model, sample_input, patch_size=Config.PATCH_SIZE, overlap=0.25, device=device
    )
    assert output_tta.shape == sample_input.shape, "TTA inference output shape mismatch"

    print("Inference functions verified.")

    # --- 8. Generate Submission ---
    print("\n>>> Generating Submission for Test Subset...")

    test_dataset = DenoisingDataset(
        subset_test_path, mode="test", load_cached_data=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    generate_submission_inference(
        model=model,
        dataloader=test_loader,
        device=device,
        output_path=Config.SUBMISSION_FILE_PATH,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission columns missing"

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    assert len(str(sample_id).split("_")) == 3, f"Invalid ID format: {sample_id}"

    print("Submission generation verified.")
    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
