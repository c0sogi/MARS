import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import DenoisingDataset
from library.model import DS_AG_CAC_ResUNet
from library.loss import MultiScaleMSELoss
from library.train import train_one_epoch
from library.inference import predict_tiled, generate_submission, apply_tta
from library.utils import seed_everything, get_device


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuring Environment ---")

    # Override Config for speed and demonstration purposes
    Config.DEBUG_SUBSET_SIZE = 4  # Only use 4 images
    Config.PATCHES_PER_IMAGE = 2  # Only 2 patches per image for training
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = (
        0  # Use 0 for simple debugging/demo to avoid multiprocessing overhead
    )
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create a temporary test metadata file with fewer images for fast submission generation
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    subset_test_df = full_test_df.head(2)  # Use only 2 test images
    temp_test_meta_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")
    subset_test_df.to_csv(temp_test_meta_path, index=False)
    Config.TEST_METADATA_PATH = temp_test_meta_path

    # Ensure directories exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")
    print("Configuration updated for demo run.")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader
    # -------------------------------------------------------------------------
    print("\n--- 2. Validating Dataset & DataLoader ---")

    # Initialize Dataset
    train_dataset = DenoisingDataset(
        Config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    # Expected length: subset_size * patches_per_image
    expected_len = Config.DEBUG_SUBSET_SIZE * Config.PATCHES_PER_IMAGE
    print(f"Dataset Length: {len(train_dataset)} (Expected: {expected_len})")
    assert len(train_dataset) == expected_len, "Dataset length mismatch!"

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Fetch one batch
    noisy_batch, clean_batch, ids = next(iter(train_loader))

    print(f"Batch Shapes -> Noisy: {noisy_batch.shape}, Clean: {clean_batch.shape}")

    # Validate Shapes (B, C, H, W) -> (2, 1, 128, 128)
    assert noisy_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    assert clean_batch.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    print("Dataset and DataLoader validated successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture
    # -------------------------------------------------------------------------
    print("\n--- 3. Validating Model Architecture ---")

    model = DS_AG_CAC_ResUNet().to(device)

    # Move batch to device
    noisy_batch = noisy_batch.to(device)

    # Forward Pass
    outputs = model(noisy_batch)

    # Check Deep Supervision outputs
    # Should return a list: [final, aux4, aux3, aux2]
    is_list = isinstance(outputs, list)
    num_outputs = len(outputs)
    print(f"Model Output Type: {type(outputs)}")
    print(f"Number of Outputs (Deep Supervision): {num_outputs}")

    assert is_list, "Model should return a list when deep supervision is enabled."
    assert num_outputs == 4, "Expected 4 outputs (1 final + 3 aux)."

    # Check shape of final output
    final_pred = outputs[0]
    print(f"Final Prediction Shape: {final_pred.shape}")
    assert final_pred.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    )
    print("Model architecture validated successfully.")

    # -------------------------------------------------------------------------
    # 4. Loss Function
    # -------------------------------------------------------------------------
    print("\n--- 4. Validating Loss Function ---")

    criterion = MultiScaleMSELoss(weights=[1.0, 0.5, 0.5, 0.5]).to(device)
    clean_batch = clean_batch.to(device)

    loss = criterion(outputs, noisy_batch, clean_batch)
    print(f"Calculated Loss: {loss.item()}")

    assert torch.is_tensor(loss), "Loss should be a tensor."
    assert loss.item() >= 0, "Loss should be non-negative."
    print("Loss function validated successfully.")

    # -------------------------------------------------------------------------
    # 5. Training Loop (One Epoch)
    # -------------------------------------------------------------------------
    print("\n--- 5. Validating Training Loop ---")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch Finished. Average Loss: {avg_loss:.6f}")

    assert avg_loss > 0, "Training loss should be positive."

    # Save model for inference step
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    print("Model checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Inference & TTA
    # -------------------------------------------------------------------------
    print("\n--- 6. Validating Inference & TTA ---")

    model.eval()

    # Create a dummy full-size image (1, 1, 300, 300) to test tiling
    dummy_h, dummy_w = 300, 300
    dummy_img = torch.rand((1, 1, dummy_h, dummy_w)).to(device)

    # Test Tiled Prediction
    with torch.no_grad():
        clean_pred = predict_tiled(model, dummy_img, device)

    print(f"Input Shape: {dummy_img.shape}")
    print(f"Prediction Shape: {clean_pred.shape}")

    assert (
        clean_pred.shape == dummy_img.shape
    ), "Prediction shape must match input shape."
    assert (
        clean_pred.min() >= 0 and clean_pred.max() <= 1
    ), "Prediction values should be in [0, 1]."

    # Test TTA specifically on a patch
    patch = dummy_img[:, :, :128, :128]
    tta_pred = apply_tta(model, patch)
    assert tta_pred.shape == patch.shape, "TTA output shape mismatch."
    print("Inference and TTA validated successfully.")

    # -------------------------------------------------------------------------
    # 7. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- 7. Validating Submission Generation ---")

    # This function loads the model from checkpoint and processes the test set
    # We pointed Config.TEST_METADATA_PATH to a 2-image subset earlier.
    generate_submission(
        model_path=Config.MODEL_CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
    )

    # Verify file existence and content
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(df_sub)}")
    print(f"Submission Columns: {list(df_sub.columns)}")

    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    assert len(sample_id.split("_")) == 3, f"Invalid ID format: {sample_id}"

    print("Submission generation validated successfully.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
