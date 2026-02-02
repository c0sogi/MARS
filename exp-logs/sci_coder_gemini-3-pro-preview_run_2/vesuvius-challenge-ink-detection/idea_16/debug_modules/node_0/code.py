import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    load_metadata,
    process_fragment_mips,
    rle_encode,
    save_checkpoint,
)
from library.dataset import get_dataset, InkDataset
from library.model import HybridSegFormer
from library.metrics import BCEDiceLoss, fbeta_score_numpy
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    predict_tiled,
    predict_with_z_scanning,
)


def run_demo():
    print("=== Starting Vesuvius Ink Detection Pipeline Demo ===")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Override Config for speed in this demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Data Loading & Preprocessing Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Data Loading & MIP Generation ---")

    # Load training metadata
    train_df = load_metadata("train")
    print(f"Total training patches available: {len(train_df)}")

    # Subset metadata for rapid demonstration
    demo_train_df = train_df.iloc[:8].copy()  # 2 batches
    demo_val_df = load_metadata("validation").iloc[:4].copy()  # 1 batch

    # Verify MIP Generation (3D -> 2D)
    # We use the first fragment found in the training set
    sample_frag_id = demo_train_df.iloc[0]["fragment_id"]
    sample_vol_path = demo_train_df.iloc[0]["volume_path"]

    print(f"Generating MIPs for Fragment {sample_frag_id}...")
    mips = process_fragment_mips(
        fragment_id=sample_frag_id,
        volume_path=sample_vol_path,
        z_start=Config.TRAIN_Z_START,
        load_cached_data=True,  # Will save to ./working/idea_16/
    )

    # Assertions for MIP structure
    assert isinstance(mips, np.ndarray), "MIPs must be a numpy array"
    assert mips.shape[0] == 3, f"Expected 3 channels, got {mips.shape[0]}"
    assert mips.dtype == np.float32, "MIPs should be float32"
    assert (
        mips.min() >= 0.0 and mips.max() <= 1.0
    ), "MIPs should be normalized to [0, 1]"
    print(f"MIP Generation Successful. Shape: {mips.shape}")

    # 3. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Dataset & DataLoader ---")

    # Initialize Dataset with the subset
    train_dataset = InkDataset(demo_train_df, split="train")
    val_dataset = InkDataset(demo_val_df, split="validation")

    # Test __getitem__
    img, lbl = train_dataset[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label Shape: {lbl.shape}")

    assert img.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect image tensor shape"
    assert lbl.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect label tensor shape"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    print("DataLoaders initialized.")

    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Architecture ---")

    # Initialize model
    # pretrained=False to avoid downloading weights during this constrained run
    model = HybridSegFormer(pretrained=False).to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Model output shape mismatch"
    print("Model forward pass successful.")

    # 5. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("\n--- Testing Training Loop ---")

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for 1 epoch
    epoch_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    assert not np.isnan(epoch_loss), "Training loss is NaN"

    # Validate
    print("\n--- Testing Validation Loop ---")
    val_loss, val_score = valid_one_epoch(model, val_loader, criterion, device)
    assert 0.0 <= val_score <= 1.0, "F0.5 score out of range"

    # Save Checkpoint
    save_checkpoint(model, optimizer, 1, val_score, filename="demo_model.pth")
    print("Training cycle completed.")

    # 6. Metric Utility Verification (RLE)
    # -------------------------------------------------------------------------
    print("\n--- Testing RLE Encoding ---")
    # Create a simple mask: 0 0 1 1 1 0 0 1 0
    # Indices (1-based):  1 2 3 4 5 6 7 8 9
    # Runs: Start 3, Len 3; Start 8, Len 1
    dummy_mask = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    # Flattened: 0 0 1 1 1 0 0 1 0
    encoded = rle_encode(dummy_mask)
    expected = "3 3 8 1"
    print(f"RLE Result: '{encoded}'")
    assert (
        encoded == expected
    ), f"RLE Encoding failed. Expected '{expected}', got '{encoded}'"

    # 7. Inference Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n--- Testing Inference Pipeline ---")

    # A. Tiled Prediction on a single image (using the MIPs generated earlier)
    print("Running Tiled Prediction...")
    # mips is (3, H, W) float32 [0,1]
    prob_map = predict_tiled(model, mips, device, batch_size=2)

    assert (
        prob_map.shape == mips.shape[1:]
    ), "Probability map spatial dimensions mismatch"
    assert (
        prob_map.min() >= 0.0 and prob_map.max() <= 1.0
    ), "Probability map values out of range"
    print("Tiled prediction successful.")

    # B. Full Z-Scanning Inference on Test Data
    print("Running Z-Scanning Inference on Test Set...")
    test_df = load_metadata("test")

    # We will run this on the provided test metadata.
    # Note: The test folder 'a' exists in the input description.
    predictions = predict_with_z_scanning(model, test_df, device)

    assert isinstance(predictions, list), "Predictions should be a list"
    assert len(predictions) == len(
        test_df
    ), "Missing predictions for some test fragments"
    assert (
        "Id" in predictions[0] and "Predicted" in predictions[0]
    ), "Prediction format incorrect"

    # 8. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission File ---")
    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv("submission.csv", index=False)
    print("submission.csv created successfully.")
    print(sub_df.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
