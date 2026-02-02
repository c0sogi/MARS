import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.dataset import UWMadisonDataset
from library.model import HRNetSegmentation
from library.losses import BCETverskyLoss
from library.train import train_one_epoch, validate, get_transforms
from library.utils import rle_encode, rle_decode, calculate_dice
from library.inference import predict_sliding_window, post_process_volume


def run_demo():
    print("=== Starting Demonstration of Stomach/Intestine Segmentation Pipeline ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config defaults to run a small, fast experiment
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set hyperparams for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.SAMPLE_SIZE = 40  # Use only 40 samples
    Config.NUM_WORKERS = 2  # Reduce workers for small data
    Config.PRETRAINED = False  # Disable downloading weights for demo

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Sample Size: {Config.SAMPLE_SIZE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print("    Configuration complete.\n")

    # -------------------------------------------------------------------------
    # 2. Dataset Loading and Verification
    # -------------------------------------------------------------------------
    print("[2] Verifying Dataset Logic...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Instantiate Dataset
    # We use load_cached_data=False to demonstrate processing logic
    train_dataset = UWMadisonDataset(
        df_train,
        phase="train",
        transform=get_transforms("train"),
        load_cached_data=False,
    )

    # Fetch a single sample
    sample = train_dataset[0]

    # Assertions
    image = sample["image"]
    mask = sample["mask"]

    # Check 1: 2.5D Stacking (Should be 3 channels: slice i-1, i, i+1)
    assert (
        image.shape[0] == 3
    ), f"Expected 3 channels (2.5D stack), got {image.shape[0]}"

    # Check 2: Mask Channels (Should be 3 classes)
    assert (
        mask.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} mask channels, got {mask.shape[0]}"

    # Check 3: Spatial alignment
    assert (
        image.shape[1:] == mask.shape[1:]
    ), f"Image and mask spatial dimensions mismatch: {image.shape} vs {mask.shape}"

    print(f"    Sample ID: {sample['id']}")
    print(f"    Image Shape: {tuple(image.shape)} (Channels, Height, Width)")
    print(f"    Mask Shape: {tuple(mask.shape)}")
    print("    Dataset verification passed.\n")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("[3] Initializing Model and Checking Forward Pass...")

    device = Config.DEVICE
    model = HRNetSegmentation(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(device)
    model.eval()

    # Create a dummy batch
    dummy_input = image.unsqueeze(0).to(device)  # Add batch dim: (1, 3, H, W)

    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    # Output should be (B, Num_Classes, H, W)
    expected_shape = (1, Config.NUM_CLASSES, image.shape[1], image.shape[2])
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"    Model Output Shape: {tuple(output.shape)}")
    print("    Model initialization passed.\n")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("[4] Verifying Loss Function...")

    criterion = BCETverskyLoss(alpha=Config.TVERSKY_ALPHA, beta=Config.TVERSKY_BETA)

    dummy_target = mask.unsqueeze(0).to(device)
    loss = criterion(output, dummy_target)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"    Calculated Loss: {loss.item():.6f}")
    print("    Loss function verification passed.\n")

    # -------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("[5] Simulating Training Loop (1 Epoch)...")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    # Validation dataset (using train metadata for demo purposes to ensure files exist)
    val_dataset = UWMadisonDataset(
        df_train.iloc[:20],  # Small subset
        phase="val",
        transform=get_transforms("val"),
        load_cached_data=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    scaler = GradScaler()

    # Run Training Step
    print("    Running training step...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device
    )

    # Run Validation Step
    print("    Running validation step...")
    val_loss, val_dice = validate(model, val_loader, criterion, device)

    print(
        f"    Epoch 1 Results -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"
    )

    # Save model for inference demo
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("    Training simulation passed.\n")

    # -------------------------------------------------------------------------
    # 6. Inference and Post-Processing
    # -------------------------------------------------------------------------
    print("[6] Demonstrating Inference and Post-Processing...")

    # Load the saved model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Take a sample image for sliding window inference
    sample_val = val_dataset[0]
    val_img = sample_val["image"].to(device)  # (3, H, W)

    # Run sliding window
    prob_map = predict_sliding_window(model, val_img, device)

    assert prob_map.shape == (
        Config.NUM_CLASSES,
        sample_val["orig_shape"][0],
        sample_val["orig_shape"][1],
    ), f"Inference output shape mismatch: {prob_map.shape}"

    print(f"    Sliding Window Output: {prob_map.shape}")

    # Demonstrate 3D Post-Processing (Simulating a volume of 5 slices)
    # We'll just stack the same probability map 5 times
    volume_probs = np.stack([prob_map[0]] * 5, axis=0)  # (Depth, H, W) for class 0
    print(f"    Simulated Volume Shape: {volume_probs.shape}")

    processed_mask = post_process_volume(volume_probs, threshold=0.5)

    assert (
        processed_mask.shape == volume_probs.shape
    ), "Post-processed mask shape mismatch"
    assert processed_mask.dtype == np.uint8, "Mask should be uint8"

    print("    Inference and post-processing passed.\n")

    # -------------------------------------------------------------------------
    # 7. Utility Verification (RLE)
    # -------------------------------------------------------------------------
    print("[7] Verifying RLE Encoding/Decoding...")

    # Create a simple synthetic mask (10x10)
    synthetic_mask = np.zeros((10, 10), dtype=np.uint8)
    synthetic_mask[2:5, 2:5] = 1  # 3x3 square

    # Encode
    rle_str = rle_encode(synthetic_mask)

    # Decode
    decoded_mask = rle_decode(rle_str, (10, 10))

    # Check equality
    assert np.array_equal(synthetic_mask, decoded_mask), "RLE Round-trip failed"

    print(f"    Original Area: {synthetic_mask.sum()}")
    print(f"    Decoded Area: {decoded_mask.sum()}")
    print(f"    RLE String: {rle_str}")
    print("    RLE verification passed.\n")

    print("=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
