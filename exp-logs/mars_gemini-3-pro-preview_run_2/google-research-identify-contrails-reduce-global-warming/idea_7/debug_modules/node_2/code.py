import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encode, dice_coef_metric
from library.dataset import ContrailDataset, get_train_transform
from library.model import DeformableResNetUNet
from library.loss import HybridLoss
from library.train_engine import train_one_epoch, validate
from library.inference import predict_with_tta


def run_demo():
    print("--- Starting Contrail Detection Library Demo ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to run a fast check on a tiny subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small number for quick demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # Set seeds
    seed_everything(Config.SEED)
    print("Seeds set.")

    # 2. Dataset Verification
    print("\n--- Verifying Dataset ---")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Initialize Dataset
    dataset = ContrailDataset(train_df, transform=get_train_transform(), debug=True)

    print(f"Dataset initialized with {len(dataset)} samples (Debug Mode).")

    # Test __getitem__
    image, mask = dataset[0]

    # Check shapes
    # Image: (C, H, W) -> (6, 256, 256)
    assert image.shape == (
        6,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape (6, 256, 256), got {image.shape}"
    # Mask: (C, H, W) -> (1, 256, 256)
    assert mask.shape == (
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected mask shape (1, 256, 256), got {mask.shape}"

    # Check value ranges (Ash vector is [0, 1], Temporal Difference is [-1, 1])
    assert (
        image.min() >= -1.0 and image.max() <= 1.0
    ), "Image values should be normalized between -1 and 1."

    print("Dataset shapes and value ranges verified.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    model = DeformableResNetUNet(
        n_channels=Config.N_CHANNELS,
        n_classes=1,
        pretrained=False,  # False for speed, we just check architecture
    ).to(device)

    # Create a dummy batch
    dummy_batch = image.unsqueeze(0).to(device)  # (1, 6, 256, 256)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_batch)

    assert output.shape == (
        1,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected output shape (1, 1, 256, 256), got {output.shape}"

    print("Model forward pass successful. Output shape verified.")

    # 4. Loss Function Verification
    print("\n--- Verifying Loss Function ---")
    criterion = HybridLoss(bce_weight=0.5, dice_weight=0.5).to(device)

    # Dummy targets
    dummy_target = mask.unsqueeze(0).to(device)  # (1, 1, 256, 256)

    # Calculate loss
    loss = criterion(output, dummy_target)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"HybridLoss computed successfully: {loss.item():.4f}")

    # 5. Metric and Utility Verification
    print("\n--- Verifying Metrics and Utilities ---")

    # Test RLE Encoding
    # Create a simple 2x2 mask: [[0, 1], [0, 0]]
    # Flatten order 'F' (top-down, left-right): 0 (0,0), 0 (1,0), 1 (0,1), 0 (1,1) -> [0, 0, 1, 0]
    # Run: Start at index 3 (1-based), length 1.
    simple_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    rle_result = rle_encode(simple_mask)
    expected_rle = "3 1"
    assert (
        rle_result == expected_rle
    ), f"RLE failed. Expected '{expected_rle}', got '{rle_result}'"
    print("RLE Encoding verified.")

    # Test Dice Coefficient
    # Perfect overlap
    y_true = torch.ones((1, 1, 10, 10))
    y_pred = torch.ones((1, 1, 10, 10))  # Logits need to be high for sigmoid > 0.5
    # Since metric takes logits or probs?
    # utils.dice_coef_metric says: "Assumes values are probabilities (0-1) for thresholding."
    # So we pass probabilities.
    dice_score = dice_coef_metric(y_pred, y_true, threshold=0.5)
    assert abs(dice_score - 1.0) < 1e-5, f"Expected Dice 1.0, got {dice_score}"

    # No overlap
    y_pred_zero = torch.zeros((1, 1, 10, 10))
    dice_score_zero = dice_coef_metric(y_pred_zero, y_true, threshold=0.5)
    assert (
        abs(dice_score_zero - 0.0) < 1e-5
    ), f"Expected Dice 0.0, got {dice_score_zero}"
    print("Dice metric verified.")

    # 6. Training Loop Simulation
    print("\n--- Simulating Training Loop (1 Epoch) ---")

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()

    # Run one epoch
    epoch_loss = train_one_epoch(
        model, dataloader, optimizer, criterion, device, scaler
    )
    print(f"Training Epoch completed. Avg Loss: {epoch_loss:.4f}")

    # Run validation
    val_dice = validate(
        model, dataloader, device
    )  # Using train loader as valid for demo
    print(f"Validation completed. Dice: {val_dice:.4f}")

    # 7. Inference TTA Verification
    print("\n--- Verifying Inference TTA ---")
    model.eval()
    with torch.no_grad():
        # Use the dummy batch from earlier
        # predict_with_tta expects raw images and returns probabilities
        tta_output = predict_with_tta(model, dummy_batch)

        assert tta_output.shape == (
            1,
            1,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"TTA output shape mismatch. Got {tta_output.shape}"

        # Check values are probabilities [0, 1]
        assert (
            tta_output.min() >= 0 and tta_output.max() <= 1
        ), "TTA output should be probabilities between 0 and 1"

    print("TTA Inference verified.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
