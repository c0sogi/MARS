import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp

# Import from the provided library files
from library.utils import seed_everything, rle_encode, rle_decode, calculate_map_score
from library.dataset import get_dataloaders, get_test_loader
from library.model import SaltUNetPlusPlus
from library.loss import BCEDiceLoss, LovaszHingeLoss
from library.engine import train_one_epoch, validate_one_epoch


def main():
    # 1. Setup and Configuration
    print("--- Setting up environment ---")
    SEED = 42
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparameters optimized for demonstration speed
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    DEBUG_MODE = True  # Limits dataset to 100 samples
    EPOCHS = 1

    # 2. Demonstrate Library Utils
    print("\n--- Testing Library Utils ---")

    # Test RLE Encoding/Decoding
    # Create a dummy 101x101 mask with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE decoded mask does not match original"
    print("RLE Encode/Decode verification passed.")

    # Test mAP Calculation
    # Create dummy predictions and targets (Batch=2, H=101, W=101)
    # Sample 1: Perfect match
    # Sample 2: No overlap
    pred_tensor = torch.zeros((2, 101, 101), dtype=torch.float32)
    target_tensor = torch.zeros((2, 101, 101), dtype=torch.uint8)

    # Sample 1 (Perfect match, high confidence)
    pred_tensor[0, 50:60, 50:60] = 0.9
    target_tensor[0, 50:60, 50:60] = 1

    # Sample 2 (Mismatch)
    pred_tensor[1, 10:20, 10:20] = 0.9
    target_tensor[1, 80:90, 80:90] = 1

    # Calculate score
    score = calculate_map_score(pred_tensor, target_tensor, decision_threshold=0.5)
    # Sample 1 should have AP=1.0 (IoU=1.0 > all thresholds)
    # Sample 2 should have AP=0.0 (IoU=0.0 < all thresholds)
    # Mean AP should be 0.5
    print(f"Calculated mAP: {score}")
    assert 0.49 < score < 0.51, f"Expected mAP around 0.5, got {score}"
    print("mAP calculation verification passed.")

    # 3. Demonstrate Dataset Loading
    print("\n--- Testing Dataset and Dataloaders ---")

    # Get Train/Val Loaders in Debug mode
    train_loader, val_loader = get_dataloaders(
        fold_idx=0,
        n_folds=5,
        batch_size=BATCH_SIZE,
        load_cached_data=False,  # Force reload from metadata for demo
        num_workers=NUM_WORKERS,
        debug=DEBUG_MODE,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    ids = batch["id"]

    # Expected shape: (B, 3, 128, 128) -> 3 channels (Seismic, Seismic, Depth), 128x128 padded
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")

    assert images.shape == (BATCH_SIZE, 3, 128, 128), "Incorrect image batch shape"
    assert masks.shape == (BATCH_SIZE, 1, 128, 128), "Incorrect mask batch shape"
    assert len(ids) == BATCH_SIZE, "Incorrect IDs length"
    print("Dataset shapes verification passed.")

    # Test Test-Loader
    test_loader = get_test_loader(batch_size=BATCH_SIZE, debug=DEBUG_MODE)
    test_batch = next(iter(test_loader))
    assert test_batch["image"].shape == (
        BATCH_SIZE,
        3,
        128,
        128,
    ), "Incorrect test image shape"
    print("Test loader verification passed.")

    # 4. Demonstrate Model Initialization and Forward Pass
    print("\n--- Testing Model ---")

    model = SaltUNetPlusPlus(
        encoder_name="resnext50_32x4d", in_channels=3, classes=1, deep_supervision=True
    )
    model.to(device)

    # Forward Pass (Training Mode - Deep Supervision)
    model.train()
    images = images.to(device)
    outputs = model(images)

    assert isinstance(
        outputs, list
    ), "Model in training mode should return a list (Deep Supervision)"
    assert len(outputs) == 4, "Model should return 4 outputs for Deep Supervision"
    assert outputs[-1].shape == (BATCH_SIZE, 1, 128, 128), "Final output shape mismatch"
    print("Model training forward pass passed.")

    # Forward Pass (Eval Mode)
    model.eval()
    with torch.no_grad():
        output = model(images)

    assert torch.is_tensor(output), "Model in eval mode should return a single tensor"
    assert output.shape == (BATCH_SIZE, 1, 128, 128), "Eval output shape mismatch"
    print("Model eval forward pass passed.")

    # 5. Demonstrate Loss Functions
    print("\n--- Testing Loss Functions ---")

    bce_dice_loss = BCEDiceLoss()
    lovasz_loss = LovaszHingeLoss()

    masks = masks.to(device)

    # Calculate BCE Dice Loss
    # Note: BCEDiceLoss expects logits
    loss_val_bce = bce_dice_loss(output, masks)
    print(f"BCE Dice Loss: {loss_val_bce.item()}")
    assert not torch.isnan(loss_val_bce), "BCE Dice Loss returned NaN"

    # Calculate Lovasz Loss
    loss_val_lovasz = lovasz_loss(output, masks)
    print(f"Lovasz Loss: {loss_val_lovasz.item()}")
    assert not torch.isnan(loss_val_lovasz), "Lovasz Loss returned NaN"
    print("Loss functions verification passed.")

    # 6. Demonstrate Engine (Training Loop)
    print("\n--- Testing Engine (Train/Val Loop) ---")

    # Setup Optimizer and Scaler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = amp.GradScaler()

    # Train One Epoch
    print(f"Starting training for {EPOCHS} epoch...")
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        criterion=bce_dice_loss,
        device=device,
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive"

    # Validate One Epoch
    print("Starting validation...")
    val_loss, val_map = validate_one_epoch(
        model=model, loader=val_loader, criterion=bce_dice_loss, device=device
    )
    print(f"Epoch 1 Val Loss: {val_loss:.4f}")
    print(f"Epoch 1 Val mAP: {val_map:.4f}")

    assert val_loss > 0, "Validation loss should be positive"
    assert 0 <= val_map <= 1, "mAP score should be between 0 and 1"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    main()
