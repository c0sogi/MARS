import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config, setup_directories, set_seed
from library.dataset import get_dataloaders
from library.model import SETIEfficientNet
from library.engine import fit, predict_and_submit
from library.utils import get_score


def main():
    # 1. Setup and Configuration Overrides for Speed
    print("Initializing demonstration...")

    # Override Config for a quick demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample size for speed
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small data

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Create necessary directories (working, submission, cache)
    setup_directories()

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n--- Testing Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Train Loader
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions for Data Integrity
    # Expected shape: (Batch, Channels=6, Height=224, Width=224)
    expected_image_shape = (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    )
    # Note: The last batch might be smaller if drop_last=False, but train_loader has drop_last=True
    if images.shape != expected_image_shape:
        # If dataset size < batch size, drop_last might result in empty or specific behavior,
        # but with size 50 and batch 8, we expect full batches.
        assert (
            images.shape == expected_image_shape
        ), f"Image shape mismatch. Expected {expected_image_shape}, got {images.shape}"

    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"

    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"

    print("Data Loading verification passed.")

    # 3. Model Instantiation Demonstration
    print("\n--- Testing Model Instantiation ---")
    model = SETIEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # False for speed/offline demo, usually True
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # Forward pass verification
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for Model Output
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {outputs.shape}"

    print("Model instantiation and forward pass verification passed.")

    # 4. Training Loop Demonstration
    print("\n--- Testing Training Loop (Engine) ---")

    # Setup training components
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Run fit function (1 epoch as configured above)
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=Config.NUM_EPOCHS,
        patience=1,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {Config.MODEL_SAVE_PATH}"

    print("Training loop execution passed.")

    # 5. Inference and Submission Demonstration
    print("\n--- Testing Inference and Submission ---")

    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate submission
    predict_and_submit(
        model=model,
        test_loader=test_loader,
        device=device,
        submission_path=Config.SUBMISSION_PATH,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head(3)}")

    # Check submission format
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Submission columns mismatch. Expected ['id', 'target'], got {list(df_sub.columns)}"

    # Check length (should match debug sample size or full test size depending on logic)
    # In get_dataloaders, debug mode samples the test set as well.
    expected_len = min(len(pd.read_csv(Config.TEST_CSV)), Config.DEBUG_SAMPLE_SIZE)
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print("Inference and submission verification passed.")

    # 6. Utility Verification
    print("\n--- Testing Utilities ---")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = get_score(y_true, y_pred)
    print(f"Calculated AUC Score: {score}")

    assert 0 <= score <= 1, "AUC score must be between 0 and 1"

    print("Utility verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
