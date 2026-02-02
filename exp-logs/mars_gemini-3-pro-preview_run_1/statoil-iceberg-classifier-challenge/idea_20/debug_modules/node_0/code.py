import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import Adam

# Import from the provided library
from library.config import Config, set_seed
from library.data import get_loaders, get_test_loader
from library.model import SymmetrizedResNet18
from library.engine import train_one_epoch, evaluate, generate_submission


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print("Setting up configuration and seeding...")
    set_seed(Config.SEED)

    # Override Config for a quick demonstration run
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for demonstration
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    Config.setup()

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("\nInitializing DataLoaders...")
    # We force load_cached_data=False to demonstrate processing logic at least once,
    # or rely on the library's internal check.
    # Since we want to be fast and the cache might not exist for this specific 'idea_20' folder,
    # the library will generate it.
    train_loader, val_loader = get_loaders(load_cached_data=True)
    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # --- Verification: Data Shapes ---
    print("Verifying data shapes...")
    images, angles, labels = next(iter(train_loader))

    # Expected: (B, 3, 224, 224) for images (after resizing/stacking)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"

    # Expected: (B,) for angles
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect angle shape: {angles.shape}"

    # Expected: (B, 1) for labels
    assert labels.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect label shape: {labels.shape}"

    print("Data shapes verified.")

    # 3. Model Initialization
    print("\nInitializing SymmetrizedResNet18...")
    model = SymmetrizedResNet18()
    model.to(device)

    # --- Verification: Model Invariance ---
    # The SymmetrizedResNet18 averages predictions over 4 geometric views (Original, FlipLR, FlipUD, Rot180).
    # Therefore, f(x) should be approximately equal to f(flip_lr(x)).
    print("Verifying model geometric invariance...")
    model.eval()
    with torch.no_grad():
        # Take one sample image and angle
        sample_img = images[0:1].to(device)  # (1, 3, 224, 224)
        sample_angle = angles[0:1].to(device)  # (1,)

        # Original prediction
        out_original = model(sample_img, sample_angle)

        # Flip the input image horizontally (dim 3)
        img_flipped = torch.flip(sample_img, dims=[3])
        out_flipped = model(img_flipped, sample_angle)

        # Check difference
        diff = torch.abs(out_original - out_flipped).item()
        print(f"Difference between Original and Flipped input predictions: {diff:.6f}")

        # Tolerance: Floating point arithmetic might cause tiny deviations,
        # but logic should make them very close (< 1e-5).
        assert diff < 1e-5, "Model is not invariant to horizontal flips as expected!"

    print("Model invariance verified.")

    # 4. Training Loop (Short Demo)
    print(f"\nStarting training for {Config.MAX_EPOCHS} epochs...")

    optimizer = Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss = evaluate(model, val_loader, criterion, device, epoch)

        # Basic assertion to ensure loss is valid
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # 5. Submission Generation
    print("\nGenerating submission...")
    generate_submission(model, test_loader, test_ids, device)

    # --- Verification: Submission File ---
    print("Verifying submission file...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    # test.json has 1 line in the provided sample description, but the sample_submission has 321 rows.
    # The metadata generation script output says "Test set shape: (321, 4)".
    # So we expect 321 predictions.
    expected_rows = 321
    if len(df_sub) != expected_rows:
        # If the test set size differs in the actual environment, we adjust,
        # but based on provided metadata logs, it is 321.
        print(
            f"Warning: Expected {expected_rows} rows, got {len(df_sub)}. Checking against test_ids length."
        )
        assert len(df_sub) == len(
            test_ids
        ), "Submission rows do not match number of test IDs."

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], f"Incorrect columns: {df_sub.columns}"

    # Check value range
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Probabilities out of range [0, 1]"

    print("Submission verified successfully.")
    print("\nDemo execution completed.")


if __name__ == "__main__":
    run_demo()
