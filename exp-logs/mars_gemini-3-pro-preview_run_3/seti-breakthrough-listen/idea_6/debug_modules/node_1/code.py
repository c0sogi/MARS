import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.dataset import get_train_val_loaders, get_test_loader
from library.model import SiameseDifferenceNet
from library.utils import mixup_data, seed_everything
from library import engine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # --- 1. Setup & Configuration ---
    print(">>> Setting up demonstration environment...")
    seed_everything(42)

    # Define demo paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    model_save_path = os.path.join(demo_dir, "best_model.pth")
    submission_save_path = os.path.join(demo_dir, "submission.csv")

    # Create a mini test set for speed (50 samples instead of 6000)
    # We monkey-patch Config.TEST_METADATA so the engine uses this file
    full_test_df = pd.read_csv(Config.TEST_METADATA)
    mini_test_df = full_test_df.head(50).copy()
    mini_test_path = os.path.join(demo_dir, "test_mini.csv")
    mini_test_df.to_csv(mini_test_path, index=False)

    # Override Config path
    Config.TEST_METADATA = mini_test_path
    print(
        f"Created mini test set at {mini_test_path} with {len(mini_test_df)} samples."
    )

    # --- 2. Verify Data Loaders ---
    print("\n>>> Verifying Data Loaders...")
    # debug=True limits the dataset size significantly (Config.DEBUG_SAMPLE_SIZE)
    train_loader, val_loader = get_train_val_loaders(debug=True)

    # Fetch one batch
    on_imgs, off_imgs, targets = next(iter(train_loader))

    # Verify Shapes
    # Batch size is Config.BATCH_SIZE (64)
    # Dimensions: (Batch, Channels=3, Height=288, Width=256)
    expected_shape = (on_imgs.shape[0], 3, 288, 256)

    print(
        f"Batch shapes: On={on_imgs.shape}, Off={off_imgs.shape}, Target={targets.shape}"
    )

    if on_imgs.shape[1:] != (3, 288, 256):
        raise AssertionError(
            f"Incorrect On-Target image shape. Expected (B, 3, 288, 256), got {on_imgs.shape}"
        )
    if off_imgs.shape[1:] != (3, 288, 256):
        raise AssertionError(
            f"Incorrect Off-Target image shape. Expected (B, 3, 288, 256), got {off_imgs.shape}"
        )
    if len(targets.shape) != 1:
        raise AssertionError(
            f"Incorrect target shape. Expected (B,), got {targets.shape}"
        )

    print("Data Loader verification passed.")

    # --- 3. Verify Model Architecture ---
    print("\n>>> Verifying Model Architecture...")
    device = Config.DEVICE
    model = SiameseDifferenceNet().to(device)

    # Move batch to device
    on_imgs = on_imgs.to(device)
    off_imgs = off_imgs.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(on_imgs, off_imgs)

    print(f"Model output shape: {logits.shape}")

    # Verify output shape (Batch, Num_Classes=1)
    if logits.shape != (on_imgs.shape[0], 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected ({on_imgs.shape[0]}, 1), got {logits.shape}"
        )

    print("Model verification passed.")

    # --- 4. Verify Utils (Mixup) ---
    print("\n>>> Verifying Mixup Utility...")
    # Create dummy data on CPU
    dummy_on = torch.randn(4, 3, 288, 256)
    dummy_off = torch.randn(4, 3, 288, 256)
    dummy_target = torch.tensor([0.0, 1.0, 0.0, 1.0])

    mixed_on, mixed_off, y_a, y_b, lam = mixup_data(
        dummy_on, dummy_off, dummy_target, alpha=0.2, device="cpu"
    )

    if mixed_on.shape != dummy_on.shape:
        raise AssertionError("Mixup changed input tensor shape.")
    if not (0.0 <= lam <= 1.0):
        raise AssertionError("Mixup lambda coefficient out of bounds [0, 1].")

    print(f"Mixup verification passed. Lambda: {lam:.4f}")

    # --- 5. Test Training Engine ---
    print("\n>>> Testing Training Engine (Fast Mode)...")
    # Run for 2 epochs with debug=True to use subset
    engine.run_training(debug=True, epochs=2, patience=1, save_path=model_save_path)

    if not os.path.exists(model_save_path):
        raise AssertionError("Training finished but model checkpoint was not found.")

    print(f"Training demo complete. Model saved to {model_save_path}")

    # --- 6. Test Inference Engine ---
    print("\n>>> Testing Inference Engine...")
    # Predict using the mini test set we configured earlier
    engine.predict(model_path=model_save_path, output_path=submission_save_path)

    if not os.path.exists(submission_save_path):
        raise AssertionError("Inference finished but submission file was not found.")

    # Verify submission content
    sub_df = pd.read_csv(submission_save_path)
    print(f"Submission shape: {sub_df.shape}")

    if len(sub_df) != 50:
        raise AssertionError(
            f"Submission length mismatch. Expected 50 (mini test set), got {len(sub_df)}"
        )

    if list(sub_df.columns) != ["id", "target"]:
        raise AssertionError(
            f"Submission columns mismatch. Expected ['id', 'target'], got {list(sub_df.columns)}"
        )

    print("Inference verification passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
