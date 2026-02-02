import os
import torch
import pandas as pd
import sys

# Import from the provided library
from library.config import Config
from library.dataset import get_loaders
from library.model import get_model
from library.trainer import Trainer
from library.inference import generate_submission


def main():
    print("Starting Animal Classification Library Demo...")

    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    # We modify the Config class directly to ensure the demo runs quickly.
    # These changes propagate to other modules because they reference the same Config class.
    print("\n[1] Configuring environment for rapid demonstration...")

    Config.set_seed(42)

    # Override defaults for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.IMG_SIZE = (128, 128)  # Reduce image size for faster processing

    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Image Size: {Config.IMG_SIZE}")
    print(f"   Device: {Config.DEVICE}")

    # ==========================================
    # 2. Verify Data Loaders
    # ==========================================
    print("\n[2] Verifying Data Loaders...")

    # Initialize loaders in debug mode (uses a small subset of data)
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch from the training loader
    images, labels = next(iter(train_loader))

    # Assertions to verify data integrity
    print(f"   Train Batch Image Shape: {images.shape}")
    print(f"   Train Batch Label Shape: {labels.shape}")

    expected_image_shape = (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    )
    assert (
        images.shape == expected_image_shape
    ), f"Image shape mismatch. Expected {expected_image_shape}, got {images.shape}"

    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    assert labels.dtype == torch.long, "Labels must be of type torch.long"

    print("   Data Loader verification passed.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    # Initialize model
    model = get_model(device=Config.DEVICE)

    # Move the sample batch to the correct device
    images = images.to(Config.DEVICE)

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(images)

    print(f"   Model Output Shape: {outputs.shape}")

    # Assertions to verify model output
    expected_output_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        outputs.shape == expected_output_shape
    ), f"Model output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"

    print("   Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n[4] Running Training Loop (Trainer.fit)...")

    # Initialize Trainer with debug=True (small dataset) and modified epochs
    trainer = Trainer(debug=True, epochs=Config.EPOCHS)

    # Execute training
    trainer.fit()

    # Verify that the model checkpoint was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"   Success: Model saved to {Config.MODEL_SAVE_PATH}")
    else:
        # Note: If validation F1 doesn't improve (unlikely in 1st epoch starting from -1),
        # the code might not save. However, Trainer init sets best_val_f1 = -1.0.
        # Any valid F1 (>=0) will trigger a save on the first epoch.
        raise FileNotFoundError(
            f"Model file was not created at {Config.MODEL_SAVE_PATH}"
        )

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\n[5] Running Inference (generate_submission)...")

    # Run inference using the saved model
    # We use debug=True to predict on a small subset of the test data
    generate_submission(
        weights_path=Config.MODEL_SAVE_PATH, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Verify submission file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(submission_path):
        print(f"   Success: Submission file created at {submission_path}")

        # Validate content format
        df_sub = pd.read_csv(submission_path)
        print(f"   Submission Rows: {len(df_sub)}")
        print(f"   Submission Columns: {df_sub.columns.tolist()}")

        assert (
            "Id" in df_sub.columns and "Predicted" in df_sub.columns
        ), "Submission file missing required columns."
        assert len(df_sub) > 0, "Submission file is empty."

    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
