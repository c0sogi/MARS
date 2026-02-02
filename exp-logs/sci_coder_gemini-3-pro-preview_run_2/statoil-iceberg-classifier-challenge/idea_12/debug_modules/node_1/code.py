import os
import sys
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the path so library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import get_loaders
from library.model import GLPPN
from library.train_eval import run_fold


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Override
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("Configuring for fast demonstration...")
    Config.MAX_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.PATIENCE = 2  # Short patience for early stopping
    Config.SCHEDULER_PATIENCE = 1  # Short patience for scheduler
    Config.N_FOLDS = 1  # Only run 1 fold

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Ensure directories exist
    Config.setup()
    print(f"Running on device: {Config.DEVICE}")

    # 2. Data Loading Demonstration
    print("\n[Data Loading]")
    # get_loaders handles loading raw data, processing (normalization/reshaping), and creating DataLoaders
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Inspect a training batch
    images, angles, labels = next(iter(train_loader))
    print(
        f"Train Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions to verify data integrity
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Train image batch shape mismatch"
    assert angles.shape == (Config.BATCH_SIZE, 1), "Train angle batch shape mismatch"
    assert labels.shape == (Config.BATCH_SIZE, 1), "Train label batch shape mismatch"
    assert images.dtype == torch.float32, "Images should be float32"
    # Check normalization: values should be roughly within [0, 1] due to min-max scaling
    assert images.max() <= 1.0 + 1e-5, "Image max value exceeds 1.0"
    assert images.min() >= 0.0 - 1e-5, "Image min value is below 0.0"

    # Inspect a test batch
    test_images, test_angles, test_ids = next(iter(test_loader))
    print(
        f"Test Batch Shapes  -> Images: {test_images.shape}, IDs count: {len(test_ids)}"
    )
    assert len(test_ids) == Config.BATCH_SIZE, "Test ID batch size mismatch"

    # 3. Model Architecture Demonstration
    print("\n[Model Architecture]")
    model = GLPPN().to(Config.DEVICE)

    # Run a dummy forward pass to verify architecture
    dummy_img = images.to(Config.DEVICE)
    dummy_ang = angles.to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    print(f"Forward Pass Output Shape: {output.shape}")
    # Output should be (Batch_Size, 1) logits
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n[Training Loop]")
    # run_fold handles the full training lifecycle: training, validation, scheduler, early stopping, and saving
    # It returns the best model state and the best validation metric
    trained_model, best_val_loss = run_fold(
        fold_index=0, train_loader=train_loader, val_loader=val_loader
    )

    print(f"Training finished. Best Validation Log Loss: {best_val_loss:.6f}")
    assert isinstance(best_val_loss, float), "Metric should be a float"
    assert best_val_loss > 0, "Log loss should be positive"

    # 5. Inference and Submission
    print("\n[Inference & Submission]")
    trained_model.eval()
    device = torch.device(Config.DEVICE)

    all_probs = []
    all_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch_imgs, batch_angs, batch_ids in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_angs = batch_angs.to(device)

            # Forward pass
            logits = trained_model(batch_imgs, batch_angs)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_ids.extend(batch_ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": all_ids, "is_iceberg": all_probs})

    print(f"Submission DataFrame Head:\n{submission_df.head()}")

    # Verify submission format
    assert len(submission_df) == len(
        pd.read_csv(Config.TEST_META)
    ), "Submission row count mismatch"
    assert list(submission_df.columns) == [
        "id",
        "is_iceberg",
    ], "Submission columns mismatch"
    assert submission_df["is_iceberg"].min() >= 0.0, "Probabilities must be >= 0"
    assert submission_df["is_iceberg"].max() <= 1.0, "Probabilities must be <= 1"

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    # 6. Utility Verification
    print("\n[Utility Verification]")
    # Verify log loss calculation manually
    y_true_dummy = np.array([0, 1, 1, 0])
    y_pred_dummy = np.array([0.1, 0.9, 0.8, 0.2])
    calculated_loss = calculate_log_loss(y_true_dummy, y_pred_dummy)
    print(f"Manual Log Loss Check: {calculated_loss:.4f}")
    assert (
        calculated_loss < 0.3
    ), "Log loss calculation seems incorrect for good predictions"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
