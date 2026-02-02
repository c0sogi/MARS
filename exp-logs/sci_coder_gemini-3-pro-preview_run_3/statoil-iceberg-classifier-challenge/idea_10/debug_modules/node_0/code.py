import os
import torch
import numpy as np
from library.config import Config, set_seed
from library.data_loader import get_loaders
from library.model import MicroResNet
from library.utils import save_checkpoint, load_checkpoint
from library.train_eval import train_fold


def main():
    # 1. Setup Configuration
    # Enable debug mode to reduce epochs to 2 and Num Folds to 2 for quick execution
    print("Initializing Configuration in Debug Mode...")
    Config.setup(debug=True)
    set_seed(Config.SEED)
    Config.print_summary()

    # 2. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")
    # This will trigger load_and_process_data, creating cache if needed
    train_loader, val_loader, test_loader = get_loaders(fold=0, load_cached_data=True)

    # Fetch one batch to verify shapes
    try:
        images, angles, labels = next(iter(train_loader))
        print(
            f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
        )

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            75,
            75,
        ), f"Expected image shape ({Config.BATCH_SIZE}, 3, 75, 75), got {images.shape}"
        assert angles.shape == (
            Config.BATCH_SIZE,
        ), f"Expected angle shape ({Config.BATCH_SIZE},), got {angles.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected label shape ({Config.BATCH_SIZE},), got {labels.shape}"
        print("Data Loader shapes verified successfully.")
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    model = MicroResNet().to(Config.DEVICE)
    print(f"Model moved to device: {Config.DEVICE}")

    # Create dummy inputs matching the batch structure
    dummy_images = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(Config.DEVICE)
    dummy_angles = torch.randn(Config.BATCH_SIZE).to(Config.DEVICE)

    # Forward pass
    output = model(dummy_images, dummy_angles)
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"
    print("Model forward pass verified successfully.")

    # 4. Checkpoint Utility Verification
    print("\n--- Verifying Checkpoint Utilities ---")
    dummy_fold = 999
    dummy_state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": {},  # Dummy empty dict
        "best_log_loss": 0.12345,
        "fold": dummy_fold,
    }

    # Save checkpoint
    save_checkpoint(dummy_state, is_best=True, fold=dummy_fold)

    expected_ckpt_path = os.path.join(
        Config.CHECKPOINT_DIR, f"checkpoint_fold_{dummy_fold}.pth"
    )
    expected_best_path = os.path.join(
        Config.CHECKPOINT_DIR, f"model_best_fold_{dummy_fold}.pth"
    )

    # Verify file creation
    assert os.path.exists(expected_ckpt_path), "Checkpoint file was not created."
    assert os.path.exists(expected_best_path), "Best model file was not created."

    # Load checkpoint
    loaded_state = load_checkpoint(model, filename=expected_ckpt_path)
    assert loaded_state is not None, "Failed to load checkpoint."
    assert loaded_state["fold"] == dummy_fold, "Loaded checkpoint content mismatch."
    print("Checkpoint save/load verified successfully.")

    # 5. Training Loop Integration (Fold 0)
    print("\n--- Verifying Training Loop (Fold 0) ---")
    # train_fold handles model instantiation, training, validation, and metric calculation
    # Since debug=True, this runs for limited epochs (Config.EPOCHS=2)
    best_loss = train_fold(fold=0, train_loader=train_loader, val_loader=val_loader)

    print(f"Training Loop Completed. Best Validation Log Loss: {best_loss}")

    # Assertions
    assert isinstance(best_loss, float), "Returned loss is not a float."
    assert best_loss > 0, "Log loss should be positive."

    print("\nAll demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
