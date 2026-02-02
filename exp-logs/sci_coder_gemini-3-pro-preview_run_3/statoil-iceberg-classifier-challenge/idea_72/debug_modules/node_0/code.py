import torch
import numpy as np
import sys
import os

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

from library import config, utils, data, model as model_lib, train


def main():
    print("=== Iceberg Classification Task Demonstration ===\n")

    # 1. Setup Environment
    print(">>> Step 1: Setting up environment and device...")
    utils.set_seed(42)
    device = utils.get_device()
    print(f"Using device: {device}")

    # 2. Data Loading Verification
    print("\n>>> Step 2: Loading and verifying data...")
    # load_data handles caching logic internally.
    # It returns ((X_train, angle_train, y_train, ids_train), (X_test, angle_test, ids_test))
    train_set, test_set = data.load_data(load_cached_data=True)

    X_train, angle_train, y_train, ids_train = train_set
    X_test, angle_test, ids_test = test_set

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Training Angles Shape: {angle_train.shape}")
    print(f"Training Labels Shape: {y_train.shape}")

    # Assertions to ensure data integrity
    # Images should be (N, 3, 75, 75) -> 3 channels are HH, HV, Avg
    assert X_train.ndim == 4
    assert X_train.shape[1] == 3
    assert X_train.shape[2] == 75 and X_train.shape[3] == 75
    assert len(X_train) == len(y_train) == len(angle_train)
    print("Data shapes verified successfully.")

    # 3. Model Architecture Verification
    print("\n>>> Step 3: Verifying Model Architecture...")
    net = model_lib.TSICNN().to(device)

    # Create dummy inputs: Batch of 4 images, 3 channels, 75x75
    dummy_batch_size = 4
    dummy_images = torch.randn(dummy_batch_size, 3, 75, 75).to(device)
    # Angles are 1D tensor of shape (Batch,)
    dummy_angles = torch.randn(dummy_batch_size).to(device)

    # Perform forward pass
    net.eval()
    with torch.no_grad():
        outputs = net(dummy_images, dummy_angles)

    print(f"Model Output Shape: {outputs.shape}")

    # The model output should be a 1D tensor of logits with size (Batch_Size,)
    assert outputs.shape == (dummy_batch_size,)
    print("Model forward pass verified successfully.")

    # 4. Training Loop Demonstration (Fast Mode)
    print("\n>>> Step 4: Demonstrating Training Loop (Debug Mode)...")
    # We use run_fold with debug=True to limit the number of batches and epochs=1 for speed
    # This verifies the training pipeline without waiting for full convergence
    try:
        best_val_loss = train.run_fold(
            fold_index=0,
            total_folds=5,
            epochs=1,  # Run only 1 epoch
            batch_size=8,  # Small batch size for speed
            debug=True,  # Debug mode runs only a few batches per epoch
        )
        print(
            f"Training simulation complete. Best Validation Loss: {best_val_loss:.4f}"
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # 5. Inference Demonstration
    print("\n>>> Step 5: Demonstrating Inference...")
    # Get test loader
    test_loader, test_ids_loader = data.get_test_loader(batch_size=4, num_workers=0)

    net.eval()
    sample_preds = []

    # Run inference on a single batch
    with torch.no_grad():
        for i, (images, angles) in enumerate(test_loader):
            images = images.to(device)
            angles = angles.to(device)

            logits = net(images, angles)
            probs = torch.sigmoid(logits)

            sample_preds.extend(probs.cpu().numpy())

            # Stop after first batch for demonstration
            break

    print(f"Generated {len(sample_preds)} predictions from the first batch.")
    print(f"Sample predictions: {sample_preds}")

    # Verify predictions are probabilities
    assert all(0.0 <= p <= 1.0 for p in sample_preds)
    print("Inference verified successfully.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
