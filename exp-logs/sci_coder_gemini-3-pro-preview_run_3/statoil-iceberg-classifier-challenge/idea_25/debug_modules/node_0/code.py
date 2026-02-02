import sys
import os
import torch
import numpy as np

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def main():
    print("=== Iceberg Classifier Library Demo ===")

    # 1. Setup and Configuration
    # We set the seed for reproducibility
    utils.set_seed(42)

    # Override configuration for the purpose of a quick demo
    # The default is 75 epochs, which is too long for a demo run.
    print(f"Default NUM_EPOCHS: {config.NUM_EPOCHS}")
    config.NUM_EPOCHS = 2
    print(f"Overridden NUM_EPOCHS: {config.NUM_EPOCHS}")

    # Ensure working directory exists (though config.py does this, we double check)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 2. Data Processing
    print("\n--- Data Processing ---")
    # This function loads the JSON data, processes images (bands -> 3 channels),
    # imputes angles, and saves .npy files to config.WORKING_DIR
    data = data_loader.process_data(load_cached_data=True)

    # Verify data shapes
    X_train = data["X_train"]
    y_train = data["y_train"]
    angles_train = data["angles_train"]

    print(f"Processed X_train shape: {X_train.shape}")
    print(f"Processed y_train shape: {y_train.shape}")
    print(f"Processed angles_train shape: {angles_train.shape}")

    assert X_train.ndim == 4 and X_train.shape[1] == 3, "X_train should be (N, 3, H, W)"
    assert y_train.ndim == 1, "y_train should be (N,)"
    assert len(X_train) == len(y_train) == len(angles_train), "Data lengths mismatch"

    # 3. Data Loading (Fold 0)
    print("\n--- Data Loading (Fold 0) ---")
    train_loader, val_loader = data_loader.get_loaders(fold=0, load_cached_data=True)

    # Fetch a single batch to verify loader logic
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Angles Shape: {angles.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions for batch structure
    assert images.shape == (config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (config.BATCH_SIZE,)
    assert labels.shape == (config.BATCH_SIZE,)

    # 4. Model Instantiation and Forward Pass
    print("\n--- Model Initialization & Forward Pass ---")
    model = model_lib.MAPCNN()
    model = model.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)
    angles = angles.to(config.DEVICE)

    # Run forward pass
    logits = model(images, angles)
    print(f"Output Logits Shape: {logits.shape}")

    # Assert output shape matches batch size (BCEWithLogitsLoss expects flattened output)
    assert logits.shape == (config.BATCH_SIZE,)

    # 5. Training Loop Execution
    print("\n--- Executing Training Loop (Fold 0) ---")
    # train_fold handles the loop, validation, early stopping, and saving
    best_val_loss = train_lib.train_fold(
        fold_idx=0, train_loader=train_loader, val_loader=val_loader
    )

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_fold_0.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint successfully created at: {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"Expected checkpoint at {checkpoint_path} but not found."
        )

    # 6. Inference / Test Prediction
    print("\n--- Inference Demonstration ---")
    test_loader = data_loader.get_test_loader(load_cached_data=True)

    # Load the trained model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    # Run inference on one batch
    with torch.no_grad():
        test_imgs, test_angles, test_ids = next(iter(test_loader))
        test_imgs = test_imgs.to(config.DEVICE)
        test_angles = test_angles.to(config.DEVICE)

        test_logits = model(test_imgs, test_angles)
        test_probs = torch.sigmoid(test_logits)

    print(f"Test Batch Predictions (First 5): {test_probs[:5].cpu().numpy()}")

    # Assertions for predictions
    assert (
        test_probs.min() >= 0.0 and test_probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"
    assert len(test_probs) == config.BATCH_SIZE or len(test_probs) == len(test_ids)

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
