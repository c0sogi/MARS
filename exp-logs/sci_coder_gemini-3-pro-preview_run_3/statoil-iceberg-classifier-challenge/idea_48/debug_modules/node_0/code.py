import os
import shutil
import torch
import numpy as np
import pandas as pd
import logging

# Import provided library modules
import library.config
import library.utils
import library.data_loader
import library.model
import library.train_eval


def run_demo():
    print("============================================================")
    print("       Iceberg Classification Library Demo Script           ")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration & Monkey Patching for Speed/Isolation
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Define a specific directory for this demo to avoid cache conflicts
    DEMO_WORK_DIR = "./working/demo_usage"
    DEMO_CACHE_DIR = os.path.join(DEMO_WORK_DIR, "cache/")

    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Patch configuration constants to reduce runtime (2 epochs, 1 patience)
    # We must patch them in all modules where they were imported via `from ... import ...`

    # Patch library.config
    library.config.CACHE_DIR = DEMO_CACHE_DIR
    library.config.NUM_EPOCHS = 2
    library.config.PATIENCE = 1
    library.config.BATCH_SIZE = 16
    library.config.NUM_FOLDS = (
        3  # Reduce folds for any loop iterating them (though we run one)
    )

    # Patch library.train_eval (imports constants directly)
    library.train_eval.NUM_EPOCHS = 2
    library.train_eval.PATIENCE = 1

    # Patch library.model (imports constants directly)
    library.model.NUM_EPOCHS = 2
    library.model.PATIENCE = 1
    library.model.BATCH_SIZE = 16
    library.model.NUM_FOLDS = 3

    # Patch library.data_loader
    library.data_loader.BATCH_SIZE = 16
    library.data_loader.NUM_FOLDS = 3

    # Set Reproducibility
    library.utils.set_seed(42)
    print(f"    Cache Directory: {library.config.CACHE_DIR}")
    print(f"    Epochs: {library.config.NUM_EPOCHS}")

    # ------------------------------------------------------------------
    # 2. Data Loading & Processing
    # ------------------------------------------------------------------
    print("\n[2] Testing Data Loading & Processing...")

    # Trigger data processing. load_cached_data=False forces reading from json and saving to new cache
    X_train, angles_train, y_train, X_test, angles_test, test_ids = (
        library.config.load_and_process_data(load_cached_data=False)
    )

    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train shape: {y_train.shape}")
    print(f"    X_test shape : {X_test.shape}")

    # Assertions to verify data integrity
    assert (
        len(X_train) == len(y_train) == len(angles_train)
    ), "Training data length mismatch"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image shape: {X_train.shape[1:]}"
    assert not np.isnan(X_train).any(), "NaN values found in X_train"
    assert not np.isnan(angles_train).any(), "NaN values found in angles_train"

    # ------------------------------------------------------------------
    # 3. Dataset Class Verification
    # ------------------------------------------------------------------
    print("\n[3] Verifying IcebergDataset...")

    dataset = library.config.IcebergDataset(
        X_train[:10], angles_train[:10], y_train[:10]
    )
    img, ang, lbl = dataset[0]

    print(f"    Sample Item - Image: {img.shape}, Angle: {ang}, Label: {lbl}")
    assert img.shape == (3, 75, 75)
    assert isinstance(img, torch.Tensor)
    assert isinstance(lbl, torch.Tensor)

    # ------------------------------------------------------------------
    # 4. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[4] Verifying IDPH_CNN Architecture...")

    model = library.model.IDPH_CNN().to(library.config.DEVICE)
    model.eval()

    # Create dummy batch
    dummy_imgs = torch.randn(4, 3, 75, 75).to(library.config.DEVICE)
    dummy_angs = torch.randn(4).to(library.config.DEVICE)

    with torch.no_grad():
        output = model(dummy_imgs, dummy_angs)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), "Model output should be (Batch_Size, 1)"

    # ------------------------------------------------------------------
    # 5. Training Loop Demonstration (Single Fold)
    # ------------------------------------------------------------------
    print("\n[5] Running Training Loop for Fold 0...")

    # Ensure checkpoints directory exists
    os.makedirs("./checkpoints", exist_ok=True)

    # Run the training for Fold 0 using the utility function
    # This uses the patched NUM_EPOCHS=2
    best_val_loss = library.train_eval.run_fold(fold_idx=0, load_cached_data=True)

    print(f"    Fold 0 Training Complete. Best Val Loss: {best_val_loss:.4f}")

    # Verify checkpoint was saved
    checkpoint_path = "./checkpoints/model_fold_0.pth"
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"    Checkpoint confirmed at: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 6. Inference Demonstration
    # ------------------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load the trained model
    model = library.model.IDPH_CNN().to(library.config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Get Test Loader using library utility
    test_loader, ids = library.data_loader.get_test_loader(
        batch_size=library.config.BATCH_SIZE, load_cached=True
    )

    preds = []
    print("    Processing first 3 batches...")
    with torch.no_grad():
        for i, (images, angles) in enumerate(test_loader):
            if i >= 3:
                break  # Limit for demo speed

            images = images.to(library.config.DEVICE)
            angles = angles.to(library.config.DEVICE)

            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            preds.extend(probs.cpu().numpy().flatten())

    print(f"    Generated {len(preds)} predictions.")
    print(f"    First 5 predictions: {preds[:5]}")

    # Verify predictions are probabilities
    assert all(
        0.0 <= p <= 1.0 for p in preds
    ), "Predictions are not valid probabilities (0-1)"

    print("\n============================================================")
    print("       Demo Completed Successfully                          ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
