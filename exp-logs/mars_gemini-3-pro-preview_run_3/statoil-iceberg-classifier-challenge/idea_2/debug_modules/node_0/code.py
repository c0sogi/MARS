import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import model
from library import train


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Setup and Seeding
    print("--- Initializing Environment ---")
    utils.seed_everything(config.SEED)
    print(f"Device: {config.DEVICE}")
    print(f"Working Directory: {config.WORKING_DIR}")

    # 2. Data Loading and Verification
    print("\n--- Loading Data ---")
    # Load data (will use cache in ./working/idea_2 if available)
    X_train, y_train, angles_train, X_test, ids_test, angles_test = (
        data_loader.load_and_process_data(load_cached_data=True)
    )

    # Verify Data Shapes
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Assertions to ensure data integrity
    assert (
        len(X_train) == len(y_train) == len(angles_train)
    ), "Train data length mismatch"
    assert len(X_test) == len(ids_test) == len(angles_test), "Test data length mismatch"

    # Verify Image Dimensions (N, 224, 224, 3)
    assert X_train.shape[1:] == (
        224,
        224,
        3,
    ), f"Unexpected train image shape: {X_train.shape}"
    assert X_test.shape[1:] == (
        224,
        224,
        3,
    ), f"Unexpected test image shape: {X_test.shape}"

    # Verify Normalization (should be roughly [0, 1])
    assert (
        X_train.min() >= -0.1 and X_train.max() <= 1.1
    ), "Data does not appear normalized"
    print("Data loaded and verified successfully.")

    # 3. DataLoader and Model Verification
    print("\n--- Verifying Model and DataLoader ---")
    # Create a loader for Fold 0
    train_loader, val_loader = data_loader.get_fold_loaders(
        fold_idx=0, X=X_train, y=y_train, angles=angles_train, batch_size=8
    )

    # Fetch a single batch
    images, angles, labels = next(iter(train_loader))

    # Verify Tensor Shapes
    # PyTorch expects (Batch, Channel, Height, Width)
    assert images.shape == (
        8,
        3,
        224,
        224,
    ), f"Incorrect batch image shape: {images.shape}"
    assert angles.shape == (8,), f"Incorrect batch angle shape: {angles.shape}"
    assert labels.shape == (8,), f"Incorrect batch label shape: {labels.shape}"

    # Instantiate Model
    net = model.IcebergVGG16(dropout_rate=0.5)
    net.to(config.DEVICE)
    net.eval()

    # Run Forward Pass
    images = images.to(config.DEVICE)
    angles = angles.to(config.DEVICE)

    with torch.no_grad():
        outputs = net(images, angles)

    # Verify Output
    assert outputs.shape == (8, 1), f"Model output shape mismatch: {outputs.shape}"
    assert (
        outputs.min() >= 0.0 and outputs.max() <= 1.0
    ), "Model output not in [0, 1] range (Sigmoid check)"
    print("Model architecture and forward pass verified.")

    # 4. Training Demonstration (1 Epoch, Fold 0)
    print("\n--- Running Training Demonstration (Fold 0, 1 Epoch) ---")

    # We use the library function run_fold but override parameters for speed
    best_val_loss = train.run_fold(
        fold_idx=0,
        X_train=X_train,
        y_train=y_train,
        angles_train=angles_train,
        num_epochs=1,  # Reduced from config.NUM_EPOCHS (30)
        batch_size=16,
        learning_rate=1e-4,
        patience=1,
        device=config.DEVICE,
    )

    print(f"Training demo completed. Best Val Loss: {best_val_loss:.6f}")

    # Verify Checkpoint
    checkpoint_dir = os.path.join(config.WORKING_DIR, "fold_0")
    checkpoint_path = os.path.join(checkpoint_dir, "model_best.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Checkpoint file verified.")

    # 5. Inference Demonstration
    print("\n--- Running Inference Demonstration ---")

    # Load the best model from the training demo
    best_checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    net.load_state_dict(best_checkpoint["state_dict"])
    net.eval()

    # Use a small subset of test data for quick verification
    subset_size = 32
    X_test_sub = X_test[:subset_size]
    angles_test_sub = angles_test[:subset_size]
    ids_test_sub = ids_test[:subset_size]

    # Create test loader
    test_loader = data_loader.get_test_loader(
        X_test_sub, angles_test_sub, batch_size=16
    )

    # Predict
    preds = model.predict(test_loader, net, config.DEVICE)

    # Verify Predictions
    assert len(preds) == subset_size, "Prediction count mismatch"
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions out of probability range"
    print("Inference successful.")

    # 6. Submission File Generation
    print("\n--- Generating Demo Submission File ---")

    # Create dataframe
    df_sub = pd.DataFrame({"id": ids_test_sub, "is_iceberg": preds})

    # Save to submission directory
    demo_submission_path = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")
    df_sub.to_csv(demo_submission_path, index=False)

    assert os.path.exists(demo_submission_path), "Submission file was not created"

    # Check format
    df_check = pd.read_csv(demo_submission_path)
    assert list(df_check.columns) == ["id", "is_iceberg"], "Submission columns mismatch"
    assert len(df_check) == subset_size, "Submission row count mismatch"

    print(f"Demo submission saved to: {demo_submission_path}")
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
