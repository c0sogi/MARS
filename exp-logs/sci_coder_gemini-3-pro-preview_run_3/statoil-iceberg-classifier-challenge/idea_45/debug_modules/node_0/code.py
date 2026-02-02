import os
import shutil
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.utils import seed_everything
from library.dataset import process_data, get_loaders, get_test_loader
from library.model import BDPH_CNN
from library.engine import train_fold, generate_submission

if __name__ == "__main__":
    print("=== Starting Demo Script ===")

    # 1. Configure for Demo (Speed and Isolation)
    print("\n[1] Setting up configuration...")
    # Override Config to run a minimal version
    Config.WORK_DIR = "./working/demo_usage"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.EPOCHS = 1  # Run only 1 epoch
    Config.N_FOLDS = 1  # Run only 1 fold
    Config.BATCH_SIZE = 16  # Small batch size

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)

    Config.setup()
    seed_everything(Config.SEED)

    # 2. Data Processing and Loading
    print("\n[2] Verifying Data Processing...")
    # Force processing from raw json to verify logic
    (X_train, angle_train, y_train), (X_test, angle_test, ids_test) = process_data(
        load_cached_data=False
    )

    # Assertions on raw processed data
    print("   Verifying data shapes...")
    # Images should be (N, 3, 75, 75)
    assert len(X_train.shape) == 4
    assert X_train.shape[1:] == (3, 75, 75)
    # Angles should be (N,)
    assert len(angle_train.shape) == 1
    assert X_train.shape[0] == angle_train.shape[0]
    # Labels should be (N,)
    assert X_train.shape[0] == y_train.shape[0]

    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Get DataLoaders for Fold 0
    print("\n[3] Verifying DataLoaders...")
    train_loader, val_loader = get_loaders(fold=0, batch_size=Config.BATCH_SIZE)

    # Fetch one batch to verify structure
    images, angles, labels = next(iter(train_loader))

    print(f"   Batch Image Shape: {images.shape}")
    print(f"   Batch Angle Shape: {angles.shape}")
    print(f"   Batch Label Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)
    assert images.dtype == torch.float32

    # 3. Model Instantiation and Forward Pass
    print("\n[4] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = BDPH_CNN().to(device)

    # Move batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    outputs = model(images_dev, angles_dev)

    print(f"   Model Output Shape: {outputs.shape}")

    # Assert output shape (Batch_Size, 1)
    assert outputs.shape == (Config.BATCH_SIZE, 1)
    # Assert outputs are not NaN
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    # 4. Training Loop (1 Epoch, 1 Fold)
    print("\n[5] Running Training Loop (Fold 0, 1 Epoch)...")
    # This calls engine.train_fold which runs the loop
    best_score = train_fold(0, train_loader, val_loader)

    print(f"   Training complete. Best Validation Score: {best_score:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("   Checkpoint verified.")

    # 5. Inference and Submission
    print("\n[6] Generating Submission...")
    test_loader = get_test_loader(batch_size=Config.BATCH_SIZE)

    # Generate submission (uses the saved checkpoint from step 4)
    generate_submission(test_loader)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"   Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Check format
    assert list(df_sub.columns) == ["id", "is_iceberg"]
    assert len(df_sub) == len(ids_test)
    assert df_sub["is_iceberg"].min() >= 0.0
    assert df_sub["is_iceberg"].max() <= 1.0

    print("\n=== Demo Completed Successfully ===")
