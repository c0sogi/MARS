import torch
import numpy as np
import pandas as pd
import os
import sys

# Import from the provided library files
from library.config import Config, seed_everything
from library.utils import AverageMeter, apk, mapk, IdEncoder, get_id_encoder
from library.data import WhaleDataset, get_transforms, get_loaders
from library.model import GeM, WhaleEfficientNet
from library.train import run_training

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Environment Setup
    # ---------------------------------------------------------
    print("Initializing demonstration...")
    seed_everything(42)

    # Modify Config for the demonstration to ensure speed and low resource usage
    print("Configuring environment for demo...")
    Config.BATCH_SIZE = 4  # Small batch size for testing
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in short run
    Config.IMG_SIZE = 224  # Reduce image size for faster processing in demo

    # ---------------------------------------------------------
    # 2. Utility Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Utilities ---")

    # Test Average Precision at K (APK)
    # Scenario: Target is 1. Prediction has 1 at rank 5. Score should be 1/5 = 0.2.
    score = apk([1], [2, 3, 4, 5, 1], k=5)
    assert np.isclose(
        score, 0.2
    ), f"APK calculation incorrect. Expected 0.2, got {score}"

    # Scenario: Target is 1. Prediction has 1 at rank 1. Score should be 1/1 = 1.0.
    score_perfect = apk([1], [1, 2, 3, 4, 5], k=5)
    assert np.isclose(
        score_perfect, 1.0
    ), f"APK calculation incorrect. Expected 1.0, got {score_perfect}"
    print("Metric (APK) logic verified.")

    # Test IdEncoder
    dummy_classes = ["whale_a", "whale_b", "new_whale"]
    encoder = IdEncoder(dummy_classes)

    # Test Transform
    idx = encoder.transform("whale_b")
    assert idx == 1, f"Encoder transform failed. Expected 1, got {idx}"

    # Test Inverse Transform
    label = encoder.inverse_transform(1)
    assert (
        label == "whale_b"
    ), f"Encoder inverse transform failed. Expected 'whale_b', got {label}"
    print("IdEncoder logic verified.")

    # ---------------------------------------------------------
    # 3. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")

    # Ensure metadata exists
    assert os.path.exists(Config.TRAIN_CSV), "Training metadata file missing."

    # Load a small sample of metadata manually for unit testing the Dataset class
    df_train_sample = pd.read_csv(Config.TRAIN_CSV).head(10)

    # Create Encoder (force creation to ensure logic works)
    real_encoder = get_id_encoder(load_cached_data=False)

    # Instantiate Dataset
    dataset = WhaleDataset(
        df_train_sample, transforms=get_transforms("train"), id_encoder=real_encoder
    )

    # Check single item retrieval
    img, lbl = dataset[0]

    # Verify Image Shape: (Channels, Height, Width)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {img.shape}"

    # Verify Label Type
    assert isinstance(lbl, torch.Tensor), "Label should be a torch.Tensor"
    print("WhaleDataset instantiation and retrieval verified.")

    # Verify DataLoaders
    # Using debug=True to load only a small subset defined in Config.DEBUG_SAMPLE_SIZE
    train_loader, val_loader = get_loaders(debug=True, load_cached_data=True)

    # Fetch one batch
    batch_imgs, batch_lbls = next(iter(train_loader))

    assert (
        batch_imgs.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {batch_imgs.shape[0]}"
    assert batch_lbls.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch."
    print("DataLoaders initialized and batch generation verified.")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    # Test Generalized Mean Pooling (GeM)
    gem_layer = GeM(p=3.0)
    # Create dummy feature map: (Batch=2, Channels=64, H=10, W=10)
    dummy_features = torch.randn(2, 64, 10, 10)
    pooled_features = gem_layer(dummy_features)

    # Expected output: (Batch=2, Channels=64, 1, 1)
    assert pooled_features.shape == (
        2,
        64,
        1,
        1,
    ), f"GeM output shape mismatch. Got {pooled_features.shape}"
    print("GeM Pooling layer verified.")

    # Test Full Model
    # Initialize without pretrained weights for speed
    model = WhaleEfficientNet(pretrained=False)
    model.eval()

    # Forward pass with the batch retrieved earlier
    with torch.no_grad():
        logits = model(batch_imgs)

    # Expected output: (Batch, NumClasses)
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"
    print("WhaleEfficientNet forward pass verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n--- Running Training Loop (Integration Test) ---")
    print("Running 1 epoch on debug subset...")

    # Run the training pipeline
    # debug=True limits the dataset size
    # epochs=1 limits the duration
    try:
        run_training(debug=True, epochs=1, patience=1)
        print("Training loop executed successfully.")
    except Exception as e:
        print(f"Training loop failed with error: {e}")
        raise e

    print("\nAll demonstrations completed successfully.")
