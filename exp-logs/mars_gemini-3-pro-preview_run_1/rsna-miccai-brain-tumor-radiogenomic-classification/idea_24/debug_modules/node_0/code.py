import sys
import os
import pandas as pd
import torch
import numpy as np

# Ensure the current directory is in the path for module resolution
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import get_centroids_with_caching
from library.dataset import BraTSDataset, get_transforms
from library.model import CAWIVModel
from library.trainer import run_training
from library.inference import predict_test_set


def main():
    print("=== Starting Demonstration of MGMT Classification Library ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # ------------------------------------------------------------------------
    print("\n[1] Configuring parameters for speed...")
    # Override Config class attributes to ensure the demo runs quickly
    Config.NUM_EPOCHS = 1
    Config.N_FOLDS = 2
    Config.BATCH_SIZE = 4
    # We must use an EfficientNet backbone because CAWIVModel expects 'conv_stem'
    Config.BACKBONE = "efficientnet_b0"
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------------
    # 2. Data Processing Demonstration (Centroids)
    # ------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Processing (Centroids)...")
    # Load a small subset of metadata for testing logic
    df_train_subset = pd.read_csv(Config.TRAIN_METADATA).head(10)

    # Compute centroids for this subset (disabling cache load to force computation)
    centroids = get_centroids_with_caching(
        df_train_subset,
        Config.INPUT_DIR,
        cache_name="demo_centroids",
        load_cached_data=False,
    )

    print("Computed Centroids (Head):")
    print(centroids.head(2))

    # Verification
    assert "BraTS21ID" in centroids.columns
    # Check for modality specific centroid columns
    for mod in Config.MODALITIES:
        assert f"{mod}_centroid" in centroids.columns
    assert len(centroids) == 10
    print("-> Centroid calculation verified.")

    # ------------------------------------------------------------------------
    # 3. Dataset Demonstration
    # ------------------------------------------------------------------------
    print("\n[3] Demonstrating Dataset and Augmentations...")
    transforms = get_transforms(mode="train")
    dataset = BraTSDataset(
        df_train_subset, centroids, Config.INPUT_DIR, transform=transforms, mode="train"
    )

    # Fetch one sample to verify tensor construction
    img_tensor, target = dataset[0]

    print(f"Image Tensor Shape: {img_tensor.shape}")
    print(f"Target Value: {target}")

    # Verification
    # Expected shape: (9, 224, 224) corresponding to 3 modalities * 3 depths
    expected_shape = (9, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        img_tensor.shape == expected_shape
    ), f"Expected {expected_shape}, got {img_tensor.shape}"
    assert isinstance(target, torch.Tensor)
    print("-> Dataset shape and type verified.")

    # ------------------------------------------------------------------------
    # 4. Model Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating CA-WIV Model...")
    # Instantiate model (pretrained=False for speed/offline safety in demo)
    model = CAWIVModel(model_name=Config.BACKBONE, pretrained=False)
    model.to(device)
    model.eval()

    # Create a dummy batch matching the dataset output
    batch_imgs = img_tensor.unsqueeze(0).to(device)  # Shape: (1, 9, 224, 224)

    with torch.no_grad():
        output = model(batch_imgs)

    print(f"Model Output Shape: {output.shape}")

    # Verification
    # Binary classification logits -> (Batch_Size, 1)
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    print("-> Model forward pass verified.")

    # ------------------------------------------------------------------------
    # 5. Training Pipeline Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Running Training Pipeline (Debug Mode)...")
    # run_training with debug=True uses a small subset of data and runs quickly.
    # It will train for Config.NUM_EPOCHS (1) on Config.N_FOLDS (2).
    run_training(load_cached_data=False, debug=True, patience=1)

    # Verify that model checkpoints were created
    fold0_path = os.path.join(Config.CACHE_DIR, "best_model_fold0.pth")
    assert os.path.exists(
        fold0_path
    ), f"Fold 0 model checkpoint not found at {fold0_path}!"
    print("-> Training pipeline completed and checkpoint verified.")

    # ------------------------------------------------------------------------
    # 6. Inference Pipeline Demonstration
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference Pipeline...")
    # This generates predictions on the test set using the models trained above.
    # Note: In debug mode, run_training might not save a model if validation fails completely,
    # but with random initialization it usually saves at least epoch 0.
    predict_test_set(load_cached_data=False)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    assert "BraTS21ID" in df_sub.columns
    assert "MGMT_value" in df_sub.columns
    assert len(df_sub) > 0
    print("-> Inference pipeline completed and submission verified.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
