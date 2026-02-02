import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config, seed_everything
from library.utils import unscale_data, laplace_log_likelihood_metric
from library.image_processing import process_patient
from library.dataset import LungFVCDataset
from library.model import DualPathTransformer
from library.trainer import Trainer


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup and Configuration Override for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(42)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10  # Use very small subset
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"Config configured: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}")

    # 2. Verify Utils
    print("\n[2] Verifying Utility Functions...")

    # Test unscale_data
    # Create dummy standardized data
    # Mean=2654.6528, Std=801.7017
    fvc_std = torch.tensor([0.0, 1.0])  # Should become Mean, Mean+Std
    sigma_std = torch.tensor([1.0, 0.5])  # Should become Std, 0.5*Std

    fvc_raw, sigma_raw = unscale_data(fvc_std, sigma_std)

    expected_fvc_0 = Config.TARGET_MEAN
    expected_fvc_1 = Config.TARGET_MEAN + Config.TARGET_STD

    assert torch.isclose(
        fvc_raw[0], torch.tensor(expected_fvc_0), atol=1e-3
    ), f"Unscale FVC mismatch: {fvc_raw[0]} vs {expected_fvc_0}"
    assert torch.isclose(
        sigma_raw[0], torch.tensor(Config.TARGET_STD), atol=1e-3
    ), f"Unscale Sigma mismatch: {sigma_raw[0]} vs {Config.TARGET_STD}"
    print("  -> unscale_data logic verified.")

    # Test laplace_log_likelihood_metric
    # Case: Perfect prediction with sigma=clipped value (70)
    # Metric = - (sqrt(2) * 0 / 70) - ln(sqrt(2) * 70) = -ln(sqrt(2)*70)
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma = np.array([70.0])

    score = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 70)

    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric calculation mismatch: {score} vs {expected_score}"
    print("  -> laplace_log_likelihood_metric logic verified.")

    # 3. Verify Image Processing
    print("\n[3] Verifying Image Processing...")
    # Load metadata to get a real patient ID
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    sample_patient = train_df.iloc[0]["Patient"]
    sample_path = train_df.iloc[0]["image_path"]

    print(f"  Processing patient: {sample_patient}")
    img_data = process_patient(sample_patient, sample_path, load_cached_data=False)

    # Check shape: (3, IMG_SIZE, IMG_SIZE)
    expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        img_data.shape == expected_shape
    ), f"Image data shape mismatch. Expected {expected_shape}, got {img_data.shape}"
    assert (
        img_data.dtype == np.float32
    ), f"Image data type mismatch. Expected float32, got {img_data.dtype}"

    # Check value range (should be normalized roughly 0-1, though padding is 0)
    print(f"  Image stats: Min={img_data.min():.4f}, Max={img_data.max():.4f}")
    print("  -> process_patient execution verified.")

    # 4. Verify Dataset
    print("\n[4] Verifying Dataset...")
    dataset = LungFVCDataset(train_df, mode="train", debug=True)
    print(f"  Dataset length (Debug): {len(dataset)}")

    sample = dataset[0]
    required_keys = [
        "images",
        "meta_age",
        "meta_sex",
        "meta_smoke",
        "linear_features",
        "target",
    ]
    for k in required_keys:
        assert k in sample, f"Missing key in dataset sample: {k}"

    # Check tensor shapes
    assert sample["images"].shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Dataset image shape incorrect"
    assert sample["linear_features"].shape == (
        2,
    ), "Dataset linear features shape incorrect"
    assert isinstance(sample["target"], torch.Tensor), "Target is not a tensor"

    print("  -> LungFVCDataset structure verified.")

    # 5. Verify Model Architecture
    print("\n[5] Verifying Model Architecture...")
    model = DualPathTransformer()
    model.eval()

    # Create dummy batch
    B = 2
    dummy_images = torch.randn(B, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    dummy_age = torch.randn(B)
    dummy_sex = torch.randint(0, 2, (B,))
    dummy_smoke = torch.randint(0, 3, (B,))
    dummy_linear = torch.randn(B, 2)

    with torch.no_grad():
        fvc_pred, sigma_pred = model(
            dummy_images, dummy_age, dummy_sex, dummy_smoke, dummy_linear
        )

    assert fvc_pred.shape == (B,), f"Model FVC output shape mismatch: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        B,
    ), f"Model Sigma output shape mismatch: {sigma_pred.shape}"
    assert torch.all(sigma_pred > 0), "Model produced non-positive sigma values"

    print("  -> DualPathTransformer forward pass verified.")

    # 6. Verify Trainer (Integration)
    print("\n[6] Verifying Trainer Integration...")
    trainer = Trainer(debug=True)

    # Ensure checkpoint directory is clean or exists
    if os.path.exists(Config.CHECKPOINT_DIR):
        # We don't delete it as per instructions not to delete provided things,
        # but we check if file is created later.
        pass

    print("  Starting training loop (1 epoch)...")
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model was saved
    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Trainer failed to save best_model.pth"

    print("  -> Trainer fit loop completed and model saved.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
