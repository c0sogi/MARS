import os
import sys
import shutil
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import library components
import library.config
import library.data
import library.model
import library.train

# Cite debug_lesson_12: Verify Fix Execution (Ghost Fix)
importlib.reload(library.config)
importlib.reload(library.data)
importlib.reload(library.model)
importlib.reload(library.train)

from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_data, LungDataset
from library.model import CI_OP_DS_Net
from library.train import SmoothLaplaceLoss, run_training


def verify_data_pipeline():
    print("\n=== Verifying Data Pipeline ===")
    # Load data in debug mode (small subset)
    train_ds, val_ds, test_ds, processor = get_data(debug=True)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")
    print(f"Test Dataset Size: {len(test_ds)}")

    # Verify item structure
    # Expected: img, meta_a, meta_b, target
    sample = train_ds[0]
    img, meta_a, meta_b, target = sample

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Meta A Shape: {meta_a.shape}")
    print(f"Sample Meta B Shape: {meta_b.shape}")
    print(f"Sample Target Shape: {target.shape}")

    # Assertions
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    assert meta_a.shape == (
        5,
    ), f"Incorrect Meta A shape (expected 5 features): {meta_a.shape}"
    assert meta_b.shape == (
        2,
    ), f"Incorrect Meta B shape (expected 2 features): {meta_b.shape}"
    assert target.shape == (1,), f"Incorrect target shape: {target.shape}"

    return train_ds


def verify_model_forward_pass(dataset):
    print("\n=== Verifying Model Forward Pass ===")
    device = torch.device("cpu")  # Use CPU for simple logic check
    model = CI_OP_DS_Net().to(device)
    model.eval()

    # Create a batch of size 2
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    img, meta_a, meta_b, target = next(iter(loader))

    with torch.no_grad():
        mu, sigma = model(img, meta_a, meta_b)

    print(f"Output Mean (mu) Shape: {mu.shape}")
    print(f"Output Sigma Shape: {sigma.shape}")
    print(f"Sigma values: {sigma.numpy()}")

    # Assertions
    assert mu.shape == (2,), "Output mean shape mismatch"
    assert sigma.shape == (2,), "Output sigma shape mismatch"
    assert torch.all(sigma > 0), "Sigma must be positive (Softplus constraint)"

    return model, img, meta_a, meta_b, target


def verify_loss_function(model, img, meta_a, meta_b, target):
    print("\n=== Verifying Loss Function ===")
    criterion = SmoothLaplaceLoss()

    # Forward pass to get predictions
    with torch.no_grad():
        mu, sigma = model(img, meta_a, meta_b)

    # Ensure target is correct shape (B,)
    target = target.squeeze(-1)

    loss = criterion(mu, sigma, target)
    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    assert loss.item() != 0, "Loss should not be exactly zero for random init"


def verify_metric_logic():
    print("\n=== Verifying Metric Logic ===")
    # Metric: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    # sigma_clipped = max(sigma, 70)
    # delta = min(|true - pred|, 1000)

    # Case 1: Perfect prediction, Sigma=100
    # Delta = 0
    # Term1 = 0
    # Term2 = ln(sqrt(2)*100) = ln(141.42) approx 4.95
    # Score approx -4.95
    y_true = np.array([2000])
    y_pred = np.array([2000])
    y_sigma = np.array([100])

    score_1 = metric_score(y_true, y_pred, y_sigma)
    print(f"Metric (Perfect Match, Sigma=100): {score_1:.4f}")
    assert (
        -5.0 < score_1 < -4.9
    ), f"Metric calculation incorrect for Case 1. Got {score_1}"

    # Case 2: Error=100, Sigma=50 (Clipped to 70)
    # Delta = 100
    # Sigma_clipped = 70
    # Term1 = sqrt(2)*100 / 70 = 1.414 * 1.428 = 2.02
    # Term2 = ln(sqrt(2)*70) = ln(98.99) = 4.595
    # Score = -2.02 - 4.595 = -6.615
    y_true = np.array([2000])
    y_pred = np.array([2100])
    y_sigma = np.array([50])

    score_2 = metric_score(y_true, y_pred, y_sigma)
    print(f"Metric (Error=100, Sigma=50->70): {score_2:.4f}")
    assert (
        -6.7 < score_2 < -6.5
    ), f"Metric calculation incorrect for Case 2. Got {score_2}"


def run_demo_training():
    print("\n=== Running Demo Training Loop ===")

    # Override Config for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Run training with debug parameters
    # debug=True in run_training calls get_data(debug=True) internally
    run_training(epochs=2, batch_size=4, debug=True)

    # Verify output
    expected_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(expected_model_path):
        print(f"Success: Best model saved at {expected_model_path}")
        file_size = os.path.getsize(expected_model_path)
        print(f"Model file size: {file_size / (1024*1024):.2f} MB")
    else:
        raise FileNotFoundError("Training failed to save best_model.pth")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Verify Data Loading and Processing
        train_ds = verify_data_pipeline()

        # 2. Verify Model Architecture
        model, img, meta_a, meta_b, target = verify_model_forward_pass(train_ds)

        # 3. Verify Loss Calculation
        verify_loss_function(model, img, meta_a, meta_b, target)

        # 4. Verify Metric Calculation
        verify_metric_logic()

        # 5. Run Integration Test (Training Loop)
        run_demo_training()

        print("\nAll verification steps completed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
