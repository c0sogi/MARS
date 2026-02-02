import os
import sys
import pandas as pd
import torch
import numpy as np
import torch.optim as optim

# Import from provided library
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, prepare_inference_data
from library.model import MultiViewNet
from library.train import train_one_epoch, validate


def run_demo():
    print("=== Starting Demonstration of Lung Function Decline Prediction Pipeline ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    seed_everything(Config.seed)

    # Override Config for speed
    # We reduce epochs and batch size, and point to subset CSVs
    Config.epochs = 2
    Config.batch_size = 4
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Create a subset of data for training/validation to ensure speed
    os.makedirs(Config.working_dir, exist_ok=True)

    # Load original metadata
    full_train_df = pd.read_csv(Config.train_csv_path)
    full_val_df = pd.read_csv(Config.val_csv_path)

    # Create subsets (8 rows for train, 4 rows for val)
    subset_train_path = os.path.join(Config.working_dir, "train_subset.csv")
    subset_val_path = os.path.join(Config.working_dir, "val_subset.csv")

    full_train_df.head(8).to_csv(subset_train_path, index=False)
    full_val_df.head(4).to_csv(subset_val_path, index=False)

    # Update Config paths to point to subsets
    Config.train_csv_path = subset_train_path
    Config.val_csv_path = subset_val_path

    print(f"    Subset training data saved to: {subset_train_path}")
    print(f"    Subset validation data saved to: {subset_val_path}")
    print(f"    Configured Epochs: {Config.epochs}, Batch Size: {Config.batch_size}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading and Processing...")
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers
    )

    print(f"    Train Loader batches: {len(train_loader)}")
    print(f"    Val Loader batches: {len(val_loader)}")

    # Fetch one batch to verify structure and shapes
    images, tabular, targets = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Tabular: {tabular.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Images: (B, 3, 256, 256)
    assert images.shape == (
        Config.batch_size,
        3,
        Config.img_size,
        Config.img_size,
    ), "Image tensor shape mismatch"
    # Tabular: (B, 5) -> Age, Sex, Smoke, Weeks, Base_FVC
    assert tabular.shape == (Config.batch_size, 5), "Tabular tensor shape mismatch"
    # Targets: (B,)
    assert targets.shape == (Config.batch_size,), "Target tensor shape mismatch"

    print("    Data loading logic verified.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = MultiViewNet().to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward pass
    mu, sigma = model(images, tabular)

    print(f"    Output Shapes -> Mu: {mu.shape}, Sigma: {sigma.shape}")
    print(f"    Sample Sigma values: {sigma.detach().cpu().numpy()}")

    # Assertions
    assert mu.shape == (Config.batch_size,), "Output Mu shape mismatch"
    assert sigma.shape == (Config.batch_size,), "Output Sigma shape mismatch"
    assert torch.all(sigma > 0), "Sigma must be positive"

    print("    Model forward pass verified.")

    # 4. Metric Calculation Verification
    print("\n[4] Verifying Metric Calculation...")
    # Create deterministic dummy data
    # True FVC: 2000
    # Pred FVC: 2100 (Delta = 100)
    # Pred Sigma: 50 (Clipped to 70)
    # Metric = - (sqrt(2) * 100) / 70 - ln(sqrt(2) * 70)
    #        = - (1.4142 * 100) / 70 - ln(98.99)
    #        = - 2.0203 - 4.5950 = -6.6153 approx

    y_true = np.array([2000.0])
    y_pred = np.array([2100.0])
    y_conf = np.array([50.0])

    metric_val = calculate_metric(y_true, y_pred, y_conf)
    print(f"    Calculated Metric: {metric_val:.4f}")

    # Manual check
    sigma_clipped = max(50.0, 70.0)
    delta = min(abs(2000.0 - 2100.0), 1000.0)
    expected = -(np.sqrt(2) * delta) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )

    assert np.isclose(
        metric_val, expected, atol=1e-4
    ), f"Metric mismatch. Got {metric_val}, expected {expected}"
    print("    Metric calculation verified.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Short Training Loop...")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate
    )

    for epoch in range(1, Config.epochs + 1):
        print(f"    --- Epoch {epoch} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, device, epoch)
        print(f"    Train Loss: {train_loss:.4f}")

        val_metric = validate(val_loader, model, device)
        print(f"    Val Metric: {val_metric:.4f}")

        # Basic sanity check
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_metric), "Validation metric is NaN"

    print("    Training loop execution verified.")

    # 6. Inference Preparation Verification
    print("\n[6] Verifying Inference Preparation...")
    # This reads the actual test.csv and sample_submission.csv
    # The pipeline merges them and creates a dataset.
    # Since there are only ~18 unique patients in the test set, caching images is fast.

    test_loader, sub_df = prepare_inference_data()
    print(f"    Test Loader batches: {len(test_loader)}")
    print(f"    Submission DataFrame shape: {sub_df.shape}")

    # Check one batch
    test_images, test_tabular = next(iter(test_loader))
    print(
        f"    Test Batch -> Images: {test_images.shape}, Tabular: {test_tabular.shape}"
    )

    # Assertions
    assert test_images.shape[1:] == (
        3,
        Config.img_size,
        Config.img_size,
    ), "Test image dimensions incorrect"
    assert test_tabular.shape[1] == 5, "Test tabular dimensions incorrect"

    print("    Inference preparation verified.")

    print("\n=== Demonstration Complete: All components functioning as expected. ===")


if __name__ == "__main__":
    run_demo()
