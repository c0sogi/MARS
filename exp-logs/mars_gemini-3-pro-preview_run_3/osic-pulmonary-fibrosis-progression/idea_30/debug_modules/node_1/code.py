import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, get_test_dataloader
from library.model import MAZR_DS
from library.train import LaplaceNLLLoss, run_training


def run_demo():
    print("=== Starting MAZR-DS Demo Execution ===")

    # 1. Configuration & Setup
    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINTS_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    # Use a smaller image size if needed for speed, but keeping 260 ensures model compatibility

    # Setup directories
    Config.setup()
    seed_everything(Config.SEED)

    print("\n[1] Verifying Data Loading...")
    # Load data
    train_loader, val_loader, stats = get_dataloaders(load_cached_data=True)

    # Check stats
    assert "fvc_mean" in stats
    assert "fvc_std" in stats
    print(f"   Stats loaded: Mean FVC={stats['fvc_mean']:.2f}")

    # Fetch one batch
    imgs, tabular, targets = next(iter(train_loader))

    # Verify shapes
    # Imgs: [B, 3, 260, 260]
    assert imgs.ndim == 4
    assert imgs.shape[1] == 3
    assert imgs.shape[2] == Config.IMG_SIZE
    assert imgs.shape[3] == Config.IMG_SIZE

    # Tabular: [B, 5]
    assert tabular.ndim == 2
    assert tabular.shape[1] == 5

    # Targets: [B]
    assert targets.ndim == 1
    assert targets.shape[0] == imgs.shape[0]

    print(
        f"   Batch shapes verified: Imgs {imgs.shape}, Tabular {tabular.shape}, Targets {targets.shape}"
    )

    print("\n[2] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = MAZR_DS().to(device)

    # Move batch to device
    imgs = imgs.to(device)
    tabular = tabular.to(device)

    # Forward pass
    mu, sigma = model(imgs, tabular)

    # Verify output shapes and constraints
    assert mu.shape == targets.shape
    assert sigma.shape == targets.shape
    # Sigma must be positive
    assert torch.all(sigma > 0), "Sigma predictions must be positive"

    print("   Forward pass successful. Output shapes correct.")

    print("\n[3] Verifying Loss Function...")
    criterion = LaplaceNLLLoss()
    targets = targets.to(device)

    loss = criterion(mu, sigma, targets)

    # Check loss is scalar and valid
    assert loss.dim() == 0
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print(f"   Loss calculation successful: {loss.item():.4f}")

    print("\n[4] Running Full Training Loop (1 Epoch)...")
    # This calls the library function which runs the loop, validation, and saving
    best_metric = run_training(load_cached_data=True)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    assert isinstance(best_metric, float), "run_training did not return a float metric"

    print(f"   Training complete. Best Metric: {best_metric}")

    print("\n[5] Verifying Inference & Submission Generation...")
    # Get test loader
    test_loader, sub_df = get_test_dataloader(
        stats, batch_size=4, load_cached_data=True
    )

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    predictions = []
    confidences = []

    print("   Running inference on test set...")
    with torch.no_grad():
        for imgs_test, tabular_test, _ in test_loader:
            imgs_test = imgs_test.to(device)
            tabular_test = tabular_test.to(device)

            mu_test, sigma_test = model(imgs_test, tabular_test)

            # Denormalize
            mu_raw = mu_test.cpu().numpy() * stats["fvc_std"] + stats["fvc_mean"]
            sigma_raw = sigma_test.cpu().numpy() * stats["fvc_std"]

            predictions.extend(mu_raw)
            confidences.extend(sigma_raw)

    # Verify prediction count matches submission file
    assert len(predictions) == len(sub_df)
    assert len(confidences) == len(sub_df)

    # Update submission dataframe
    sub_df["FVC"] = predictions
    sub_df["Confidence"] = confidences

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(submission_path, index=False)

    print(f"   Submission generated at {submission_path}")
    print(f"   First 3 rows:\n{sub_df.head(3)}")

    print("\n[6] Verifying Metric Logic...")
    # Test case 1: Perfect prediction
    # If pred == true, delta = 0. Metric = -ln(sqrt(2)*sigma_clipped)
    # If sigma = 70 (min), metric = -ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    y_true = np.array([2000.0])
    y_pred = np.array([2000.0])
    sigma_pred = np.array([50.0])  # Should clip to 70

    score = calculate_metric(y_true, y_pred, sigma_pred)
    expected_score = -np.log(np.sqrt(2) * 70)
    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric mismatch. Got {score}, expected {expected_score}"

    # Test case 2: Large error (clipped at 1000)
    y_true_bad = np.array([2000.0])
    y_pred_bad = np.array([4000.0])  # Delta 2000 -> clipped to 1000
    sigma_bad = np.array([100.0])

    score_bad = calculate_metric(y_true_bad, y_pred_bad, sigma_bad)
    # Metric = - (sqrt(2)*1000)/100 - ln(sqrt(2)*100)
    #        = - 14.142 - ln(141.42)
    #        = - 14.142 - 4.95
    #        = - 19.09 approx

    delta = 1000.0
    sigma_c = 100.0
    expected_bad = -(np.sqrt(2) * delta) / sigma_c - np.log(np.sqrt(2) * sigma_c)

    assert np.isclose(score_bad, expected_bad, atol=1e-4)

    print("   Metric logic verified.")
    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
