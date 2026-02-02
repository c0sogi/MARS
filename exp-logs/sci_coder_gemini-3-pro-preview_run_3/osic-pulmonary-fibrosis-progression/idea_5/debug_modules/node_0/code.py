import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_loss, score_metric
from library.data import LungDataset, get_dataloaders
from library.model import WideAndDeepNet
from library.train import run_training


def demo_setup():
    """
    Sets up the environment and modifies Config for a fast demonstration run.
    """
    print("=== 1. Setup and Configuration ===")
    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config for a quick demonstration
    # These changes propagate to other modules as Config is a class with static attributes
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use a tiny subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    print(
        f"Configured for demo: Debug={Config.DEBUG}, Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}"
    )
    print("Setup complete.\n")


def demo_data_loading():
    """
    Demonstrates dataset instantiation and verifies data loading logic.
    """
    print("=== 2. Data Loading Demonstration ===")

    # Load metadata manually to verify LungDataset
    train_df = pd.read_csv(Config.TRAIN_META_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    print("Initializing LungDataset...")
    dataset = LungDataset(train_df, mode="train", load_cached_data=True)
    print(f"Dataset initialized with {len(dataset)} samples.")

    # Fetch a single item
    sample = dataset[0]

    # Verify keys
    expected_keys = [
        "image",
        "weeks",
        "baseline_fvc",
        "age",
        "sex",
        "smoke",
        "target",
        "patient_id",
    ]
    for k in expected_keys:
        assert k in sample, f"Missing key {k} in dataset sample"

    # Verify shapes
    # Image: (3, 224, 224) - 3 slices (Apical, Middle, Basal)
    assert sample["image"].shape == (
        3,
        224,
        224,
    ), f"Image shape mismatch: {sample['image'].shape}"
    # Target: (1,) scalar
    assert sample["target"].shape == (
        1,
    ), f"Target shape mismatch: {sample['target'].shape}"

    print("Dataset item verification passed.")

    # Verify DataLoaders
    print("Initializing DataLoaders...")
    train_loader, _ = get_dataloaders(train_df, train_df)  # Use same df for demo
    batch = next(iter(train_loader))

    assert batch["image"].shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("DataLoader batch verification passed.\n")
    return batch


def demo_model_forward(batch):
    """
    Demonstrates model instantiation and performs a forward pass.
    """
    print("=== 3. Model Instantiation and Forward Pass ===")

    device = Config.DEVICE
    print(f"Using device: {device}")

    model = WideAndDeepNet().to(device)
    model.eval()

    # Move batch to device
    img = batch["image"].to(device)
    weeks = batch["weeks"].to(device)
    base_fvc = batch["baseline_fvc"].to(device)
    age = batch["age"].to(device)
    sex = batch["sex"].to(device)
    smoke = batch["smoke"].to(device)

    # Forward pass
    print("Executing forward pass...")
    with torch.no_grad():
        mu, sigma = model(img, weeks, base_fvc, age, sex, smoke)

    # Verify output shapes: (Batch_Size, 1)
    assert mu.shape == (Config.BATCH_SIZE, 1), f"Mu shape mismatch: {mu.shape}"
    assert sigma.shape == (Config.BATCH_SIZE, 1), f"Sigma shape mismatch: {sigma.shape}"

    # Verify sigma positivity (Model enforces softplus + epsilon)
    assert (sigma > 0).all(), "Sigma must be positive"

    print(
        f"Model forward pass successful. Output shapes: mu={mu.shape}, sigma={sigma.shape}"
    )
    print("Sigma values are positive.\n")

    return mu, sigma


def demo_metrics(batch, mu, sigma):
    """
    Demonstrates loss and metric calculation.
    """
    print("=== 4. Loss and Metric Verification ===")

    target = batch["target"].to(Config.DEVICE)

    # Calculate Loss
    loss = laplace_log_likelihood_loss(target, mu, sigma)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # Calculate Metric (requires numpy arrays and unscaling)
    # Unscale for metric calculation demo
    mu_np = (mu * Config.FVC_STD + Config.FVC_MEAN).cpu().numpy().flatten()
    sigma_np = (sigma * Config.FVC_STD).cpu().numpy().flatten()
    target_np = (target * Config.FVC_STD + Config.FVC_MEAN).cpu().numpy().flatten()

    metric = score_metric(target_np, mu_np, sigma_np)
    print(f"Calculated Metric: {metric:.4f}")
    assert np.isfinite(metric), "Metric is not finite"

    print("Metric verification passed.\n")


def demo_full_training_pipeline():
    """
    Executes the full training pipeline (train, val, submission) using the library function.
    """
    print("=== 5. Full Training Pipeline Execution ===")

    # Run the training pipeline using the library function
    # This handles training, validation, checkpointing, and submission generation
    print("Running training loop (debug mode)...")
    best_score = run_training(debug=True, epochs=1)

    print(f"Pipeline finished with Best Validation Score: {best_score}")

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {sub_df.shape}")

        # Check columns
        expected_cols = ["Patient_Week", "FVC", "Confidence"]
        assert all(
            col in sub_df.columns for col in expected_cols
        ), "Submission columns mismatch"

        # Check values
        assert not sub_df.isnull().values.any(), "Submission contains null values"

        # Check if confidence is clipped as per metric requirements (>= 70)
        min_conf = sub_df["Confidence"].min()
        assert (
            min_conf >= 70
        ), f"Confidence values must be clipped at 70, found min {min_conf}"

        print("Submission file verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n")


if __name__ == "__main__":
    try:
        demo_setup()
        batch = demo_data_loading()
        mu, sigma = demo_model_forward(batch)
        demo_metrics(batch, mu, sigma)
        demo_full_training_pipeline()
        print("All demonstrations completed successfully.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        # Re-raise to ensure the task fails if there's an error
        raise e
