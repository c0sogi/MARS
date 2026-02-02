import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import PCDSNet
from library.utils import metric_laplace_log_likelihood
from library.train import Runner


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("[1/6] Setting up configuration for demo execution...")

    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config parameters for a fast demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo

    # Define a separate working directory for this demo to avoid overwriting existing work
    Config.WORKING_DIR = "./working/demo_task_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create the necessary directories
    Config.setup()

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[2/6] Initializing Data Loaders...")

    train_loader, val_loader, test_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print("    Fetching a sample batch from Train Loader...")
    # Fetch one batch to verify shapes
    img_batch, clinical_batch, target_batch = next(iter(train_loader))

    print(f"    Image Batch Shape: {img_batch.shape}")  # Expected: (B, 3, 260, 260)
    print(f"    Clinical Batch Shape: {clinical_batch.shape}")  # Expected: (B, 5)
    print(f"    Target Batch Shape: {target_batch.shape}")  # Expected: (B,)

    # Assertions to ensure data integrity
    assert img_batch.ndim == 4, "Image batch should be 4D (B, C, H, W)"
    assert img_batch.shape[1] == 3, "Images should have 3 channels"
    assert clinical_batch.shape[1] == 5, "Clinical features should have dimension 5"
    assert target_batch.ndim == 1, "Target should be a 1D vector"

    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Inference
    # -------------------------------------------------------------------------
    print("\n[3/6] Instantiating PCDSNet and running forward pass...")

    device = Config.DEVICE
    model = PCDSNet().to(device)

    # Move batch to device
    img_batch = img_batch.to(device)
    clinical_batch = clinical_batch.to(device)

    # Forward pass
    mu, sigma = model(img_batch, clinical_batch)

    print(f"    Prediction Mean (mu) Shape: {mu.shape}")
    print(f"    Prediction Conf (sigma) Shape: {sigma.shape}")

    # Assertions
    assert mu.shape == target_batch.shape, "Mu output shape mismatch"
    assert sigma.shape == target_batch.shape, "Sigma output shape mismatch"
    assert torch.all(sigma > 0), "Sigma (confidence) must be positive"

    print("    Model inference verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Calculation Demonstration
    # -------------------------------------------------------------------------
    print("\n[4/6] Calculating Metric on sample batch...")

    # Denormalize values (Metric expects absolute ml, not Z-scores)
    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    mu_abs = mu * fvc_std + fvc_mean
    sigma_abs = sigma * fvc_std
    target_abs = target_batch.to(device) * fvc_std + fvc_mean

    # Calculate metric
    score = metric_laplace_log_likelihood(target_abs, mu_abs, sigma_abs)

    print(f"    Sample Batch Score: {score:.4f}")
    assert isinstance(score, float), "Metric function should return a float"

    print("    Metric calculation verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5/6] Executing Training Loop (via Runner)...")

    # Instantiate Runner with modified config
    runner = Runner(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, device=device)

    # Run training
    runner.run()

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Checkpoint successfully created at: {best_model_path}")
    else:
        raise FileNotFoundError(
            "Training completed but 'best_model.pth' was not found."
        )

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating Submission...")

    runner.generate_submission()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"    Submission file created with {len(sub_df)} rows.")
        print("    First 3 rows:")
        print(sub_df.head(3))

        # Validate submission format
        required_cols = ["Patient_Week", "FVC", "Confidence"]
        for col in required_cols:
            assert col in sub_df.columns, f"Submission missing column: {col}"

        # Check constraints
        assert (
            sub_df["Confidence"].min() >= 70
        ), "Confidence values must be clipped at 70"

        print("    Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nAll demo tasks completed successfully!")


if __name__ == "__main__":
    main()
