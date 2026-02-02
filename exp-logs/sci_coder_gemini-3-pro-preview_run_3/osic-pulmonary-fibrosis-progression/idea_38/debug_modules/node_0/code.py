import os
import torch
import numpy as np
import pandas as pd
import math

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import SCOSRNet
from library.train import train_one_epoch, validate, run_inference


def demonstrate_task():
    # 1. Setup and Configuration Override for Demo Speed
    print(">>> Setting up environment and configuration...")
    seed_everything(Config.SEED)

    # Override Config for rapid execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid multiprocessing overhead in demo

    # Ensure working directories exist (Config.setup_directories called on import, but good to ensure)
    Config.setup_directories()

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading and Verification
    print("\n>>> Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
    )

    print("Verifying Train Loader batch structure...")
    # Fetch one batch
    images, clinical, targets, meta = next(iter(train_loader))

    # Verify Shapes
    # Image: (Batch, 3, H, W)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Clinical: (Batch, 4) -> [Age, Sex, Smoking, RelativeTime]
    expected_clin_shape = (Config.BATCH_SIZE, Config.CLINICAL_INPUT_DIM)
    assert (
        clinical.shape == expected_clin_shape
    ), f"Clinical shape mismatch. Expected {expected_clin_shape}, got {clinical.shape}"

    # Targets: (Batch, 1) -> Normalized FVC
    expected_target_shape = (Config.BATCH_SIZE, 1)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("Data loading verification passed.")

    # 3. Metric Verification
    print("\n>>> Verifying Laplace Log Likelihood Metric...")
    # Test Case:
    # True FVC = 2000, Pred FVC = 2000 (Delta=0), Sigma = 100
    # Metric = - (sqrt(2) * 0) / 100 - ln(sqrt(2) * 100)
    #        = - ln(141.421356)
    #        = - 4.95174

    y_true = torch.tensor([2000.0])
    y_pred = torch.tensor([2000.0])
    sigma = torch.tensor([100.0])

    metric_val = laplace_log_likelihood(y_true, y_pred, sigma)
    expected_val = -np.log(np.sqrt(2) * 100)

    print(f"Calculated Metric: {metric_val.item():.5f}")
    print(f"Expected Metric:   {expected_val:.5f}")

    assert np.isclose(
        metric_val.item(), expected_val, atol=1e-4
    ), "Metric calculation failed verification."
    print("Metric verification passed.")

    # 4. Model Instantiation and Forward Pass
    print("\n>>> Instantiating SCOSRNet and testing forward pass...")
    model = SCOSRNet().to(device)

    # Move batch to device
    images = images.to(device)
    clinical = clinical.to(device)

    # Forward pass
    mu_final, sigma_final, mu_base, sigma_base = model(images, clinical)

    # Verify Output Shapes
    assert mu_final.shape == (Config.BATCH_SIZE, 1), "mu_final shape incorrect"
    assert sigma_final.shape == (Config.BATCH_SIZE, 1), "sigma_final shape incorrect"

    # Verify Sigma Positivity (Softplus output)
    assert (sigma_final > 0).all(), "sigma_final contains non-positive values"
    assert (sigma_base > 0).all(), "sigma_base contains non-positive values"

    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n>>> Running simplified training loop (1 Epoch)...")

    # Setup Optimizer (Differential Learning Rates)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Get stats for denormalization inside loss function
    stats = train_loader.dataset.stats
    fvc_mean = stats.get("fvc_mean", 2500.0)
    fvc_std = stats.get("fvc_std", 500.0)

    # Train one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, device, fvc_mean, fvc_std
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_metric = validate(model, val_loader, device, fvc_mean, fvc_std)
    print(f"Validation Metric: {val_metric:.4f}")

    # Save Model (Required for inference step)
    print(f"Saving model to {Config.BEST_MODEL_PATH}...")
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model file was not saved."

    # 6. Inference Demonstration
    print("\n>>> Running Inference on Test Set...")

    # Run inference (loads the saved model internally)
    run_inference(test_loader, device)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())

    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column {col} in submission."

    print("\n>>> Task demonstration completed successfully.")


if __name__ == "__main__":
    demonstrate_task()
