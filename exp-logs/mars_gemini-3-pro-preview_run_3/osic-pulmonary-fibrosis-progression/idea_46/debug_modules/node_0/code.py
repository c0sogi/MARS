import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.data import LungDataset, get_dataloaders, get_submission_dataloader
from library.model import BCOSRNet
from library.utils import seed_everything, LaplaceNLLLoss, inverse_scale
from library.engine import run_training, generate_submission


def demo_configuration():
    """
    Sets up the configuration for a fast demonstration run.
    """
    print("\n=== 1. Configuring Environment for Demo ===")

    # Override Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.N_DEBUG_SAMPLES = 10  # Very small subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # Ensure working directory is clean/ready
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")


def demo_data_loading():
    """
    Demonstrates and validates the data loading pipeline.
    """
    print("\n=== 2. Testing Data Pipeline ===")

    # 1. Test Dataset Instantiation
    # Load metadata manually to create a dataset instance
    train_df = pd.read_csv(Config.TRAIN_CSV).head(Config.N_DEBUG_SAMPLES)
    dataset = LungDataset(train_df, mode="train")

    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == Config.N_DEBUG_SAMPLES, "Dataset length mismatch"

    # 2. Test __getitem__
    image, clinical, target = dataset[0]

    print(f"Image Shape: {image.shape}")  # Expected: (3, 260, 260)
    print(f"Clinical Shape: {clinical.shape}")  # Expected: (5,)
    print(f"Target Shape: {target.shape}")  # Expected: scalar (0-d tensor)

    assert image.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image dimensions"
    assert clinical.shape == (5,), "Incorrect clinical feature dimensions"
    assert isinstance(target, torch.Tensor), "Target is not a tensor"

    # 3. Test DataLoaders
    train_loader, val_loader = get_dataloaders(debug=True)
    batch_imgs, batch_clin, batch_targets = next(iter(train_loader))

    print(f"Batch Image Shape: {batch_imgs.shape}")
    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_imgs.shape[1] == 3, "Channel dimension mismatch"


def demo_model_architecture():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== 3. Testing Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = BCOSRNet().to(device)
    model.eval()

    # Create dummy inputs
    B = 2
    dummy_imgs = torch.randn(B, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_clin = torch.randn(B, 5).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_imgs, dummy_clin)

    print(f"Model Output Shape: {outputs.shape}")  # Expected: (B, 2)

    # Validation
    assert outputs.shape == (B, 2), "Model output shape incorrect"

    # Check constraints: Sigma (2nd column) must be positive and >= floor
    # Note: The model outputs normalized sigma. The constraint is applied in the forward pass.
    # The floor in the model is: F.softplus(logit) + (70 / TARGET_STD)
    sigma_preds_norm = outputs[:, 1]
    min_sigma_norm = Config.SIGMA_FLOOR / Config.TARGET_STD

    print(f"Min Predicted Sigma (Norm): {sigma_preds_norm.min().item():.4f}")
    print(f"Theoretical Floor (Norm): {min_sigma_norm:.4f}")

    assert torch.all(
        sigma_preds_norm >= min_sigma_norm
    ), "Sigma floor constraint violated"


def demo_loss_function():
    """
    Demonstrates the custom Laplace NLL Loss.
    """
    print("\n=== 4. Testing Loss Function ===")

    loss_fn = LaplaceNLLLoss()

    # Dummy predictions (Normalized)
    pred_mean = torch.tensor([0.1, -0.1], requires_grad=True)
    pred_sigma = torch.tensor([0.5, 0.5], requires_grad=True)  # Ensure > floor

    # Dummy targets (Normalized)
    targets = torch.tensor([0.1, -0.1])  # Perfect prediction scenario

    # Calculate loss
    loss = loss_fn(pred_mean, pred_sigma, targets)

    print(f"Loss Value: {loss.item():.4f}")

    # Backprop check
    loss.backward()
    assert pred_mean.grad is not None, "Gradients not flowing to mean"
    assert pred_sigma.grad is not None, "Gradients not flowing to sigma"

    # Check logic: Perfect prediction should still have loss due to log(sigma) term
    # Loss ~= log(sqrt(2) * sigma_abs) since delta is 0
    # sigma_abs = 0.5 * 801.7 = 400.85
    # expected ~= log(1.414 * 400.85) ~= log(566) ~= 6.33
    print("Loss calculation seems valid.")


def demo_full_pipeline():
    """
    Runs the full training and submission generation pipeline on a subset.
    """
    print("\n=== 5. Running Full Training Pipeline (Debug Mode) ===")

    # 1. Run Training
    # This calls run_training from library.engine
    # It will train for Config.EPOCHS (set to 1) on the debug subset
    best_metric = run_training(debug=True)

    print(f"Training finished. Best Metric: {best_metric}")

    # Check if checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"

    # 2. Generate Submission
    print("\n=== 6. Generating Submission ===")
    generate_submission()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission Rows: {len(sub_df)}")
    print(sub_df.head())

    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Configure
        demo_configuration()

        # 2. Data
        demo_data_loading()

        # 3. Model
        demo_model_architecture()

        # 4. Loss
        demo_loss_function()

        # 5. Pipeline
        demo_full_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print full traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
