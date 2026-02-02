import sys
import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import DogCatDataset, get_transforms
from library.model import DogCatClassifier, ModelEMA
from library.engine import run_training


def run_demo():
    # 1. Setup and Configuration Override for Speed
    print(">>> Setting up configuration for fast demonstration...")

    # Override Config attributes to ensure the demo runs in < 60 seconds
    Config.SEED = 42
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for train/val/test
    Config.IMG_SIZE = 128  # Reduce image size for faster processing
    Config.MODEL_NAME = "resnet18"  # Use a lightweight model instead of ConvNeXt
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directories exist
    Config.setup()

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    logger = get_logger("demo")
    logger.info("Configuration updated for speed.")

    # 2. Verify Dataset Logic
    print("\n>>> Verifying Dataset Logic...")

    # Load a small slice of metadata manually to test Dataset class
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    train_df = pd.read_csv(Config.TRAIN_CSV).head(10)
    transform = get_transforms("train")

    # Instantiate Dataset
    dataset = DogCatDataset(
        train_df, Config.INPUT_DIR, transform=transform, is_test=False
    )

    # Test __len__
    assert (
        len(dataset) == 10
    ), f"Dataset length mismatch. Expected 10, got {len(dataset)}"

    # Test __getitem__
    img, label = dataset[0]

    # Check Image Tensor
    assert isinstance(img, torch.Tensor), "Image is not a torch.Tensor"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert img.dtype == torch.float32, "Image dtype is not float32"

    # Check Label
    assert isinstance(label, torch.Tensor), "Label is not a torch.Tensor"
    assert label.ndim == 0, "Label should be a scalar tensor"

    logger.info("Dataset logic verified successfully.")

    # 3. Verify Model Logic
    print("\n>>> Verifying Model Logic...")
    device = Config.DEVICE

    # Instantiate Model (using the lighter resnet18 defined in config override)
    model = DogCatClassifier(model_name=Config.MODEL_NAME, pretrained=False).to(device)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape (Batch, 1) for binary classification
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    logger.info("Model forward pass verified successfully.")

    # 4. Verify EMA Logic
    print("\n>>> Verifying EMA Logic...")
    # Initialize EMA with a decay that allows visible updates
    ema = ModelEMA(model, decay=0.5)

    # Get a parameter to track (e.g., first parameter)
    param_name, param = next(model.named_parameters())
    ema_param = dict(ema.get_model().named_parameters())[param_name]

    # Ensure they start identical
    assert torch.allclose(
        param, ema_param
    ), "EMA parameters should initially match model parameters"

    # Modify model parameter manually
    with torch.no_grad():
        param.add_(1.0)

    # Update EMA
    ema.update(model)

    # Check if EMA moved.
    # New EMA = decay * Old EMA + (1-decay) * New Model
    # Since Old EMA != New Model, New EMA must differ from Old EMA (and New Model)
    assert not torch.allclose(
        ema_param, param
    ), "EMA parameter should not be equal to model parameter after update"

    logger.info("EMA logic verified successfully.")

    # 5. Run Full Training Pipeline (Debug Mode)
    print("\n>>> Running Full Training Pipeline (Debug Mode)...")

    # This will run training for 1 epoch on 20 samples, validate, and predict on 20 test samples
    # It uses the functions in library.engine
    run_training(debug=True)

    # 6. Verify Submission Output
    print("\n>>> Verifying Submission...")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    sub_df = pd.read_csv(submission_path)

    # Check length: Should be equal to Config.DEBUG_SUBSET_SIZE because run_training(debug=True) subsets the test set too
    expected_len = Config.DEBUG_SUBSET_SIZE
    assert (
        len(sub_df) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"

    # Check columns
    assert list(sub_df.columns) == [
        "id",
        "label",
    ], f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check value ranges
    assert (
        sub_df["label"].min() >= 0.0 and sub_df["label"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    logger.info(f"Submission verified. File saved at {submission_path}")
    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
