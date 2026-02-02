import os
import sys
import torch
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.utils import Config, set_seed
from library.dataset import create_dataloaders
from library.model import EfficientNetV2B0
from library.engine import train_model, predict_and_submit


def run_demonstration():
    print("Initializing Configuration...")
    # Initialize Config with debug settings for speed
    # We use a very small sample size and 1 epoch to verify the pipeline quickly.
    config = Config(
        epochs=1,
        batch_size=8,
        debug=True,
        debug_sample_size=50,  # Small subset for demonstration
        learning_rate=1e-3,
    )

    # Redirect output paths to a demo directory to avoid conflicts
    config.working_dir = "./working/demo_execution"
    config.model_path = os.path.join(config.working_dir, "demo_model.pth")
    config.submission_path = os.path.join(config.working_dir, "demo_submission.csv")

    os.makedirs(config.working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.submission_path), exist_ok=True)

    set_seed(config.seed)

    print("-" * 40)
    print("1. Verifying Data Loading...")
    loaders = create_dataloaders(config)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Check Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"   Batch Image Shape: {images.shape}")
        print(f"   Batch Label Shape: {labels.shape}")

        # Assertions
        assert images.shape == (
            config.batch_size,
            3,
            config.img_size,
            config.img_size,
        ), "Incorrect image batch shape."
        assert labels.shape == (config.batch_size,), "Incorrect label batch shape."
        assert labels.dtype == torch.float32, "Labels should be float32."
        print("   -> Train loader verification passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    # Check Test Loader
    try:
        t_images, t_ids = next(iter(test_loader))
        assert t_images.shape == (
            config.batch_size,
            3,
            config.img_size,
            config.img_size,
        ), "Incorrect test image batch shape."
        # IDs might be int or tensor depending on collate, dataset returns raw item.
        # DataLoader default collate converts ints to tensor.
        assert len(t_ids) == config.batch_size, "Incorrect number of IDs in batch."
        print("   -> Test loader verification passed.")
    except StopIteration:
        raise AssertionError("Test loader is empty.")

    print("-" * 40)
    print("2. Verifying Model Architecture...")
    device = config.device
    model = EfficientNetV2B0(
        pretrained=False
    )  # Use false here just for shape check speed
    model.to(device)
    model.eval()

    with torch.no_grad():
        dummy_input = torch.randn(2, 3, config.img_size, config.img_size).to(device)
        output = model(dummy_input)

        print(f"   Input Shape: {dummy_input.shape}")
        print(f"   Output Shape: {output.shape}")

        assert output.shape == (2, 1), "Model output shape mismatch. Expected (B, 1)."
        print("   -> Model architecture verification passed.")

    print("-" * 40)
    print("3. Executing Training Loop (Engine)...")
    # This function handles the loop, validation, and saving the best model
    trained_model = train_model(config)

    # Verify model file creation
    if not os.path.exists(config.model_path):
        raise AssertionError(f"Model file was not created at {config.model_path}")

    print(f"   -> Model successfully saved to {config.model_path}")
    print("   -> Training execution passed.")

    print("-" * 40)
    print("4. Executing Inference and Submission...")
    predict_and_submit(config)

    # Verify submission file
    if not os.path.exists(config.submission_path):
        raise AssertionError(f"Submission file not found at {config.submission_path}")

    df_sub = pd.read_csv(config.submission_path)
    print(f"   Submission rows: {len(df_sub)}")
    print(f"   Submission columns: {list(df_sub.columns)}")

    # Assertions on submission
    assert (
        "id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns."
    assert (
        len(df_sub) == config.debug_sample_size
    ), f"Submission length mismatch. Expected {config.debug_sample_size} (debug size), got {len(df_sub)}."

    # Check probability range
    assert (
        df_sub["label"].min() >= 0.0 and df_sub["label"].max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print("   -> Inference and submission verification passed.")

    print("-" * 40)
    print("Demonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
