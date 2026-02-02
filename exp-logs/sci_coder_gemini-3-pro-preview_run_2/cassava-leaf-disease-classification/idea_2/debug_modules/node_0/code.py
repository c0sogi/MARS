import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_accuracy
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaClassifier
from library.engine import run_training
from library.inference import run_inference

# Suppress warnings
warnings.filterwarnings("ignore")


def verify_utilities():
    """
    Verifies utility functions like seed setting and accuracy calculation.
    """
    print("\n=== Verifying Utilities ===")

    # 1. Test Seed
    seed_everything(Config.SEED)
    r1 = torch.rand(1).item()
    seed_everything(Config.SEED)
    r2 = torch.rand(1).item()
    assert r1 == r2, "Seed setting did not produce reproducible results."
    print("Seed verification passed.")

    # 2. Test Accuracy Calculation
    # Scenario: 2 samples.
    # Sample 0: Pred class 1 (Logits: [0.1, 0.9]), Target 1 -> Correct
    # Sample 1: Pred class 0 (Logits: [0.8, 0.2]), Target 1 -> Incorrect
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 1])
    acc = calculate_accuracy(logits, targets)

    expected_acc = 0.5
    assert (
        acc == expected_acc
    ), f"Accuracy calculation failed. Expected {expected_acc}, got {acc}"
    print("Accuracy calculation verification passed.")


def verify_dataset_and_transforms():
    """
    Verifies the Dataset class and Transformations.
    """
    print("\n=== Verifying Dataset & Transforms ===")

    subset_size = 16

    # Initialize Dataset (Train split)
    dataset = CassavaDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        data_split="train",
        subset_size=subset_size,
    )

    # Check length
    assert (
        len(dataset) == subset_size
    ), f"Dataset length mismatch. Expected {subset_size}, got {len(dataset)}"

    # Check item retrieval
    img, label = dataset[0]

    # Check Image Tensor Shape: [Channels, Height, Width]
    expected_shape = (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert (
        img.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {img.shape}"

    # Check Label Type
    assert isinstance(label, torch.Tensor), "Label is not a torch Tensor."
    assert (
        0 <= label.item() < Config.NUM_CLASSES
    ), f"Label {label.item()} is out of bounds (0-{Config.NUM_CLASSES-1})."

    print(
        f"Dataset verification passed. Sample shape: {img.shape}, Label: {label.item()}"
    )


def verify_model():
    """
    Verifies the Model architecture and forward pass.
    """
    print("\n=== Verifying Model Architecture ===")

    model = CassavaClassifier(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # False for speed in demo, though Config has True
        num_classes=Config.NUM_CLASSES,
    )
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_input)

    # Check Output Shape: [Batch, Num_Classes]
    expected_shape = (batch_size, Config.NUM_CLASSES)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"Model verification passed. Output shape: {output.shape}")


def run_training_demo():
    """
    Runs a minimal training loop using the engine.
    """
    print("\n=== Running Training Demo ===")

    # Use a very small subset and 1 epoch for speed
    subset_size = 32  # One batch
    epochs = 1

    # Run training
    best_acc = run_training(subset_size=subset_size, epochs=epochs, patience=1)

    # Verify Checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"

    print(f"Training demo completed. Best Accuracy: {best_acc}")


def run_inference_demo():
    """
    Runs the inference pipeline and verifies submission generation.
    """
    print("\n=== Running Inference Demo ===")

    subset_size = 20

    # Run inference
    submission_df = run_inference(subset_size=subset_size)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify DataFrame shape
    assert (
        len(submission_df) == subset_size
    ), f"Submission rows mismatch. Expected {subset_size}, got {len(submission_df)}"
    assert list(submission_df.columns) == [
        "image_id",
        "label",
    ], "Submission columns mismatch."

    # Verify Label content (should be integers)
    assert pd.api.types.is_integer_dtype(
        submission_df["label"]
    ), "Submission label column is not integer type."

    print("Inference demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    print(f"Using Device: {Config.DEVICE}")

    # 1. Verify Utilities
    verify_utilities()

    # 2. Verify Dataset
    verify_dataset_and_transforms()

    # 3. Verify Model
    verify_model()

    # 4. Run Training Pipeline (Minimal)
    run_training_demo()

    # 5. Run Inference Pipeline (Minimal)
    run_inference_demo()

    print("\nAll demonstrations and verifications passed successfully.")
