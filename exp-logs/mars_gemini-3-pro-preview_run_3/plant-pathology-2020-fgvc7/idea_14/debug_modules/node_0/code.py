import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Configuration Override
# =============================================================================
# We import Config and modify it before importing other modules that might use it.
from library.config import Config

print("Configuring environment for demo execution...")

# Enable Debug mode to use data subsets (100 train, 50 val)
Config.DEBUG = True

# Reduce training duration
Config.EPOCHS = 1
Config.BATCH_SIZE = 8  # Small batch size
Config.NUM_WORKERS = 2

# Use lightweight backbones for speed
Config.TEACHER_BACKBONE = "resnet18"
Config.STUDENT_BACKBONE = "resnet18"

# Reduce image size for faster processing
Config.TEACHER_IMG_SIZE = 128
Config.STUDENT_IMG_SIZE = 128

# Redirect outputs to a specific demo directory
Config.WORKING_DIR = "./working/demo_execution"
Config.SUBMISSION_DIR = "./working/demo_submission"

# Update file paths based on new directories
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

Config.TEACHER_CHECKPOINT = os.path.join(Config.WORKING_DIR, "teacher_demo.pth")
Config.STUDENT_CHECKPOINT = os.path.join(Config.WORKING_DIR, "student_demo.pth")
Config.FINAL_SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
Config.CLASS_WEIGHTS_PATH = os.path.join(Config.WORKING_DIR, "class_weights.npy")

# =============================================================================
# 2. Import Library Modules
# =============================================================================
from library.utils import seed_everything, get_class_weights
from library.dataset import load_data, get_transforms, AppleDataset
from library.models import AppleNet
from library.losses import HybridLoss, DistillationLoss
from library.train_engine import run_training
from library.inference import run_inference

# =============================================================================
# 3. Demonstration Functions
# =============================================================================


def demo_utils():
    """Demonstrates utility functions."""
    print("\n[Demo] Utils...")
    seed_everything(Config.SEED)

    # Create a dummy dataframe to test class weight calculation
    dummy_data = {
        "healthy": [1, 0, 0, 0, 1],
        "multiple_diseases": [0, 1, 0, 0, 0],
        "rust": [0, 0, 1, 0, 0],
        "scab": [0, 0, 0, 1, 0],
        "file_path": ["a.jpg"] * 5,
        "image_id": [f"id_{i}" for i in range(5)],
    }
    df = pd.DataFrame(dummy_data)

    # Calculate weights (force re-calculation by ignoring cache)
    weights = get_class_weights(df, load_cached_data=False)

    print(f"  Computed Class Weights: {weights.cpu().numpy()}")

    # Assertions
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (4,)
    assert torch.all(weights > 0), "Weights must be positive"


def demo_dataset():
    """Demonstrates dataset loading and transformation."""
    print("\n[Demo] Dataset...")

    # Load actual training metadata
    # Config.DEBUG is True, but load_data loads the full file; subsetting happens in training loop.
    # We manually subset here for the demo.
    df = load_data(Config.TRAIN_CSV, "train_demo_cache", load_cached_data=False)
    df_subset = df.head(10).copy()

    # Initialize Dataset
    transforms = get_transforms("train", Config.TEACHER_IMG_SIZE)
    dataset = AppleDataset(df_subset, transforms=transforms, mode="train")

    print(f"  Dataset size: {len(dataset)}")

    # Fetch one sample
    image, label, image_id = dataset[0]

    print(f"  Sample Image Shape: {image.shape}")
    print(f"  Sample Label: {label}")
    print(f"  Sample ID: {image_id}")

    # Assertions
    assert image.shape == (3, Config.TEACHER_IMG_SIZE, Config.TEACHER_IMG_SIZE)
    assert label.shape == (4,)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


def demo_model_and_loss():
    """Demonstrates model instantiation, forward pass, and loss calculation."""
    print("\n[Demo] Model & Loss...")

    # Instantiate Model (using the lighter ResNet18 backbone defined in Config override)
    model = AppleNet(Config.TEACHER_BACKBONE, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    batch_size = 2
    dummy_input = torch.randn(
        batch_size, 3, Config.TEACHER_IMG_SIZE, Config.TEACHER_IMG_SIZE
    ).to(Config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"  Model Output Keys: {list(outputs.keys())}")
    print(f"  Main Head Output Shape: {outputs['main'].shape}")

    # Assertions for Model
    assert "main" in outputs
    assert "rust" in outputs
    assert "scab" in outputs
    assert outputs["main"].shape == (batch_size, 4)
    assert outputs["rust"].shape == (batch_size, 1)

    # Test Loss Function
    # Create dummy targets (one-hot)
    dummy_targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).to(
        Config.DEVICE
    )

    loss_fn = HybridLoss()
    loss = loss_fn(outputs, dummy_targets)

    print(f"  Calculated Loss: {loss.item():.4f}")

    # Assertions for Loss
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"


def demo_pipeline():
    """Executes the full training and inference pipeline."""
    print("\n[Demo] Full Pipeline Execution...")
    print("  Note: Running with Config.DEBUG=True (subset of data) and 1 Epoch.")

    # 1. Run Training
    # This will train Teacher then Student, saving checkpoints to Config.WORKING_DIR
    run_training()

    # Verify Checkpoints
    if not os.path.exists(Config.TEACHER_CHECKPOINT):
        raise FileNotFoundError("Teacher checkpoint was not created.")
    if not os.path.exists(Config.STUDENT_CHECKPOINT):
        raise FileNotFoundError("Student checkpoint was not created.")

    print("  Training complete. Checkpoints verified.")

    # 2. Run Inference
    # This loads the checkpoints and generates the submission file
    run_inference()

    # Verify Submission
    if not os.path.exists(Config.FINAL_SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    submission_df = pd.read_csv(Config.FINAL_SUBMISSION_PATH)
    print(f"  Submission generated at: {Config.FINAL_SUBMISSION_PATH}")
    print(f"  Submission shape: {submission_df.shape}")
    print("  Head of submission:")
    print(submission_df.head(3))

    # Assertions
    assert "image_id" in submission_df.columns
    assert len(submission_df) > 0
    # Check probability columns exist
    for col in Config.CLASSES:
        assert col in submission_df.columns


# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    try:
        demo_utils()
        demo_dataset()
        demo_model_and_loss()
        demo_pipeline()
        print("\nSUCCESS: All demonstrations completed without error.")
    except Exception as e:
        print(f"\nFAILURE: An error occurred during demonstration: {e}")
        raise e
