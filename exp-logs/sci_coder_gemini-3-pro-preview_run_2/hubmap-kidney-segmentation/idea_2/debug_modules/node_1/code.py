import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.model import AnatomyAwareUNetPlusPlus
from library.loss_metrics import HybridBCEDiceLoss, DiceScore
from library.trainer import Trainer
from library.inference import InferenceRunner

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Overrides Config settings for a fast demonstration run and prepares subset metadata.
    """
    print("[Demo] Setting up environment...")

    # 1. Override Config for speed
    # Use a separate directory for demo outputs
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update paths based on new working dir
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_tiles.npy")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_tiles.npy")

    # Training hyperparameters for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Use main process for simplicity

    # Data processing speedups
    # Increase stride significantly to reduce number of tiles generated from large TIFFs
    Config.STRIDE = 10000
    Config.TILE_SIZE = 1024
    # Aggressive undersampling of background for demo speed
    Config.BACKGROUND_SAMPLE_RATE = 0.01

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Create Subset Metadata (1 sample each)
    # This prevents processing the entire dataset
    subset_meta_dir = os.path.join(Config.WORKING_DIR, "metadata_subset")
    os.makedirs(subset_meta_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv("./metadata/train_metadata.csv")
    val_df = pd.read_csv("./metadata/val_metadata.csv")
    test_df = pd.read_csv("./metadata/test_metadata.csv")

    # Save subsets
    train_subset_path = os.path.join(subset_meta_dir, "train.csv")
    val_subset_path = os.path.join(subset_meta_dir, "val.csv")
    test_subset_path = os.path.join(subset_meta_dir, "test.csv")

    train_df.head(1).to_csv(train_subset_path, index=False)
    val_df.head(1).to_csv(val_subset_path, index=False)
    test_df.head(1).to_csv(test_subset_path, index=False)

    # Point Config to subsets
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.VAL_METADATA_PATH = val_subset_path
    Config.TEST_METADATA_PATH = test_subset_path

    print(f"[Demo] Config configured. Working dir: {Config.WORKING_DIR}")


def verify_model_logic():
    """
    Verifies model instantiation, input adaptation, and forward pass.
    """
    print("\n[Demo] Verifying Model Logic...")

    model = AnatomyAwareUNetPlusPlus()
    model.eval()

    # Check 1: Input layer adaptation
    # The first conv layer should have 4 input channels (3 RGB + 1 Anatomy)
    first_layer = model.model.encoder.conv1
    assert (
        first_layer.in_channels == 4
    ), f"Expected 4 input channels, got {first_layer.in_channels}"

    # Check 2: Weight initialization strategy
    # The 4th channel weights should be the mean of the RGB weights
    with torch.no_grad():
        rgb_weights = first_layer.weight[:, :3, :, :]
        anat_weights = first_layer.weight[:, 3:4, :, :]
        expected_anat = torch.mean(rgb_weights, dim=1, keepdim=True)
        # Allow small floating point differences
        assert torch.allclose(
            anat_weights, expected_anat, atol=1e-6
        ), "Anatomy channel weights not initialized as mean of RGB weights"

    # Check 3: Forward pass
    # Create dummy input (Batch=2, Channels=4, H=256, W=256)
    # Note: Using smaller spatial dim for speed check, model handles variable size
    dummy_input = torch.randn(2, 4, 256, 256)
    output = model(dummy_input)

    assert output.shape == (
        2,
        1,
        256,
        256,
    ), f"Expected output shape (2, 1, 256, 256), got {output.shape}"

    print("[Demo] Model logic verified successfully.")


def verify_loss_metrics():
    """
    Verifies the hybrid loss function and dice metric.
    """
    print("\n[Demo] Verifying Loss and Metrics...")

    loss_fn = HybridBCEDiceLoss()
    metric_fn = DiceScore(threshold=0.5)

    # Case 1: Perfect prediction (Logits -> Sigmoid -> 1.0)
    # Large positive logits for target 1, large negative for target 0
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits_perfect = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])

    loss = loss_fn(logits_perfect, targets)
    score = metric_fn(logits_perfect, targets)

    # Loss should be near 0, Score near 1
    assert (
        loss.item() < 0.01
    ), f"Expected near-zero loss for perfect pred, got {loss.item()}"
    assert (
        score.item() > 0.99
    ), f"Expected near-one score for perfect pred, got {score.item()}"

    # Case 2: Worst prediction
    logits_worst = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]])
    score_worst = metric_fn(logits_worst, targets)

    assert (
        score_worst.item() < 0.01
    ), f"Expected near-zero score for worst pred, got {score_worst.item()}"

    print("[Demo] Loss and Metrics verified successfully.")


def run_training_demo():
    """
    Runs the Trainer to demonstrate data loading and training loop.
    """
    print("\n[Demo] Starting Training Demo...")

    # Initialize Trainer
    # This will trigger data processing and caching based on our subset metadata
    trainer = Trainer(load_cached_data=False)

    # Run training
    # Config.EPOCHS is set to 1
    trainer.fit()

    # Verify model was saved
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print(f"[Demo] Training finished. Model saved to {Config.MODEL_PATH}")


def run_inference_demo():
    """
    Runs the InferenceRunner to demonstrate prediction and submission generation.
    """
    print("\n[Demo] Starting Inference Demo...")

    runner = InferenceRunner(model_path=Config.MODEL_PATH)
    runner.predict_and_submit()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify content format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = {"id", "predicted"}
    assert required_cols.issubset(
        df.columns
    ), f"Submission missing columns. Found {df.columns}"
    assert len(df) > 0, "Submission file is empty."

    print(f"[Demo] Inference finished. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"[Demo] Submission head:\n{df.head()}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Verify Components
        verify_model_logic()
        verify_loss_metrics()

        # 3. Run Pipeline
        run_training_demo()
        run_inference_demo()

        print("\n[Demo] All demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[Demo] Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Demo] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
