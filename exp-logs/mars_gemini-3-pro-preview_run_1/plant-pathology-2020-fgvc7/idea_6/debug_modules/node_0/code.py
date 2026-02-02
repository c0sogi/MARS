import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import AppleDataset, get_transforms, mixup, cutmix
from library.model import AppleResNet
from library.loss import WeightedSoftCrossEntropy
from library.train import run_training
from library.inference import predict

# Setup Logger
logger = get_logger("demo_script")


def verify_dataset_and_transforms():
    logger.info("=== Verifying Dataset and Transforms ===")

    # Load metadata
    df = pd.read_csv(Config.TRAIN_METADATA_PATH).head(10)

    # Initialize Dataset
    dataset = AppleDataset(df, transform=get_transforms("train"))

    # Check length
    assert (
        len(dataset) == 10
    ), f"Dataset length mismatch. Expected 10, got {len(dataset)}"

    # Check item structure
    sample = dataset[0]
    assert "image" in sample, "Sample missing 'image' key"
    assert "target" in sample, "Sample missing 'target' key"
    assert "image_id" in sample, "Sample missing 'image_id' key"

    # Check tensor shapes
    # Config.IMG_SIZE is 256
    img = sample["image"]
    target = sample["target"]

    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch. Expected (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
    assert target.shape == (
        4,
    ), f"Target shape mismatch. Expected (4,), got {target.shape}"

    logger.info("Dataset structure and shapes verified.")

    # Verify Mixup/CutMix
    logger.info("Verifying Augmentations (Mixup/CutMix)...")
    batch_size = 4
    # Create dummy batch
    dummy_images = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    dummy_targets = torch.randint(0, 2, (batch_size, 4)).float()

    # Mixup
    mixed_imgs, _, _, lam = mixup(
        dummy_images.clone(), dummy_targets.clone(), alpha=1.0
    )
    assert mixed_imgs.shape == dummy_images.shape, "Mixup output shape mismatch"
    assert 0 <= lam <= 1, "Mixup lambda out of range"

    # CutMix
    cut_imgs, _, _, lam = cutmix(dummy_images.clone(), dummy_targets.clone(), alpha=1.0)
    assert cut_imgs.shape == dummy_images.shape, "CutMix output shape mismatch"

    logger.info("Augmentations verified.")


def verify_model_architecture():
    logger.info("=== Verifying Model Architecture ===")

    model = AppleResNet()
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, N_Classes)
    expected_shape = (batch_size, Config.N_CLASSES)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    logger.info("Model forward pass and output shape verified.")


def verify_loss_function():
    logger.info("=== Verifying Loss Function ===")

    # Initialize Loss
    # We use reduction='mean' by default
    criterion = WeightedSoftCrossEntropy()

    batch_size = 4
    n_classes = 4

    # Dummy Logits (Model Output)
    logits = torch.randn(batch_size, n_classes, requires_grad=True)

    # Dummy Soft Targets (e.g., from Mixup)
    targets = torch.softmax(torch.randn(batch_size, n_classes), dim=1)

    # Calculate Loss
    loss = criterion(logits, targets)

    # Check if loss is a scalar tensor
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"

    # Check backward pass capability
    loss.backward()
    assert logits.grad is not None, "Gradients not computed for logits"

    logger.info("Loss calculation and backward pass verified.")


def run_pipeline_demo():
    logger.info("=== Running Full Pipeline Demo (Training + Inference) ===")

    # 1. Patch Configuration for Speed
    # We modify the Config class attributes directly to run a very fast demo
    Config.EPOCHS = 1  # Only 1 epoch
    Config.N_FOLDS = 2  # Only 2 folds to verify cross-validation loop
    Config.DEBUG_SAMPLE_SIZE = 32  # Small dataset size
    Config.BATCH_SIZE = 8  # Small batch size

    # Update working directory to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean state
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    # 2. Run Training
    logger.info("Starting Training Demo...")
    try:
        run_training(debug=True)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e

    # Check if model files were created
    model_fold_0 = os.path.join(Config.WORKING_DIR, "resnet34_fold_0.pth")
    assert os.path.exists(
        model_fold_0
    ), f"Model file for Fold 0 not found at {model_fold_0}"
    logger.info("Training Demo completed successfully.")

    # 3. Run Inference
    logger.info("Starting Inference Demo...")
    try:
        predict(debug=True)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise e

    # Check if submission file was created
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "image_id" in sub_df.columns, "Submission missing image_id column"
    assert len(sub_df) > 0, "Submission file is empty"
    # Check if all class columns are present
    for label in Config.CLASS_LABELS:
        assert label in sub_df.columns, f"Submission missing column {label}"

    logger.info("Inference Demo completed successfully.")
    logger.info(f"Submission generated at: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    try:
        # Step 1: Component Verification
        verify_dataset_and_transforms()
        verify_model_architecture()
        verify_loss_function()

        # Step 2: Pipeline Execution
        run_pipeline_demo()

        logger.info("\nAll demonstrations and verifications passed successfully!")

    except AssertionError as e:
        logger.error(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        # Print stack trace for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
