import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# 1. Suppress Warnings and Progress Bars
warnings.filterwarnings("ignore")

# Patch tqdm to disable progress bars as per requirements
# We need to patch it before importing the library modules that use it
import tqdm.auto


class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass


# Apply patch to sys.modules to ensure library imports pick it up if possible,
# or patch the module directly after import.
tqdm.auto.tqdm = SilentTqdm

# 2. Configuration Override for Speed and Testing
from library.config import Config

print("Configuring environment for rapid demonstration...")
# Enable Debug mode to use a small subset of data
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for quick execution

# Reduce computational load
Config.IMG_SIZE = 128  # Smaller images
Config.BATCH_SIZE = 8
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in script
Config.EPOCHS = 1  # Single epoch for demonstration
Config.GRADIENT_ACCUM_STEPS = 1

# Use a lightweight model for demonstration purposes
# 'resnet18' is standard and available in timm
Config.MODEL_NAME = "resnet18"
Config.PRETRAINED = False  # Skip downloading heavy weights

# Ensure directories exist (Config does this, but good to double check context)
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# 3. Import Library Modules
# We import these after patching tqdm and modifying Config
from library.utils import seed_everything, get_score
from library.dataset import get_loaders
from library.model import AppleDiseaseModel
from library.train import train_loop
from library.inference import predict_fn
import library.train
import library.inference

# Apply tqdm patch specifically to the imported modules to be safe
library.train.tqdm = SilentTqdm
library.inference.tqdm = SilentTqdm


def main():
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # Step 1: Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[1/5] Verifying Utilities...")
    seed_everything(Config.SEED)

    # Verify Metric Calculation (F1 Score)
    # Case: Perfect match
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred_perfect = np.array([[1, 0, 1], [0, 1, 0]])
    score_perfect = get_score(y_true, y_pred_perfect)
    assert score_perfect == 1.0, f"Expected F1 score 1.0, got {score_perfect}"

    # Case: Partial match
    # y_true: [1, 0, 1], [0, 1, 0]
    # y_pred: [1, 0, 0], [0, 1, 0] -> First sample missed one label
    y_pred_partial = np.array([[1, 0, 0], [0, 1, 0]])
    score_partial = get_score(y_true, y_pred_partial)
    assert (
        0.0 <= score_partial < 1.0
    ), "F1 score should be between 0 and 1 for partial match"
    print(" - Utils verification passed.")

    # -------------------------------------------------------------------------
    # Step 2: Verify Dataset and DataLoaders
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_loaders()

    # Verify Train Loader
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Check shapes
    # Images: (B, C, H, W) -> (8, 3, 128, 128)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    # Targets: (B, Num_Classes) -> (8, 6)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    # Check target values (should be multi-hot: 0.0 or 1.0)
    assert torch.all(
        (targets == 0.0) | (targets == 1.0)
    ), "Targets must be binary (0 or 1)"

    print(f" - Batch loaded successfully. Image shape: {images.shape}")
    print(" - Dataset verification passed.")

    # -------------------------------------------------------------------------
    # Step 3: Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)

    # Instantiate model
    model = AppleDiseaseModel(
        model_name=Config.MODEL_NAME, pretrained=False, num_classes=Config.NUM_CLASSES
    )
    model.to(device)

    # Forward pass with the batch from Step 2
    images = images.to(device)
    targets = targets.to(device)

    logits = model(images)

    # Check output shape
    assert (
        logits.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"

    # Check loss calculation
    loss = model.get_loss(logits, targets)
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    print(f" - Forward pass successful. Loss: {loss.item():.4f}")
    print(" - Model verification passed.")

    # -------------------------------------------------------------------------
    # Step 4: Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[4/5] Running Training Loop (1 Epoch, Debug Subset)...")

    # Run the training loop provided in library.train
    # This will train for 1 epoch on the small subset and save 'best_model.pth'
    trained_model = train_loop()

    # Verify artifact creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), "Training loop failed to save 'best_model.pth'"

    print(" - Training loop completed successfully.")
    print(f" - Model checkpoint saved at: {best_model_path}")

    # -------------------------------------------------------------------------
    # Step 5: Verify Inference
    # -------------------------------------------------------------------------
    print("\n[5/5] Running Inference...")

    # Run inference using the trained model
    # limit_batches ensures we don't iterate too long even on the subset
    predict_fn(checkpoint_path=best_model_path, limit_batches=2)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Inference failed to create submission file"

    # Check submission content format
    df_sub = pd.read_csv(submission_path)
    print(f" - Submission file loaded. Rows: {len(df_sub)}")

    required_cols = ["image", "labels"]
    for col in required_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    # Check if labels are strings
    assert df_sub["labels"].dtype == object, "Labels column should be text"

    print(" - Inference verification passed.")
    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
