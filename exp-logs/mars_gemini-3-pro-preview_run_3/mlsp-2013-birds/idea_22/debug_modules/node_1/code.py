import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, get_device
from library.data import get_dataloaders, mixup_data, mixup_criterion
from library.models import get_model
from library.optimization import get_optimizer, Lookahead
from library.training import train_fold
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== Starting Library Demonstration ====")

    # 1. Configuration Setup
    # Initialize Config with debug=True to use a small subset of data (32 train, 16 val, 16 test)
    # Set num_epochs=1 and batch_size=4 for speed.
    config = Config(debug=True, num_epochs=1, batch_size=4)

    # Override paths to use a specific demo directory
    # This ensures we don't interfere with other runs and forces data processing (since cache won't exist)
    base_demo_dir = "./working/demo_execution"
    if os.path.exists(base_demo_dir):
        shutil.rmtree(base_demo_dir)

    config.WORKING_DIR = base_demo_dir
    config.CACHE_DIR = os.path.join(base_demo_dir, "cache")
    config.CHECKPOINT_DIR = os.path.join(base_demo_dir, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(base_demo_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Restrict architectures to just one for the demo
    config.ARCHITECTURES = ["resnet18"]
    config.TOP_K_CHECKPOINTS = 1  # Save only the best model for the demo

    # Create directories
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    print(f"Configuration initialized. Working directory: {config.WORKING_DIR}")

    # 2. Test Utils
    print("\n[Test] Utils: calculate_roc_auc")
    # Case 1: Perfect prediction
    y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    auc = calculate_roc_auc(y_true, y_pred)
    assert auc == 1.0, f"Expected AUC 1.0, got {auc}"

    # Case 2: Random/Mixed (Robustness check)
    y_true_mixed = np.array([[0, 1], [0, 1], [0, 1]])  # Only one class present
    y_pred_mixed = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    # Should return 0.5 or handle gracefully without crashing
    auc_mixed = calculate_roc_auc(y_true_mixed, y_pred_mixed)
    print(f"  Robust AUC check passed: {auc_mixed}")

    # 3. Test Data Loading
    print("\n[Test] Data: get_dataloaders")
    # This will process the debug subset from scratch
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Train Loader
    images, labels = next(iter(train_loader))
    print(f"  Train Batch Shape - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == (config.BATCH_SIZE, 3, config.IMG_SIZE, config.IMG_SIZE)
    assert labels.shape == (config.BATCH_SIZE, config.NUM_CLASSES)
    assert images.dtype == torch.float32

    # Verify Mixup
    print("  Testing Mixup augmentation...")
    mixed_x, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0)
    assert mixed_x.shape == images.shape
    assert y_a.shape == labels.shape
    assert y_b.shape == labels.shape
    assert isinstance(lam, float)

    # 4. Test Model Instantiation
    print("\n[Test] Models: get_model")
    device = get_device()
    model = get_model("resnet18", config, pretrained=False)  # False for speed
    model.eval()

    # Forward pass check
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (2, config.NUM_CLASSES)

    # 5. Test Optimization
    print("\n[Test] Optimization: get_optimizer")
    optimizer = get_optimizer(model, config)
    assert isinstance(optimizer, Lookahead), "Optimizer should be wrapped in Lookahead"
    assert isinstance(
        optimizer.optimizer, torch.optim.AdamW
    ), "Base optimizer should be AdamW"
    print("  Optimizer initialized successfully.")

    # 6. Run Training (Short Demo)
    print("\n[Execution] Training Fold 0 with ResNet18...")
    # This runs the full training loop for 1 epoch on the debug dataset
    train_fold(config, fold=0, model_name="resnet18")

    # Verify checkpoint creation
    expected_ckpt = config.get_checkpoint_path("resnet18", fold=0, rank=0)
    assert os.path.exists(expected_ckpt), f"Checkpoint not found at {expected_ckpt}"
    print(f"  Checkpoint verified: {expected_ckpt}")

    # 7. Run Inference
    print("\n[Execution] Inference and Submission Generation...")
    # This uses the trained checkpoint to predict on the debug test set
    predict_and_submit(config)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {df_sub.shape}")

    # In debug mode, test set has 16 images.
    # Submission rows = 16 images * 19 classes = 304 rows.
    # The sample_submission format requires specific Ids.
    # Let's verify columns and non-emptiness.
    assert "Id" in df_sub.columns and "Probability" in df_sub.columns
    assert len(df_sub) > 0

    # Check Id format (should be integer)
    assert pd.api.types.is_integer_dtype(df_sub["Id"])

    print("==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
