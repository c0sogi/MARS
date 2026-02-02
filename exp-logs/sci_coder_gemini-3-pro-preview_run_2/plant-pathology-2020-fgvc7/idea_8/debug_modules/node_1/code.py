import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rank_normalize, calculate_metric
from library.dataset import load_data, AppleDataset, get_transforms
from library.model import DiseaseClassifier
from library.trainer import SWATrainer


def setup_demo_config():
    """
    Overrides the default configuration for a fast demonstration run.
    """
    print("--- Setting up Demo Configuration ---")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small enough for instant execution

    # Training Hyperparameters for speed
    Config.EPOCHS = 1
    Config.N_FOLDS = 2  # Minimum folds to demonstrate CV loop
    Config.SWA_START_EPOCH = 0  # Start SWA immediately

    # Use a separate working directory for this demo to avoid cache conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Use a lightweight model (ResNet18) instead of the heavy ones in default config
    Config.MODEL_CONFIGS = [
        {
            "name": "resnet18",
            "img_size": 128,  # Reduced resolution
            "batch_size": 4,
            "dropout_rate": 0.0,
            "drop_path_rate": 0.0,
        }
    ]

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Config: {Config.MODEL_CONFIGS}")


def verify_utils():
    """
    Verifies utility functions.
    """
    print("\n--- Verifying Utils ---")

    # Test Rank Normalization
    probs = np.array([[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]])
    ranked = rank_normalize(probs)
    assert ranked.shape == probs.shape, "Rank normalize shape mismatch"
    assert ranked.min() >= 0.0 and ranked.max() <= 1.0, "Ranks out of bounds"
    print("rank_normalize: OK")

    # Test Metric Calculation
    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    score = calculate_metric(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "Metric out of bounds"
    print(f"calculate_metric: OK (Score: {score:.4f})")


def verify_dataset():
    """
    Verifies data loading and dataset construction.
    """
    print("\n--- Verifying Dataset ---")

    # Load Data (this will trigger cache creation in the demo working dir)
    train_df, val_df, test_df = load_data(load_cached_data=False)

    assert len(train_df) <= Config.DEBUG_SAMPLE_SIZE, "Train DF size incorrect"
    assert len(test_df) <= Config.DEBUG_SAMPLE_SIZE, "Test DF size incorrect"
    print(f"Data Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Test Dataset Class
    ds = AppleDataset(train_df, transform=get_transforms(128, "train"))
    img, label = ds[0]

    # Check shapes
    assert img.shape == (3, 128, 128), f"Image tensor shape mismatch: {img.shape}"
    assert label.shape == (2,), f"Label shape mismatch: {label.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    print("AppleDataset: OK")

    return test_df


def verify_model():
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model ---")

    model_name = Config.MODEL_CONFIGS[0]["name"]
    model = DiseaseClassifier(model_name=model_name, num_classes=2, pretrained=False)
    model.eval()

    # Dummy Input
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2), f"Model output shape mismatch: {output.shape}"
    print(f"Model ({model_name}) Forward Pass: OK")


def verify_training_pipeline():
    """
    Runs the SWATrainer to verify the training loop and inference.
    """
    print("\n--- Verifying Training Pipeline (SWATrainer) ---")

    trainer = SWATrainer()

    # This runs the CV loop and then generates submission
    # Since we set EPOCHS=1 and N_FOLDS=2 with tiny data, this should be fast.
    trainer.run()

    # Check if artifacts were created
    fold_0_path = os.path.join(Config.WORKING_DIR, "best_model_resnet18_fold_0.pth")
    assert os.path.exists(fold_0_path), "Model checkpoint for Fold 0 not found"

    # Check Submission
    sub_path = "submission.csv"
    assert os.path.exists(sub_path), "submission.csv not found"

    sub_df = pd.read_csv(sub_path)
    print(f"Submission generated with shape: {sub_df.shape}")

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Verify row count matches the debug test set size
    # Note: load_data was called inside trainer.run() again, respecting DEBUG_SAMPLE_SIZE
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(sub_df)}"

    print("Training Pipeline: OK")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        # 1. Setup
        setup_demo_config()

        # 2. Verify Components
        verify_utils()
        test_df = verify_dataset()
        verify_model()

        # 3. Run Full Pipeline
        verify_training_pipeline()

        print("\n=== All Demonstrations Completed Successfully ===")

    except AssertionError as e:
        print(f"\n!!! Verification Failed: {e} !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
