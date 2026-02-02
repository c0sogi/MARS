import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library components
from library.config import Config
from library.feature_extractor import FeatureExtractor
from library.dataset import SpeechCommandDataset
from library.model import SKResNetConformer
from library.trainer import Trainer


def run_demo():
    print("=== Setting up Configuration for Demo ===")
    # 1. Optimize Config for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup working directories for the demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR)
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Create directories
    Config.setup()

    # 2. Create Subset Metadata for Fast Execution
    print("=== Creating Subset Metadata ===")
    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Sample subsets (50 train, 20 val, 20 test)
    demo_train = orig_train.head(50).copy()
    demo_val = orig_val.head(20).copy()
    demo_test = orig_test.head(20).copy()

    # Save to working dir
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Override Config paths to point to subsets
    Config.TRAIN_METADATA = demo_train_path
    Config.VAL_METADATA = demo_val_path
    Config.TEST_METADATA = demo_test_path

    # Disable internal sampling in Trainer since we provided small files
    Config.MAX_TRAIN_SAMPLES = None
    Config.MAX_VAL_SAMPLES = None

    print("=== Verifying Feature Extractor ===")
    # Pick a file from the demo train set
    sample_row = demo_train.iloc[0]
    sample_rel_path = sample_row["filepath"]
    sample_full_path = os.path.join(Config.INPUT_ROOT, sample_rel_path)

    # Compute features
    features = FeatureExtractor.compute_multires_spec(sample_full_path)

    # Validate shape: (3 channels, 64 mels, 101 time steps)
    # Time steps = 1 + int(16000 * 1.0 / 160) = 101
    expected_shape = (3, 64, 101)
    if features.shape != expected_shape:
        raise AssertionError(
            f"Feature shape mismatch. Expected {expected_shape}, got {features.shape}"
        )

    if not isinstance(features, np.ndarray):
        raise AssertionError("Features should be a numpy array")

    print(f"Feature Extractor Verified. Output shape: {features.shape}")

    print("=== Verifying Dataset ===")
    # Initialize Dataset with demo dataframe
    ds = SpeechCommandDataset(demo_train, augment=True)

    # Check length
    if len(ds) != 50:
        raise AssertionError(f"Dataset length mismatch. Expected 50, got {len(ds)}")

    # Check item retrieval
    data_tensor, label_idx = ds[0]

    # Validate Tensor shape
    if data_tensor.shape != expected_shape:
        raise AssertionError(
            f"Dataset tensor shape mismatch. Expected {expected_shape}, got {data_tensor.shape}"
        )

    # Validate Label
    if not isinstance(label_idx, int):
        raise AssertionError("Label should be an integer")
    if label_idx < 0 or label_idx >= Config.NUM_CLASSES:
        raise AssertionError(
            f"Label index {label_idx} out of bounds (0-{Config.NUM_CLASSES-1})"
        )

    print("Dataset Verified.")

    print("=== Verifying Model Architecture ===")
    model = SKResNetConformer()

    # Create dummy batch: (Batch=2, Channels=3, Freq=64, Time=101)
    dummy_input = torch.randn(2, 3, 64, 101)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch=2, NumClasses=12)
    expected_out_shape = (2, Config.NUM_CLASSES)
    if output.shape != expected_out_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_out_shape}, got {output.shape}"
        )

    print("Model Architecture Verified.")

    print("=== Running Trainer (Integration Test) ===")
    # Initialize Trainer
    # This will load the subset metadata, cache the features, and setup loaders
    trainer = Trainer()

    # Run Training (1 Epoch)
    print("Starting fit()...")
    trainer.fit()

    # Check if model was saved
    if not os.path.exists(Config.MODEL_PATH):
        # Note: Model is only saved if validation accuracy improves.
        # In 1 epoch with random init, it might not "improve" from 0.0 if logic is strict,
        # but usually 0.0 is the baseline.
        # Let's force a save if not present for the sake of the prediction test,
        # or ensure the Trainer logic handles it.
        # The provided Trainer saves if val_acc > best_acc (init 0.0).
        # If val_acc is 0, it won't save.
        print(
            "Model not saved by Trainer (likely low accuracy). Saving manually for prediction test."
        )
        torch.save(trainer.model.state_dict(), Config.MODEL_PATH)

    # Run Inference
    print("Starting predict()...")
    trainer.predict()

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    if len(sub_df) != 20:
        raise AssertionError(
            f"Submission length mismatch. Expected 20, got {len(sub_df)}"
        )

    print(f"Integration Test Complete. Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
