import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import library components
from library.config import Config
from library.utils import set_seeds, get_device
from library.data_loader import get_dataloaders
from library.model import RLK_RN
from library.trainer import Trainer
from library.inference import InferenceEngine


def main():
    print(">>> Starting RLK-RN Demo Script")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Configuring environment...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_work_dir = "./working/demo_run_script"

    # Override Config paths and parameters for the demo
    Config.WORK_DIR = demo_work_dir
    Config.CACHE_DIR = os.path.join(demo_work_dir, "cache")
    Config.MODEL_DIR = demo_work_dir  # Save model in the demo root
    Config.SUBMISSION_DIR = demo_work_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "submission.csv")

    # Optimization for speed (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples for training
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 2  # Small batch size
    Config.WINDOW_SIZE = 32  # Smaller window for faster processing
    Config.STRIDE = 32  # Non-overlapping for speed

    # Initialize directories
    Config.setup()

    # Set seeds for reproducibility
    set_seeds(Config.SEED)

    # ==========================================
    # 2. Prepare Subset Metadata
    # ==========================================
    # To ensure the InferenceEngine runs quickly, we create mini versions of the metadata CSVs.
    print(">>> Creating subset metadata for rapid testing...")

    os.makedirs(os.path.join(demo_work_dir, "metadata"), exist_ok=True)

    # Load original metadata
    full_train = pd.read_csv("./metadata/train.csv")
    full_val = pd.read_csv("./metadata/val.csv")
    full_test = pd.read_csv("./metadata/test.csv")

    # Create subsets
    mini_train = full_train.head(Config.DEBUG_SUBSET_SIZE)
    mini_val = full_val.head(5)
    mini_test = full_test.head(5)

    # Define paths for mini metadata
    mini_train_path = os.path.join(demo_work_dir, "metadata", "train.csv")
    mini_val_path = os.path.join(demo_work_dir, "metadata", "val.csv")
    mini_test_path = os.path.join(demo_work_dir, "metadata", "test.csv")

    # Save to disk
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # ==========================================
    # 3. Data Loading Demonstration
    # ==========================================
    print(">>> Instantiating DataLoaders...")
    # num_workers=0 to avoid multiprocessing overhead in this short script
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=0
    )

    print(">>> Verifying Data Shapes...")
    try:
        # Fetch one batch
        features, labels = next(iter(train_loader))

        print(
            f"    Features Shape: {features.shape}"
        )  # Expected: (B, Window, InputDim)
        print(f"    Labels Shape:   {labels.shape}")  # Expected: (B, Window)

        # Assertions
        assert features.shape == (
            Config.BATCH_SIZE,
            Config.WINDOW_SIZE,
            Config.INPUT_DIM,
        ), f"Feature shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM)}, got {features.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
            Config.WINDOW_SIZE,
        ), "Label shape mismatch."

        print("    [Success] Data loading logic verified.")

    except StopIteration:
        print(
            "    [Warning] DataLoader returned no data. Check if subset samples are valid."
        )
        return

    # ==========================================
    # 4. Model Demonstration
    # ==========================================
    print(">>> Instantiating Model (RLK-RN)...")
    device = get_device()
    model = RLK_RN().to(device)

    print(">>> Verifying Model Forward Pass...")
    features = features.to(device)

    # Forward pass
    outputs = model(features)

    # RLK-RN returns a list of 3 outputs: [stage1, stage2, stage3]
    assert isinstance(outputs, list), "Model output should be a list."
    assert len(outputs) == 3, "Model should return outputs for 3 stages."

    # Check shape of the final stage output
    final_output = outputs[2]
    expected_shape = (Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert (
        final_output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {final_output.shape}"

    print("    [Success] Model architecture verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print(">>> Starting Training Loop (2 Epochs)...")
    trainer = Trainer(model, train_loader, val_loader)

    # Run training
    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"    [Success] Best model saved to {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Training finished but best_model.pth was not created.")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print(">>> Starting Inference Engine...")

    # Initialize inference engine with the model we just trained
    engine = InferenceEngine(model_path=Config.BEST_MODEL_PATH)

    # Generate submission for the mini test set
    # load_cached_data=False ensures we process the new mini-test data
    engine.generate_submission(
        output_path=Config.SUBMISSION_PATH, load_cached_data=False
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        with open(Config.SUBMISSION_PATH, "r") as f:
            lines = f.readlines()

        num_preds = len(lines)
        expected_preds = len(mini_test)

        print(f"    Generated {num_preds} predictions.")
        assert (
            num_preds == expected_preds
        ), f"Expected {expected_preds} predictions matching test set size, got {num_preds}."

        print(f"    [Success] Submission generated at {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print(">>> Demo completed successfully.")


if __name__ == "__main__":
    main()
