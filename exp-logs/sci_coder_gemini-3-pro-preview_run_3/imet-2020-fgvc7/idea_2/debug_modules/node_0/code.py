import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ArtworkModel
from library.train import run_training
from library.inference import run_inference
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def demo_dataset_and_model():
    """
    Demonstrates and verifies the DataLoaders and Model architecture.
    """
    print("\n=== 1. Verifying Dataset and Model Architecture ===")

    # 1. Get DataLoaders (Debug mode uses a very small subset)
    print("Initializing DataLoaders in debug mode...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduce workers for small demo
        debug=True,
        load_cached_data=False,  # Force reload to test processing logic
    )

    # 2. Verify Train Loader
    print("Fetching a batch from Train Loader...")
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    print(f"Image Batch Shape: {images.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected target shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {targets.shape}"
    assert (
        targets.dtype == torch.float32
    ), "Targets should be float32 for BCEWithLogitsLoss"

    # 3. Verify Model Forward Pass
    print("Instantiating ArtworkModel...")
    # We use pretrained=False here to ensure the unit test runs even if internet is restricted,
    # though the main training loop uses Config defaults (True).
    model = ArtworkModel(pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    print("Running forward pass...")
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected logits shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Dataset and Model verification passed.")


def demo_training_pipeline():
    """
    Demonstrates the training loop and threshold optimization.
    """
    print("\n=== 2. Verifying Training Pipeline ===")

    print(f"Starting training run with {Config.EPOCHS} epoch(s) on debug subset...")

    # run_training encapsulates the loop, validation, and threshold optimization
    trained_model, best_threshold = run_training(debug=True)

    print(f"Training finished. Optimized Threshold: {best_threshold}")

    # Assertions
    assert isinstance(best_threshold, float) or isinstance(
        best_threshold, np.float64
    ), "Threshold should be a float"
    assert (
        0.0 < best_threshold < 1.0
    ), f"Threshold {best_threshold} is out of expected range (0, 1)"
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"

    print("Training pipeline verification passed.")
    return trained_model, best_threshold


def demo_inference_pipeline(threshold):
    """
    Demonstrates inference and submission generation.
    """
    print("\n=== 3. Verifying Inference Pipeline ===")

    print(f"Running inference with threshold {threshold}...")

    # run_inference loads the model from Config.MODEL_PATH and generates submission
    run_inference(
        model_path=Config.MODEL_PATH,
        threshold=threshold,
        output_path=Config.SUBMISSION_PATH,
        debug=True,
    )

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df)}")
    print("Head of submission:")
    print(df.head())

    # Assertions
    assert "id" in df.columns, "Submission missing 'id' column"
    assert "attribute_ids" in df.columns, "Submission missing 'attribute_ids' column"
    assert len(df) > 0, "Submission dataframe is empty"

    # Check format of attribute_ids (should be string of space-separated ints or empty)
    # Note: In debug mode with random weights, predictions might be empty, which is valid.
    sample_pred = df.iloc[0]["attribute_ids"]
    if pd.notna(sample_pred) and len(str(sample_pred)) > 0:
        parts = str(sample_pred).split()
        assert all(
            p.isdigit() for p in parts
        ), "attribute_ids contains non-digit characters"

    print("Inference pipeline verification passed.")


if __name__ == "__main__":
    # --- Setup Configuration for Demo ---
    # We modify Config attributes directly to control the execution flow
    # without modifying the source file.

    # 1. Setup paths
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    Config.WORKING_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # 2. Optimize for Speed
    Config.DEBUG = True  # Use tiny subset
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.IMAGE_SIZE = 128  # Small image size for faster processing
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # 3. Set Seed
    set_seed(Config.SEED)

    print(f"Demo Configuration:")
    print(f"  Working Dir: {Config.WORKING_DIR}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Image Size: {Config.IMAGE_SIZE}")

    try:
        # --- Run Demos ---
        demo_dataset_and_model()
        _, threshold = demo_training_pipeline()
        demo_inference_pipeline(threshold)

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\n\n!!! DEMO FAILED !!!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
