import os
import sys
import shutil
import pandas as pd
import torch
import warnings
import importlib

# Ensure library is in path
sys.path.append(".")

# Cite debug_lesson_1: Reload modules to ensure updates are applied
import library.config
import library.utils
import library.dataset
import library.model
import library.engine

importlib.reload(library.config)
importlib.reload(library.utils)
importlib.reload(library.dataset)
importlib.reload(library.model)
importlib.reload(library.engine)

from library.config import Config, seed_everything
from library.dataset import ChestXRayDataset
from library.model import get_one_stage_detector, predict_and_submit
from library.engine import fit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a temporary working directory and generates small subsets of the
    metadata to ensure the demonstration runs quickly.
    """
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    # 1. Create Subset Metadata
    # Train Subset (Use random sample to avoid blocks of bad images)
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    train_subset_path = os.path.join(demo_dir, "train.csv")
    df_train.sample(n=32, random_state=42).to_csv(train_subset_path, index=False)

    # Val Subset (16 samples)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_subset_path = os.path.join(demo_dir, "val.csv")
    df_val.sample(n=16, random_state=42).to_csv(val_subset_path, index=False)

    # Test Subset (10 samples)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_subset_path = os.path.join(demo_dir, "test.csv")
    df_test.head(10).to_csv(test_subset_path, index=False)

    print(f"Created data subsets in {demo_dir}")

    # 2. Patch Configuration for Demo
    # We modify the Config class attributes directly to affect all modules
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = train_subset_path
    Config.VAL_METADATA_PATH = val_subset_path
    Config.TEST_METADATA_PATH = test_subset_path

    # Update cache paths to avoid using existing full-dataset caches
    Config.CACHED_TRAIN_DF_PATH = os.path.join(demo_dir, "cached_train_df.parquet")
    Config.CACHED_VAL_DF_PATH = os.path.join(demo_dir, "cached_val_df.parquet")
    Config.CACHED_TEST_DF_PATH = os.path.join(demo_dir, "cached_test_df.parquet")

    # Update Output Paths
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(demo_dir, "submission.csv")

    # Training Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Smaller batch size for demo


def verify_dataset_logic():
    """
    Verifies that the dataset loads images and targets correctly.
    """
    print("\n--- Verifying Dataset Logic ---")
    # Initialize dataset with the subset (load_cached_data=False to force read from new CSV)
    dataset = ChestXRayDataset(split="train", load_cached_data=False)

    # Check length
    assert len(dataset) == 32, f"Expected 32 samples, got {len(dataset)}"

    # Fetch one sample
    img, target, image_id = dataset[0]

    # Verify Image
    assert isinstance(img, torch.Tensor), "Image should be a Tensor"
    assert img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected shape (3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img.shape}"

    # Verify Target
    assert isinstance(target, dict), "Target should be a dictionary"
    assert "boxes" in target, "Target must contain 'boxes'"
    assert "labels" in target, "Target must contain 'labels'"
    assert isinstance(target["boxes"], torch.Tensor), "Boxes must be a Tensor"
    assert isinstance(target["labels"], torch.Tensor), "Labels must be a Tensor"

    print("Dataset verification passed.")
    return img, target


def verify_model_logic(sample_img, sample_target):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model Logic ---")
    model = get_one_stage_detector()
    model.to(Config.DEVICE)
    model.train()

    # Prepare batch
    images = torch.stack([sample_img, sample_img]).to(Config.DEVICE)
    targets = [
        {k: v.to(Config.DEVICE) for k, v in sample_target.items()},
        {k: v.to(Config.DEVICE) for k, v in sample_target.items()},
    ]

    # Forward pass (Training mode returns loss dict)
    loss_dict = model(images, targets)

    assert isinstance(
        loss_dict, dict
    ), "Model in train mode should return a loss dictionary"
    assert "classification" in loss_dict, "Loss dict should contain classification loss"
    assert (
        "bbox_regression" in loss_dict
    ), "Loss dict should contain bbox regression loss"

    # Forward pass (Eval mode returns detections)
    model.eval()
    with torch.no_grad():
        detections = model(images)

    assert isinstance(
        detections, list
    ), "Model in eval mode should return a list of detections"
    assert len(detections) == 2, "Should return detections for each image in batch"
    assert "boxes" in detections[0], "Detections should contain boxes"
    assert "scores" in detections[0], "Detections should contain scores"

    print("Model verification passed.")


def run_training_pipeline():
    """
    Runs the training loop using the engine.
    """
    print("\n--- Running Training Pipeline (1 Epoch) ---")

    # Run fit (uses the patched Config paths)
    fit(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force reload of subsets
        patience=1,
    )

    # Verify output
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"
    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")


def run_inference_pipeline():
    """
    Runs inference and generates submission.
    """
    print("\n--- Running Inference Pipeline ---")

    predict_and_submit(load_cached_data=False)

    # Verify submission
    assert os.path.exists(
        Config.SUBMISSION_FILE_PATH
    ), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Verify columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        "PredictionString" in df_sub.columns
    ), "Submission missing 'PredictionString' column"

    # Verify content format (sample check)
    sample_id = df_sub.iloc[0]["id"]
    sample_pred = df_sub.iloc[0]["PredictionString"]

    assert isinstance(sample_id, str), "ID should be string"
    assert isinstance(sample_pred, str), "PredictionString should be string"

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Initialization
    seed_everything(42)
    setup_demo_environment()

    # 2. Component Verification
    sample_img, sample_target = verify_dataset_logic()
    verify_model_logic(sample_img, sample_target)

    # 3. Pipeline Execution
    run_training_pipeline()
    run_inference_pipeline()

    print("\nAll demonstrations completed successfully.")
