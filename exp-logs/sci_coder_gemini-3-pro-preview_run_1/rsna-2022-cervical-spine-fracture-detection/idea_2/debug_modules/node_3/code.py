import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
import warnings
import timm
import torchvision

# Import provided library modules
from library.config import Config
from library.utils import read_dicom, competition_loss, seed_everything
from library.models import SpineLocalizer, FractureClassifier
from library.segmentation_engine import train_localizer, generate_spine_coordinates
from library.classification_engine import train_classifier, inference_and_submission
from library.dataset import (
    process_slice_metadata,
    SegmentationDataset,
    FractureCropDataset,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Configures the Config class for a quick demo run and prepares
    a subset of data guaranteed to work with the pipeline.
    """
    print(">>> Setting up Demo Environment...")

    # 1. Override Paths and Settings for Demo
    Config.WORKING_DIR = "./working/demo_run"

    # Clean up previous run to avoid stale cache issues (Cite debug_lesson_1)
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set Speed/Debug Parameters
    Config.DEBUG = True
    Config.DEBUG_DATASET_SIZE = 10  # Use only 10 studies
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script stability
    Config.SEED = 42

    seed_everything(Config.SEED)

    # 2. Prepare Valid Metadata Subset
    # We need to ensure the subset we pick for training actually has bounding boxes,
    # otherwise SegmentationDataset will be empty and training will crash.

    print(">>> Preparing Data Subset...")
    original_train = pd.read_csv(os.path.join("./metadata", "train_metadata.csv"))
    bbox_df = pd.read_csv(Config.TRAIN_BBOX_PATH)

    # Get UIDs that have bounding boxes
    bbox_uids = set(bbox_df["StudyInstanceUID"].unique())

    # Filter metadata for UIDs that exist in bbox file
    valid_train = original_train[original_train["StudyInstanceUID"].isin(bbox_uids)]

    # Select a small subset (e.g., 10)
    if len(valid_train) > Config.DEBUG_DATASET_SIZE:
        demo_train = valid_train.head(Config.DEBUG_DATASET_SIZE).copy()
    else:
        demo_train = valid_train.copy()

    # Save this subset to working dir and update Config to point to it
    demo_meta_path = os.path.join(Config.WORKING_DIR, "demo_train_metadata.csv")
    demo_train.to_csv(demo_meta_path, index=False)
    Config.TRAIN_METADATA_PATH = demo_meta_path

    # Use the same file for validation for this demo
    Config.VAL_METADATA_PATH = demo_meta_path

    print(
        f"    Created demo metadata with {len(demo_train)} studies at {demo_meta_path}"
    )


def monkey_patch_models():
    """
    Disables downloading of pretrained weights to ensure offline execution.
    """
    print(">>> Monkey-patching models for offline execution...")

    # Patch timm.create_model used in FractureClassifier
    original_create_model = timm.create_model

    def mocked_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = mocked_create_model

    # Patch torchvision.models.resnet18 used in SpineLocalizer
    # Note: SpineLocalizer calls models.resnet18(pretrained=pretrained)
    # We can also control this via Config.PREDICTION_DIR hack, but patching is cleaner.
    original_resnet18 = torchvision.models.resnet18

    def mocked_resnet18(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_resnet18(*args, **kwargs)

    torchvision.models.resnet18 = mocked_resnet18


def test_utilities():
    """
    Validates basic utility functions.
    """
    print("\n>>> Testing Utilities...")

    # 1. Test read_dicom
    # Get a valid image path from our demo metadata
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    row = df.iloc[0]
    img_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
    # Just pick the first file in the directory
    files = [f for f in os.listdir(img_dir) if f.endswith(".dcm")]
    if files:
        img_path = os.path.join(img_dir, files[0])
        img = read_dicom(img_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH)

        if not isinstance(img, np.ndarray):
            raise AssertionError("read_dicom returned wrong type")
        if img.shape != (Config.ORIGINAL_IMAGE_SIZE, Config.ORIGINAL_IMAGE_SIZE):
            raise AssertionError(f"read_dicom returned wrong shape: {img.shape}")
        print("    read_dicom: Passed")
    else:
        print("    read_dicom: Skipped (no dcm files found in sample dir)")

    # 2. Test competition_loss
    y_pred = torch.tensor([[0.5] * 8, [0.1] * 8])
    y_true = torch.tensor([[1.0] * 8, [0.0] * 8])
    loss = competition_loss(y_pred, y_true)
    if not torch.is_tensor(loss) or loss.item() <= 0:
        raise AssertionError("competition_loss failed basic check")
    print("    competition_loss: Passed")


def run_segmentation_stage():
    """
    Demonstrates the Spine Localizer training and inference.
    """
    print("\n>>> Running Stage 1: Segmentation (Spine Localizer)...")

    # 1. Train Localizer
    # We set Config.PREDICTION_DIR to False temporarily because the provided code
    # passes this path as the 'pretrained' argument to SpineLocalizer.
    # Passing a boolean False prevents download attempts.
    original_pred_dir = Config.PREDICTION_DIR
    Config.PREDICTION_DIR = False

    try:
        train_localizer(
            num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, debug=True
        )
    finally:
        Config.PREDICTION_DIR = original_pred_dir

    # Verify Checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "spine_localizer.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Spine Localizer checkpoint was not created!")
    print("    Localizer training complete. Checkpoint saved.")

    # 2. Generate Coordinates
    # This uses the trained localizer to find spine centers for the classification stage.
    # We force load_cached_data=False to verify the inference logic.
    print("    Generating spine coordinates...")
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    coords = generate_spine_coordinates(
        df, mode="train", load_cached_data=False, batch_size=Config.BATCH_SIZE
    )

    if len(coords) == 0:
        raise AssertionError("No spine coordinates were generated!")
    print(f"    Generated coordinates for {len(coords)} slices.")


def run_classification_stage():
    """
    Demonstrates the Fracture Classifier training.
    """
    print("\n>>> Running Stage 2: Classification (Fracture Classifier)...")

    # Train Classifier
    # This function internally calls generate_spine_coordinates.
    # We use the demo metadata and settings configured earlier.
    train_classifier(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=True,
        load_cached_data=True,  # Can use cache from previous step if available
    )

    # Verify Checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "fracture_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Fracture Classifier checkpoint was not created!")
    print("    Classifier training complete. Checkpoint saved.")


def run_inference_stage():
    """
    Demonstrates inference on the test set and submission generation.
    """
    print("\n>>> Running Inference & Submission...")

    # Run inference
    # Note: Config.TEST_METADATA_PATH points to the full test set.
    # For speed, we might want to limit this, but the function doesn't accept a limit arg easily.
    # However, Config.DEBUG=True might limit it if the function respects it?
    # Looking at library code: inference_and_submission loads test_meta but doesn't slice it based on DEBUG.
    # To ensure speed, let's temporarily point TEST_METADATA_PATH to our small demo file
    # (pretending our train data is test data for the sake of the API demo).

    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = Config.TRAIN_METADATA_PATH  # Use small demo set

    try:
        inference_and_submission(batch_size=Config.BATCH_SIZE, load_cached_data=False)
    finally:
        Config.TEST_METADATA_PATH = original_test_path

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created!")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(sub_df)} rows.")
    print("    Sample rows:")
    print(sub_df.head())


if __name__ == "__main__":
    try:
        setup_demo_environment()
        monkey_patch_models()
        test_utilities()
        run_segmentation_stage()
        run_classification_stage()
        run_inference_stage()
        print("\n>>> DEMO COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print(f"\n>>> DEMO FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
