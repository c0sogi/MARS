import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library files
from library.config import (
    seed_everything,
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    YOLO_DATASET_DIR,
    IMG_SIZE,
)
from library.dicom_utils import process_dicom_image
from library.data_setup import prepare_yolo_data
from library.train_engine import train_model
from library.inference import generate_submission


def run_demonstration():
    print("Starting demonstration of library modules...")

    # 1. Set Seed
    seed_everything(42)
    print("Seed set for reproducibility.")

    # ==========================================
    # DEMO: DICOM UTILS
    # ==========================================
    print("\n--- Demonstrating DICOM Processing ---")
    # Pick a random training image from metadata to test processing
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    df_train = pd.read_csv(train_meta_path)

    # Get the first file path
    sample_rel_path = df_train.iloc[0]["file_path"]
    sample_dicom_path = os.path.join(INPUT_DIR, sample_rel_path)

    print(f"Processing sample DICOM: {sample_dicom_path}")

    # Process image
    processed_img = process_dicom_image(sample_dicom_path, target_size=IMG_SIZE)

    # Verification
    assert isinstance(
        processed_img, np.ndarray
    ), "Processed image should be a numpy array"
    assert processed_img.shape == (
        IMG_SIZE,
        IMG_SIZE,
    ), f"Expected shape ({IMG_SIZE}, {IMG_SIZE}), got {processed_img.shape}"
    assert (
        processed_img.dtype == np.uint8
    ), f"Expected dtype uint8, got {processed_img.dtype}"
    print("DICOM processing verification successful.")

    # ==========================================
    # DEMO: DATA SETUP
    # ==========================================
    print("\n--- Demonstrating Data Preparation (YOLO Format) ---")
    # Generate a tiny dataset (e.g., 20 images) for debugging/speed
    SAMPLE_SIZE = 20

    # Force regeneration by setting load_cached_data=False initially
    df_yolo = prepare_yolo_data(sample_size=SAMPLE_SIZE, load_cached_data=False)

    # Verification
    assert os.path.exists(YOLO_DATASET_DIR), "YOLO dataset directory not created"
    yaml_path = os.path.join(YOLO_DATASET_DIR, "data.yaml")
    assert os.path.exists(yaml_path), "data.yaml not found"

    # Check if images were actually saved
    train_imgs_dir = os.path.join(YOLO_DATASET_DIR, "images", "train")
    saved_images = os.listdir(train_imgs_dir)
    print(f"Number of training images generated: {len(saved_images)}")
    assert len(saved_images) > 0, "No training images found in YOLO dataset"

    print("Data preparation verification successful.")

    # ==========================================
    # DEMO: TRAINING ENGINE
    # ==========================================
    print("\n--- Demonstrating Model Training ---")
    # Train for just 1 epoch with small image size for speed
    # We use load_cached_data=True to use the data we just generated

    try:
        weights_path = train_model(
            epochs=1,
            batch_size=4,
            img_size=320,  # Smaller size for faster training demo
            debug_sample_size=SAMPLE_SIZE,
            load_cached_data=True,
        )

        # Verification
        print(f"Training finished. Weights saved at: {weights_path}")
        assert os.path.exists(weights_path), "Weights file does not exist"

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # ==========================================
    # DEMO: INFERENCE
    # ==========================================
    print("\n--- Demonstrating Inference ---")

    # Create a mini test set metadata file to avoid running inference on all 1500 images
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")
    df_test_full = pd.read_csv(test_meta_path)

    # Take top 5 images
    df_test_mini = df_test_full.head(5)
    mini_test_path = os.path.join(WORKING_DIR, "mini_test.csv")
    df_test_mini.to_csv(mini_test_path, index=False)

    output_sub_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Run inference
    generate_submission(
        weights_path=weights_path,
        test_metadata_path=mini_test_path,
        output_path=output_sub_path,
    )

    # Verification
    assert os.path.exists(output_sub_path), "Submission file was not created"

    df_sub = pd.read_csv(output_sub_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    assert len(df_sub) == 5, f"Expected 5 rows in submission, got {len(df_sub)}"
    assert "image_id" in df_sub.columns, "Missing 'image_id' column"
    assert "PredictionString" in df_sub.columns, "Missing 'PredictionString' column"

    # Check format of the first prediction string
    pred_string = df_sub.iloc[0]["PredictionString"]
    print(f"Sample PredictionString: {pred_string}")
    assert isinstance(pred_string, str), "PredictionString must be a string"
    assert len(pred_string) > 0, "PredictionString is empty"

    print("Inference verification successful.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demonstration()
