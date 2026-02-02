import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
import shutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders, prepare_data
from library.model import AttentivePyramidSiamese
from library.train import run_training
from library.predict import generate_submission


def setup_synthetic_environment():
    """
    Creates a synthetic environment with dummy data to demonstrate the pipeline
    without relying on the large/complex original dataset.
    """
    print("\n=== Setting up Synthetic Environment ===")

    # Define paths for synthetic data
    base_dir = "./working/demo_env"
    input_dir = os.path.join(base_dir, "input")
    metadata_dir = os.path.join(base_dir, "metadata")
    working_dir = os.path.join(base_dir, "working")

    # Clean up previous run if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)

    # Override Config to use these paths and speed up execution
    Config.override(
        PROJECT_NAME="demo_project",
        INPUT_DIR=input_dir,
        METADATA_DIR=metadata_dir,
        WORKING_DIR=working_dir,
        # Point metadata paths to our new synthetic CSVs
        TRAIN_METADATA_PATH=os.path.join(metadata_dir, "train.csv"),
        VAL_METADATA_PATH=os.path.join(metadata_dir, "val.csv"),
        TEST_METADATA_PATH=os.path.join(metadata_dir, "test.csv"),
        SAMPLE_SUBMISSION_PATH=os.path.join(input_dir, "sample_submission.csv"),
        # Update cache paths
        TRAIN_CACHE_PATH=os.path.join(working_dir, "processed_train.parquet"),
        VAL_CACHE_PATH=os.path.join(working_dir, "processed_val.parquet"),
        TEST_CACHE_PATH=os.path.join(working_dir, "processed_test.parquet"),
        CHECKPOINT_PATH=os.path.join(working_dir, "best_model.pth"),
        # Speed optimizations
        IMG_SIZE=(128, 128),  # Small images
        BATCH_SIZE=4,
        NUM_EPOCHS=1,  # Single epoch
        DEBUG=True,
    )

    # Generate Synthetic Images and Metadata
    # We generate pairs (L/R) to satisfy the contralateral logic

    def create_data(phase, num_patients):
        data = []
        img_dir_name = "train_images" if phase in ["train", "val"] else "test_images"
        full_img_dir = os.path.join(input_dir, img_dir_name)

        for i in range(num_patients):
            patient_id = 10000 + i
            # Create patient directory
            pat_dir = os.path.join(full_img_dir, str(patient_id))
            os.makedirs(pat_dir, exist_ok=True)

            # Generate Left and Right views
            for lat in ["L", "R"]:
                image_id = np.random.randint(1e8, 9e8)
                filename = f"{image_id}.dcm"

                # Create dummy image (Save as PNG content in .dcm file for cv2 compatibility)
                # Random noise
                img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                cv2.imwrite(os.path.join(pat_dir, filename), img)

                rel_path = f"{img_dir_name}/{patient_id}/{filename}"

                row = {
                    "patient_id": patient_id,
                    "image_id": image_id,
                    "file_path": rel_path,
                    "laterality": lat,
                    "view": "CC",  # Simplified to one view
                    "age": 50.0 + np.random.randn() * 10,
                    "implant": np.random.choice([0, 1]),
                    "site_id": 1,
                    "machine_id": 10,
                }

                if phase in ["train", "val"]:
                    # Assign random target
                    row["cancer"] = np.random.choice([0, 1], p=[0.8, 0.2])
                    # Add dummy auxiliary cols
                    row["density"] = "B"
                    row["biopsy"] = 0
                    row["invasive"] = 0
                    row["BIRADS"] = 1
                    row["difficult_negative_case"] = False
                else:
                    # Test needs prediction_id
                    row["prediction_id"] = f"{patient_id}_{lat}"

                data.append(row)

        return pd.DataFrame(data)

    # Create Train (10 patients -> 20 images)
    df_train = create_data("train", 10)
    df_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)

    # Create Val (5 patients -> 10 images)
    df_val = create_data("val", 5)
    df_val.to_csv(Config.VAL_METADATA_PATH, index=False)

    # Create Test (5 patients -> 10 images)
    df_test = create_data("test", 5)
    df_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    # Create Sample Submission
    sample_sub = df_test[["prediction_id"]].copy()
    sample_sub["cancer"] = 0.5
    sample_sub.to_csv(Config.SAMPLE_SUBMISSION_PATH, index=False)

    print("Synthetic data generated successfully.")


def verify_metric():
    """
    Verifies the Probabilistic F1 score calculation.
    """
    print("\n=== Verifying Metric Logic ===")

    # Case 1: Perfect prediction
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1.0, 0.0, 1.0, 0.0])
    score = probabilistic_f1(y_true, y_pred)
    print(f"Perfect Score: {score}")
    assert abs(score - 1.0) < 1e-5, "Metric failed on perfect predictions"

    # Case 2: All zeros prediction
    y_pred_zero = np.array([0.0, 0.0, 0.0, 0.0])
    score_zero = probabilistic_f1(y_true, y_pred_zero)
    print(f"Zero Score: {score_zero}")
    assert score_zero == 0.0, "Metric failed on zero predictions"

    # Case 3: Mixed probabilities
    y_pred_mixed = np.array([0.8, 0.2, 0.6, 0.4])
    # TP = 0.8*1 + 0.2*0 + 0.6*1 + 0.4*0 = 1.4
    # FP = 0.8*0 + 0.2*1 + 0.6*0 + 0.4*1 = 0.6
    # Precision = 1.4 / (1.4 + 0.6) = 0.7
    # Recall = 1.4 / (1 + 1) = 0.7
    # F1 = 0.7
    score_mixed = probabilistic_f1(y_true, y_pred_mixed)
    print(f"Mixed Score: {score_mixed}")
    assert (
        abs(score_mixed - 0.7) < 1e-5
    ), f"Metric calculation mismatch. Got {score_mixed}, expected 0.7"

    print("Metric verification passed.")


def verify_model_architecture():
    """
    Verifies the model instantiation and forward pass.
    """
    print("\n=== Verifying Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for quick check
    model = AttentivePyramidSiamese(
        pretrained=False
    )  # Skip downloading weights for speed
    model.to(device)
    model.eval()

    # Create dummy batch: (Batch, Channels, Height, Width)
    # Channels = 3 (Image + Age + Implant)
    B, C, H, W = 2, 3, 128, 128
    dummy_target = torch.randn(B, C, H, W).to(device)
    dummy_contra = torch.randn(B, C, H, W).to(device)

    with torch.no_grad():
        output = model(dummy_target, dummy_contra)

    print(f"Output Shape: {output.shape}")

    assert output.shape == (B, 1), f"Expected output shape {(B, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model architecture verification passed.")


def run_pipeline_demo():
    """
    Runs the full training and inference pipeline using the synthetic environment.
    """
    print("\n=== Running Full Pipeline Demo ===")

    # 1. Training
    print("--- Starting Training ---")
    # load_cached_data=False ensures we process the new synthetic data
    best_pf1 = run_training(load_cached_data=False)

    # Verify checkpoint creation
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created!"
    print(f"Training finished. Best pF1: {best_pf1}")

    # 2. Inference
    print("\n--- Starting Inference ---")
    generate_submission(load_cached_data=False)

    # Verify submission creation
    submission_path = "submission.csv"  # Default output of generate_submission
    assert os.path.exists(submission_path), "Submission file was not created!"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    assert (
        "prediction_id" in df_sub.columns and "cancer" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Pipeline demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    try:
        # 1. Setup Data
        setup_synthetic_environment()

        # 2. Verify Logic
        verify_metric()
        verify_model_architecture()

        # 3. Execute Pipeline
        run_pipeline_demo()

        print("\nAll demonstrations passed successfully!")

    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
