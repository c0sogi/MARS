import os
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import read_dicom_windowed, weighted_log_loss
from library.dataset import prepare_training_data, FractureSliceDataset, get_transforms
from library.model import FractureClassifier
from library.train import run_training
from library.inference import run_inference


def run_demo():
    print("=== Starting Fracture Detection Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Configure for Speed (Debug Mode)
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config defaults to ensure the script runs quickly
    Config.DEBUG = True
    Config.DEBUG_DATASET_SIZE = 32  # Small subset for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.WORKING_DIR = "./working/demo_run"

    # Re-run setup to create the specific working directory
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Weighted Log Loss
    y_true = np.array([[0, 1, 0, 0, 0, 0, 0, 1]])  # Example label
    y_pred = np.array([[0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]])  # Good prediction
    loss = weighted_log_loss(y_true, y_pred)
    print(f"Calculated Weighted Log Loss: {loss:.4f}")
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss cannot be negative"

    # Test DICOM Reading
    # We need to find a real DICOM file from the metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if not train_meta.empty:
        sample_row = train_meta.iloc[0]
        sample_dir = os.path.join(Config.INPUT_DIR, sample_row["image_path"])

        # Find a dcm file
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.endswith(".dcm")]
            if files:
                dcm_path = os.path.join(sample_dir, files[0])
                img = read_dicom_windowed(
                    dcm_path, Config.WINDOW_CENTER, Config.WINDOW_WIDTH
                )

                print(f"Loaded DICOM Image Shape: {img.shape}")
                print(f"Pixel Value Range: [{img.min():.2f}, {img.max():.2f}]")

                assert img.ndim == 2, "read_dicom_windowed should return a 2D array"
                assert (
                    0.0 <= img.min() and img.max() <= 1.0 + 1e-6
                ), "Image should be normalized to [0, 1]"
            else:
                print("No .dcm files found in sample directory.")
        else:
            print(f"Sample directory {sample_dir} does not exist.")

    # ---------------------------------------------------------
    # 3. Verify Dataset Pipeline
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset Pipeline...")

    # Generate/Load Training Data (Debug mode will limit size)
    # We force load_cached_data=False to ensure we test the generation logic
    df_train = prepare_training_data(load_cached_data=False)

    assert not df_train.empty, "Training dataframe is empty"
    assert "StudyInstanceUID" in df_train.columns
    assert "patient_overall" in df_train.columns

    print(f"Training Dataframe Shape: {df_train.shape}")

    # Instantiate Dataset
    train_dataset = FractureSliceDataset(
        df_train,
        Config.TRAIN_IMAGES_DIR,
        transform=get_transforms("train"),
        is_test=False,
    )

    # Check item retrieval
    if len(train_dataset) > 0:
        img_tensor, label_tensor = train_dataset[0]
        print(f"Dataset Sample Tensor Shape: {img_tensor.shape}")
        print(f"Dataset Sample Label: {label_tensor}")

        # Expected shape: (Channels=3, Height=256, Width=256)
        assert img_tensor.shape == (
            3,
            Config.IMG_SIZE[0],
            Config.IMG_SIZE[1],
        ), f"Expected shape (3, {Config.IMG_SIZE[0]}, {Config.IMG_SIZE[1]}), got {img_tensor.shape}"
        assert label_tensor.shape == (
            Config.NUM_CLASSES,
        ), f"Expected label shape ({Config.NUM_CLASSES},), got {label_tensor.shape}"

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = FractureClassifier(pretrained=False)  # No need to download weights for demo
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    assert torch.all(output >= 0) and torch.all(
        output <= 1
    ), "Outputs should be probabilities [0, 1]"

    # ---------------------------------------------------------
    # 5. Execute Training Loop
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch, Debug Subset)...")

    # This function handles data loading, model init, training, and saving best_model.pth
    run_training()

    expected_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(expected_model_path), "Training failed to save best_model.pth"
    print("Training completed successfully.")

    # ---------------------------------------------------------
    # 6. Execute Inference Pipeline
    # ---------------------------------------------------------
    print("\n[6] Executing Inference Pipeline...")

    # This function loads the saved model and generates submission.csv
    # We force load_cached_data=False to verify test data generation
    run_inference(load_cached_data=False)

    expected_sub_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        expected_sub_path
    ), "Inference failed to generate submission.csv"

    # Verify Submission Content
    df_sub = pd.read_csv(expected_sub_path)
    print(f"Submission File Shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    assert (
        "row_id" in df_sub.columns and "fractured" in df_sub.columns
    ), "Submission file missing required columns"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure warnings are suppressed as per instructions
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
