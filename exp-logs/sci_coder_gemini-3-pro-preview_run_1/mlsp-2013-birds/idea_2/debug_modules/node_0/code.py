import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import the library correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, calculate_multilabel_auc
from library.dataset import BirdDataset, get_transforms
from library.model import get_model
from library.trainer import run_training


def main():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Configuration Overrides for Speed and Offline Execution
    # We modify the Config class attributes directly to control the execution environment.
    print("1. Configuring environment for fast execution...")
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = (
        0  # Use 0 workers to avoid multiprocessing overhead on small data
    )
    Config.PRETRAINED = False  # Disable downloading weights for this demo
    Config.DEBUG = True

    # Set seed for reproducibility
    set_seed(42)

    # 2. Test Utility Functions
    print("2. Testing utility functions...")
    # Test AUC calculation with synthetic data
    # Create a scenario with 3 samples and 3 classes
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    y_pred_dummy = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.1], [0.1, 0.1, 0.9]])

    auc_score = calculate_multilabel_auc(y_true_dummy, y_pred_dummy)

    # Assertions
    assert isinstance(auc_score, float), "AUC score should be a float"
    assert 0.0 <= auc_score <= 1.0, "AUC score must be between 0 and 1"
    print(f"   AUC Calculation Test Passed. Score: {auc_score:.4f}")

    # 3. Test Dataset Loading
    print("3. Testing Dataset and Transforms...")
    # Initialize dataset with a limit on samples
    train_dataset = BirdDataset(
        metadata_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms(mode="train"),
        max_samples=10,
    )

    assert len(train_dataset) == 10, f"Expected 10 samples, got {len(train_dataset)}"

    # Fetch a single item
    image, label, rec_id = train_dataset[0]

    # Verify Image Shape: (Channels, Height, Width)
    # Config defaults: IMG_HEIGHT=256, IMG_WIDTH=512, IN_CHANNELS=3 (RGB from Grayscale)
    expected_shape = (3, 256, 512)
    assert (
        image.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {image.shape}"
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"

    # Verify Label Shape: (Num_Classes,) -> 19 classes
    assert label.shape == (
        19,
    ), f"Label shape mismatch. Expected (19,), got {label.shape}"
    assert label.dtype == torch.float32, "Labels should be float32"

    print("   Dataset Loading Test Passed.")

    # 4. Test Model Initialization and Forward Pass
    print("4. Testing Model Architecture...")
    # Initialize model (pretrained=False was set in Config)
    model = get_model(pretrained=False)
    model.eval()

    # Create dummy input batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, 256, 512)

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    assert output.shape == (
        2,
        19,
    ), f"Model output shape mismatch. Expected (2, 19), got {output.shape}"
    print("   Model Architecture Test Passed.")

    # 5. Integration Test: Training Pipeline
    print("5. Running Training Pipeline Integration Test...")
    # run_training encapsulates the entire flow:
    # Data Loading -> Model Init -> Training Loop -> Validation -> Prediction -> Submission

    try:
        run_training(
            max_samples=10,  # Use very few samples
            epochs=1,  # Run only 1 epoch
            batch_size=4,  # Small batch size
            learning_rate=1e-3,
            patience=1,
            mixup_alpha=0.0,  # Disable mixup for simple logic check
        )
    except Exception as e:
        print(f"   Training pipeline failed with error: {e}")
        raise e

    # 6. Verify Submission Output
    print("6. Verifying Submission File...")
    submission_path = Config.PREDICTIONS_PATH

    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_submission = pd.read_csv(submission_path)

    # Check Columns
    assert "Id" in df_submission.columns, "Submission missing 'Id' column"
    assert (
        "Probability" in df_submission.columns
    ), "Submission missing 'Probability' column"

    # Check Row Count
    # We used max_samples=10 for the test set.
    # The submission format is 1 row per species per recording.
    # 10 recordings * 19 species = 190 rows.
    expected_rows = 10 * 19
    assert (
        len(df_submission) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_submission)}"

    # Check Probability Range
    probs = df_submission["Probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1"

    print("   Submission File Verification Passed.")
    print("\n--- All Tests Completed Successfully ---")


if __name__ == "__main__":
    main()
