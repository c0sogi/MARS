import sys
import os
import torch
import numpy as np
import pandas as pd

# Append current directory to path to ensure library imports work
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.dataset import get_datasets
from library.models import WhaleClassifier
from library.train import run_training_pipeline

if __name__ == "__main__":
    # --- 1. Configuration Patching for Fast Demonstration ---
    print(">>> Configuring pipeline for fast demonstration...")

    # Enable DEBUG mode to process only 100 samples per dataset
    Config.DEBUG = True

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for small debug dataset

    # Use only one model from the ensemble to save time
    Config.MODEL_ARCHS = ["tf_efficientnet_b2.ns_jft_in1k"]

    # Set specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the demo directory
    Config.TRAIN_DATA_CACHE = os.path.join(Config.WORKING_DIR, "train_data_debug.npy")
    Config.TRAIN_LABELS_CACHE = os.path.join(
        Config.WORKING_DIR, "train_labels_debug.npy"
    )
    Config.VAL_DATA_CACHE = os.path.join(Config.WORKING_DIR, "val_data_debug.npy")
    Config.VAL_LABELS_CACHE = os.path.join(Config.WORKING_DIR, "val_labels_debug.npy")
    Config.TEST_DATA_CACHE = os.path.join(Config.WORKING_DIR, "test_data_debug.npy")
    Config.TEST_CLIPS_CACHE = os.path.join(Config.WORKING_DIR, "test_clips_debug.npy")

    # Create the directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Architecture: {Config.MODEL_ARCHS}")
    print(f"Debug Mode: {Config.DEBUG}")

    # --- 2. Verify Utilities ---
    print("\n>>> Verifying Utilities...")
    seed_everything(Config.SEED)

    # Test AUC calculation
    y_true_dummy = np.array([0, 1, 0, 1])
    y_pred_dummy = np.array([0.1, 0.9, 0.2, 0.8])
    auc_score = calculate_auc(y_true_dummy, y_pred_dummy)
    print(f"Dummy AUC Score: {auc_score}")
    assert auc_score == 1.0, "AUC calculation verification failed."

    # --- 3. Verify Dataset Pipeline ---
    print("\n>>> Verifying Dataset Pipeline (Preprocessing & Caching)...")
    # load_cached_data=False forces processing of the raw audio files
    # Since DEBUG=True, this will only process the first 100 files of each set
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")
    print(f"Test Dataset Size: {len(test_ds)}")

    # Assertions to ensure DEBUG mode worked (sizes should be <= 100)
    assert len(train_ds) <= 100, "Train dataset size exceeds debug limit."
    assert len(val_ds) <= 100, "Val dataset size exceeds debug limit."
    assert len(test_ds) <= 100, "Test dataset size exceeds debug limit."

    # Check data shapes
    sample_img, sample_label = train_ds[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Label: {sample_label}")

    # Expecting (1, 224, 224) because Config.IN_CHANNELS is implicitly 1 via preprocessing
    assert sample_img.shape == (
        1,
        224,
        224,
    ), f"Unexpected image shape: {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label is not a Tensor."

    # --- 4. Verify Model Architecture ---
    print("\n>>> Verifying Model Architecture...")
    model = WhaleClassifier(model_name=Config.MODEL_ARCHS[0], pretrained=False)
    model.eval()

    # Check if the first layer was correctly modified to accept 1 channel
    # For EfficientNet, the first layer is usually 'conv_stem'
    first_layer = model.backbone.conv_stem
    print(f"First Layer In-Channels: {first_layer.in_channels}")
    assert (
        first_layer.in_channels == 1
    ), "Model first layer was not adapted to 1 channel."

    # Check forward pass dimensions
    dummy_input = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    # Expecting (Batch_Size, 1)
    assert output.shape == (2, 1), f"Unexpected output shape: {output.shape}"

    # --- 5. Run Training Pipeline ---
    print("\n>>> Running Full Training Pipeline...")
    # This will train the model, run validation, perform inference on test, and save submission
    # It uses the datasets we already cached in step 3.
    run_training_pipeline()

    # --- 6. Verify Submission ---
    print("\n>>> Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    # Check columns
    assert "clip" in submission_df.columns, "Submission missing 'clip' column."
    assert (
        "probability" in submission_df.columns
    ), "Submission missing 'probability' column."

    # Check length matches test dataset
    assert len(submission_df) == len(test_ds), "Submission row count mismatch."

    print("\n>>> Demonstration Completed Successfully.")
