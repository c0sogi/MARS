import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config, set_seed
from library.utils import calculate_auc
from library.dataset import get_dataloaders, WhaleDataset
from library.model import CRNN
from library.trainer import run_training, predict_and_submit


def verify_components():
    print("=== Starting Component Verification ===")

    # 1. Setup Configuration for Speed
    print("\n[1] Configuring for fast execution...")
    # Override Config for a quick debug run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small batch for testing
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small test

    # Ensure working directory is clean for this run to test processing logic
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Configuration: Debug={Config.DEBUG}, Device={device}")

    # 2. Verify Utils
    print("\n[2] Verifying Utils...")
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    auc = calculate_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")
    assert auc == 1.0, "AUC calculation failed for perfect predictions"

    y_true_single = [0, 0, 0]
    auc_single = calculate_auc(y_true_single, y_pred[:3])
    assert auc_single == 0.5, "AUC should be 0.5 for single-class ground truth"
    print("Utils verification passed.")

    # 3. Verify Data Loading & Processing
    print("\n[3] Verifying Data Loading...")
    # Force load_cached_data=False to test the audio processing pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    inputs, labels = next(iter(train_loader))
    print(f"Input Batch Shape: {inputs.shape}")
    print(f"Labels Batch Shape: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Freq, Time)
    # Freq = 128 (N_MELS), Time depends on duration/hop_length
    assert inputs.dim() == 4, "Input must be 4D tensor (B, C, F, T)"
    assert inputs.shape[1] == 1, "Input channel must be 1"
    assert inputs.shape[2] == Config.N_MELS, f"Freq dim must be {Config.N_MELS}"
    assert labels.dim() == 1, "Labels must be 1D tensor"
    print("Data loading verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model...")
    model = CRNN().to(device)

    # Forward pass
    inputs = inputs.to(device)
    logits = model(inputs)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (inputs.size(0), 1), "Output shape must be (B, 1)"
    print("Model verification passed.")

    # 5. Verify Training Loop
    print("\n[5] Verifying Training Loop...")
    # run_training returns the model and the test_loader
    trained_model, _ = run_training(
        epochs=Config.EPOCHS, load_cached_data=True, debug=True
    )

    # Check if model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Training loop verification passed.")

    # 6. Verify Submission Generation
    print("\n[6] Verifying Submission...")
    predict_and_submit(trained_model, test_loader, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head()}")

    assert list(df_sub.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch"
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"
    assert df_sub["probability"].dtype == float, "Probability column must be float"

    print("Submission verification passed.")
    print("\n=== All Systems Go ===")


if __name__ == "__main__":
    try:
        verify_components()
    except AssertionError as e:
        print(f"\nCRITICAL FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
