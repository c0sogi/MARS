import os
import torch
import pandas as pd
import numpy as np
import random
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import SpeechCommandDataset, get_dataloaders
from library.model import SimpleConvNet
from library.trainer import train
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_demonstration():
    print("=== Starting Speech Command Recognition Pipeline Demonstration ===")

    # 1. Optimize Configuration for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SUBSET_SIZE = 200  # Use only 200 samples for train/val
    Config.BATCH_SIZE = 16  # Smaller batch size for the demo

    # Ensure working directories exist
    Config.setup()
    set_seed(Config.SEED)

    # 2. Verify Dataset and DataLoader
    print("\n[2] Verifying Dataset and DataLoader...")

    # Load metadata df for manual check
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Instantiate dataset directly to check single item
    dataset = SpeechCommandDataset(df_train.head(10), Config.INPUT_ROOT, mode="train")
    sample_input, sample_label = dataset[0]

    print(f"   Input Tensor Shape: {sample_input.shape}")
    print(f"   Label: {sample_label} ({Config.IDX2LABEL.get(sample_label, 'Unknown')})")

    # Expected shape: (1, n_mels, time_steps)
    # n_mels = 64 (from Config)
    # time_steps = approx 32 (16000 sr * 1s / 512 hop) -> 31 or 32 depending on padding
    assert (
        sample_input.dim() == 3
    ), "Input should be 3-dimensional (Channel, Freq, Time)"
    assert sample_input.shape[0] == 1, "Input channel should be 1"
    assert (
        sample_input.shape[1] == Config.N_MELS
    ), f"Input Mel bands should be {Config.N_MELS}"

    # Get DataLoaders using the utility function
    train_loader, val_loader = get_dataloaders(debug=True)

    # Fetch one batch
    batch_inputs, batch_labels = next(iter(train_loader))
    print(f"   Batch Input Shape: {batch_inputs.shape}")
    print(f"   Batch Label Shape: {batch_labels.shape}")

    assert batch_inputs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_inputs.shape[1] == 1, "Channel dimension mismatch"
    assert batch_labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("   Dataset and DataLoader verification successful.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = SimpleConvNet(num_classes=Config.NUM_CLASSES).to(device)

    # Move batch to device
    batch_inputs = batch_inputs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch_inputs)

    print(f"   Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    print("   Model verification successful.")

    # 4. Run Training Loop
    print("\n[4] Running Training Loop (1 Epoch, Debug Subset)...")
    # The train function handles the loop, validation, and saving the best model
    train(debug=True, epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Model checkpoint was not created at {Config.MODEL_CHECKPOINT_PATH}"
    print(f"   Training completed. Checkpoint saved at {Config.MODEL_CHECKPOINT_PATH}")

    # 5. Run Inference and Generate Submission
    print("\n[5] Running Inference and Generating Submission...")
    generate_submission()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file was not created at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission file loaded. Shape: {df_sub.shape}")
    print(f"   Columns: {list(df_sub.columns)}")

    # Basic validation of submission content
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if labels are valid
    unique_preds = df_sub["label"].unique()
    invalid_labels = [l for l in unique_preds if l not in Config.LABELS]
    if invalid_labels:
        print(f"   Warning: Found invalid labels in submission: {invalid_labels}")
    else:
        print("   All predicted labels are valid.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
