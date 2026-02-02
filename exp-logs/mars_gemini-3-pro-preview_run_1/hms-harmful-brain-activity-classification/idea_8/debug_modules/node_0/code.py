import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, KLDivLossWithLogits
from library.data import EEGDataset, get_dataloaders
from library.model import ChronologicallyEmbeddedDualStream
from library.train import run_training


def main():
    print(">>> Initializing Demo Execution...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # -------------------------------------------------------------------------
    # We modify the Config class attributes directly to control the execution environment.
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Small subset for rapid execution
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1  # Single epoch for demonstration
    Config.NUM_WORKERS = 2  # Minimal workers to reduce overhead

    # Redirect outputs to a specific demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Re-run setup to create these new directories
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Dataset and Data Loading Logic
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Dataset...")

    # Load a small slice of metadata for verification
    train_metadata = pd.read_csv(Config.TRAIN_CSV).iloc[: Config.DEBUG_SUBSET_SIZE]
    dataset = EEGDataset(train_metadata, mode="train", config=Config)

    # Fetch a single sample
    eeg_tensor, spec_tensor, rel_tensor, target_tensor = dataset[0]

    print(f"Sample EEG Shape: {eeg_tensor.shape}")
    print(f"Sample Spec Shape: {spec_tensor.shape}")
    print(f"Sample Rel Shape: {rel_tensor.shape}")
    print(f"Sample Target: {target_tensor}")

    # Assertions to ensure data shapes are correct
    # EEG should be (Channels, Seq_Len) -> (20, 5000)
    assert eeg_tensor.shape == (
        Config.EEG_CHANNELS,
        Config.EEG_SEQ_LEN,
    ), f"EEG tensor shape mismatch. Expected {(Config.EEG_CHANNELS, Config.EEG_SEQ_LEN)}, got {eeg_tensor.shape}"

    # Spectrogram should be (3, Time, Freq) -> (3, 512, 512)
    # Note: The dataset repeats the single channel spec 3 times for the backbone
    assert spec_tensor.shape == (
        3,
        Config.SPEC_SIZE[0],
        Config.SPEC_SIZE[1],
    ), f"Spectrogram tensor shape mismatch. Expected {(3, Config.SPEC_SIZE[0], Config.SPEC_SIZE[1])}, got {spec_tensor.shape}"

    # Relative indices should be (Time,) -> (512,)
    assert rel_tensor.shape == (
        Config.SPEC_SIZE[0],
    ), f"Relative indices shape mismatch. Expected {(Config.SPEC_SIZE[0],)}, got {rel_tensor.shape}"

    print("Dataset verification successful.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture and Forward Pass
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChronologicallyEmbeddedDualStream(Config).to(device)
    model.eval()

    # Prepare a dummy batch
    batch_eeg = eeg_tensor.unsqueeze(0).to(device)
    batch_spec = spec_tensor.unsqueeze(0).to(device)
    batch_rel = rel_tensor.unsqueeze(0).to(device)
    batch_target = target_tensor.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(batch_eeg, batch_spec, batch_rel)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        1,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(1, Config.NUM_CLASSES)}, got {logits.shape}"

    # Verify Loss Calculation
    criterion = KLDivLossWithLogits()
    loss = criterion(logits, batch_target)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss calculation resulted in NaN."

    print("Model verification successful.")

    # -------------------------------------------------------------------------
    # 4. Execute Full Training Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Starting Training Pipeline via library.train.run_training...")

    # We pass the parameters explicitly to run_training, though it also reads from Config.
    # This runs training, validation, inference, and submission generation.
    run_training(
        debug=Config.DEBUG,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        patience=1,
        num_workers=Config.NUM_WORKERS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
        seed=Config.SEED,
    )

    print("Training pipeline finished.")

    # -------------------------------------------------------------------------
    # 5. Verify Submission Output
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {sub_df.shape}")
    print("First 5 rows:")
    print(sub_df.head())

    # Verify dimensions
    # In debug mode, the test set is also sliced to DEBUG_SUBSET_SIZE
    assert (
        len(sub_df) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(sub_df)}"

    # Verify columns
    expected_cols = ["eeg_id"] + Config.CLASS_NAMES
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Verify Probability Constraints (Sum to 1.0)
    # We allow a small tolerance for floating point arithmetic
    probs = sub_df[Config.CLASS_NAMES].values
    row_sums = np.sum(probs, axis=1)

    # Check if all sums are close to 1.0
    valid_sums = np.allclose(row_sums, 1.0, atol=1e-4)
    if not valid_sums:
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-4))[0]
        print(f"Invalid row sums at indices: {bad_rows}")
        print(f"Values: {row_sums[bad_rows]}")
        raise AssertionError("Submission probabilities do not sum to 1.0")

    print("Submission verification successful.")
    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
