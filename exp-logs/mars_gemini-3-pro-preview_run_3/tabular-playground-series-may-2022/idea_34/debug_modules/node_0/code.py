import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_loader import DataProcessor, ManufacturingDataset
from library.model import ARPFEModel
from library.trainer import Trainer


def run_demonstration():
    print("=" * 50)
    print("Starting ARPFE Library Demonstration")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for demo run...")

    # Use a specific directory for this demo to avoid overwriting production work
    DEMO_DIR = "./working/demo_execution/"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config class attributes
    Config.CACHE_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Optimize hyperparameters for a quick run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4096  # Large batch size to reduce steps per epoch

    print(f"Cache Directory: {Config.CACHE_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")

    # ---------------------------------------------------------
    # 2. Data Processing and Feature Engineering
    # ---------------------------------------------------------
    print("\n[2] Testing DataProcessor and Feature Engineering...")

    # Force recompute=True (load_cached_data=False) to test the processing logic
    train_df, val_df, test_df, vocab_sizes = DataProcessor.process_data(
        load_cached_data=False
    )

    # Validations
    print("Validating processed data...")
    assert not train_df.empty, "Training dataframe is empty."
    assert not val_df.empty, "Validation dataframe is empty."
    assert not test_df.empty, "Test dataframe is empty."

    # Check Feature Engineering (f_27 decomposition)
    # Config.F27_LENGTH is 10, so we expect f_27_char_0 to f_27_char_9
    expected_col = "f_27_char_0"
    assert (
        expected_col in train_df.columns
    ), f"Feature engineering failed: {expected_col} missing."

    # Check Aggregate Features
    assert (
        "unique_character_count" in train_df.columns
    ), "Aggregate feature 'unique_character_count' missing."

    # Check Target
    assert "target" in train_df.columns, "Target column missing in training data."

    print(f"Train Shape: {train_df.shape}")
    print(f"Vocab Sizes: {vocab_sizes}")
    print("Data processing logic verified.")

    # ---------------------------------------------------------
    # 3. Dataset Class Verification
    # ---------------------------------------------------------
    print("\n[3] Testing ManufacturingDataset...")

    train_ds = ManufacturingDataset(train_df, is_test=False)
    test_ds = ManufacturingDataset(test_df, is_test=True)

    # Fetch one sample
    cont_x, cat_x, target = train_ds[0]

    # Verify types and shapes
    assert isinstance(cont_x, torch.Tensor), "Continuous features must be a Tensor."
    assert isinstance(cat_x, torch.Tensor), "Categorical features must be a Tensor."
    assert isinstance(target, torch.Tensor), "Target must be a Tensor."

    assert cont_x.dtype == torch.float32, "Continuous features should be float32."
    assert cat_x.dtype == torch.long, "Categorical features should be long (int64)."

    print(f"Sample Cont Shape: {cont_x.shape}")
    print(f"Sample Cat Shape: {cat_x.shape}")
    print("Dataset implementation verified.")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4] Testing ARPFEModel Architecture...")

    model = ARPFEModel(vocab_sizes)

    # Create a dummy batch
    batch_size = 8
    dummy_cont = cont_x.unsqueeze(0).repeat(batch_size, 1)
    dummy_cat = cat_x.unsqueeze(0).repeat(batch_size, 1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Check Output Shape
    # The model returns logits for 5 independent streams -> (Batch, 5)
    expected_shape = (batch_size, 5)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"Model Output Shape: {output.shape}")
    print("Model architecture verified.")

    # ---------------------------------------------------------
    # 5. Trainer Execution (Train Loop & Inference)
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop via Trainer...")

    # Initialize Trainer
    # We set load_cached_data=True because we just generated the cache in step [2]
    trainer = Trainer(load_cached_data=True)

    # Run Training
    # This will run for 1 epoch as configured
    trainer.train(epochs=Config.EPOCHS)

    # Validate manually (optional, train() does this, but checking API)
    auc, loss = trainer.validate()
    print(f"Final Validation - AUC: {auc:.4f}, Loss: {loss:.4f}")
    assert 0 <= auc <= 1, "AUC score out of range."

    # Generate Submission
    print("Generating submission file...")
    trainer.generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(sub_df) == len(test_df), "Submission row count mismatch."
    assert (
        "id" in sub_df.columns and "target" in sub_df.columns
    ), "Submission columns missing."

    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission Head:\n{sub_df.head()}")

    print("\n" + "=" * 50)
    print("Demonstration Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    try:
        run_demonstration()
    except AssertionError as e:
        print(f"\n[ERROR] Validation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        # Print stack trace for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
