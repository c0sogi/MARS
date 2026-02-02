import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, prepare_data, ToxicityDataset
from library.model import MultiTaskRoberta
from library.loss import AggressiveMultiTaskLoss
from library.engine import Engine
from library.metrics import calculate_jigsaw_metrics


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(42)

    # Define a working directory for this demo
    DEMO_DIR = "./working/demo_run"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config to use this directory and speed up training
    Config.WORKING_DIR = DEMO_DIR
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.TEST_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.MAX_LEN = 128  # Reduce sequence length for speed

    # Update cache paths in Config to point to the new working dir
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_data.npy")
    Config.VALID_CACHE_PATH = os.path.join(DEMO_DIR, "valid_data.npy")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_data.npy")
    Config.TOKENIZER_CACHE_DIR = os.path.join(DEMO_DIR, "tokenizer")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Create Mini-Datasets (Subsampling)
    # --------------------------------------------------------------------------
    # To ensure the code runs quickly, we subsample the metadata files.
    # This forces the data pipeline to only process a small number of texts.

    print("Creating mini-datasets for rapid demonstration...")

    def create_mini_csv(original_path, new_path, n_samples):
        df = pd.read_csv(original_path)
        # Sample top N to ensure deterministic behavior
        df_mini = df.head(n_samples).copy()
        df_mini.to_csv(new_path, index=False)
        return new_path

    # Define paths for mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "train_mini.csv")
    mini_valid_path = os.path.join(DEMO_DIR, "val_mini.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test_mini.csv")

    # Create them (100 train, 20 val, 20 test)
    # Note: We read from the provided ./metadata directory
    Config.TRAIN_METADATA_PATH = create_mini_csv(
        os.path.join(Config.METADATA_DIR, "train.csv"), mini_train_path, 100
    )
    Config.VALID_METADATA_PATH = create_mini_csv(
        os.path.join(Config.METADATA_DIR, "validation.csv"), mini_valid_path, 20
    )
    Config.TEST_METADATA_PATH = create_mini_csv(
        os.path.join(Config.METADATA_DIR, "test.csv"), mini_test_path, 20
    )

    print("Mini-datasets created.")

    # --------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # --------------------------------------------------------------------------
    print("\n=== Verifying Data Pipeline ===")

    # We use get_dataloaders which internally calls prepare_data.
    # prepare_data will read our mini metadata, load the corresponding text from input/,
    # tokenize it, and save caches to DEMO_DIR.
    # load_cached_data=False ensures we process the new mini datasets.
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check batch structure
    batch = next(iter(train_loader))
    print("Batch Keys:", batch.keys())

    # Assertions to verify data loading logic
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch
    assert "identities" in batch
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE

    print("Data Pipeline Verified.")

    # --------------------------------------------------------------------------
    # 4. Verify Model & Loss Components
    # --------------------------------------------------------------------------
    print("\n=== Verifying Model & Loss ===")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = MultiTaskRoberta()
    model.to(device)
    model.eval()

    # Create dummy inputs from the fetched batch
    dummy_input_ids = batch["input_ids"].to(device)
    dummy_mask = batch["attention_mask"].to(device)
    dummy_targets = batch["target"].to(device)
    dummy_identities = batch["identities"].to(device)

    # Forward pass
    with torch.no_grad():
        tox_logits, ident_logits = model(dummy_input_ids, dummy_mask)

    print(f"Toxicity Logits Shape: {tox_logits.shape}")
    print(f"Identity Logits Shape: {ident_logits.shape}")

    # Assertions for model output shapes
    assert tox_logits.shape == (Config.TRAIN_BATCH_SIZE, 1)
    assert ident_logits.shape == (Config.TRAIN_BATCH_SIZE, len(Config.IDENTITY_COLUMNS))

    # Loss Calculation
    loss_fn = AggressiveMultiTaskLoss()
    loss = loss_fn(tox_logits, ident_logits, dummy_targets, dummy_identities)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss)
    assert loss.item() > 0

    print("Model & Loss Verified.")

    # --------------------------------------------------------------------------
    # 5. Run Training Engine
    # --------------------------------------------------------------------------
    print("\n=== Running Training Engine ===")

    # Initialize Engine with GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Engine Device: {device}")

    engine = Engine(device=device)

    # Run the full training loop (Train -> Val -> Save -> Predict)
    # This uses the mini-datasets and 1 epoch, so it should be fast.
    engine.run_training(train_loader, val_loader, test_loader)

    # --------------------------------------------------------------------------
    # 6. Verify Submission
    # --------------------------------------------------------------------------
    print("\n=== Verifying Submission ===")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission File Found: {Config.SUBMISSION_PATH}")
        print(f"Rows: {len(sub_df)}")
        print(sub_df.head())

        # Assertions
        assert len(sub_df) == 20, f"Expected 20 predictions, got {len(sub_df)}"
        assert "id" in sub_df.columns
        assert "prediction" in sub_df.columns
        assert sub_df["prediction"].dtype == float
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
