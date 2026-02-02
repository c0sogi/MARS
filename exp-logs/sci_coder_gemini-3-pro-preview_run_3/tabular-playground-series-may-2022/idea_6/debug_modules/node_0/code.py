import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, get_device
from library.preprocessing import DataProcessor
from library.dataset import ManufacturingDataset
from library.model import HybridTransformerFunnel
from library.engine import Engine


def main():
    print("Starting demonstration of the Manufacturing Control pipeline...")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    # Enable debug mode to use a small subset (10,000 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 5000  # Small enough for very fast execution

    # Reduce training duration
    Config.EPOCHS = 2
    Config.PATIENCE = 1

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(
        f"Configuration: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, DEVICE={get_device()}"
    )

    # --------------------------------------------------------------------------
    # 2. Data Processing
    # --------------------------------------------------------------------------
    print("\n[Step 1] Processing Data...")
    processor = DataProcessor()

    # Force re-processing to ensure we use the debug subset
    # In a real run, we might use load_cached_data=True
    train_df, val_df, test_df, vocab_sizes = processor.process_data(
        load_cached_data=False
    )

    # Verification
    print(f"Train shape: {train_df.shape}")
    print(f"Vocab sizes: {vocab_sizes}")

    assert (
        len(train_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} training samples in debug mode, got {len(train_df)}"
    assert (
        len(vocab_sizes) == 12
    ), f"Expected 12 categorical features (10 chars + f_29 + f_30), got {len(vocab_sizes)}"
    assert (
        "unique_character_count" in train_df.columns
    ), "Feature engineering failed: 'unique_character_count' missing."

    print("Data processing verified.")

    # --------------------------------------------------------------------------
    # 3. Datasets and DataLoaders
    # --------------------------------------------------------------------------
    print("\n[Step 2] Creating Datasets and Loaders...")

    train_dataset = ManufacturingDataset(train_df, is_test=False)
    val_dataset = ManufacturingDataset(val_df, is_test=False)
    test_dataset = ManufacturingDataset(test_df, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple script execution safety
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verification: Check one batch
    sample_batch = next(iter(train_loader))
    cat_seq = sample_batch["cat_seq"]
    cont_vec = sample_batch["cont_vec"]
    targets = sample_batch["target"]

    print(
        f"Batch shapes - Cat: {cat_seq.shape}, Cont: {cont_vec.shape}, Target: {targets.shape}"
    )

    assert cat_seq.shape == (
        Config.BATCH_SIZE,
        12,
    ), "Incorrect categorical sequence shape."
    assert cont_vec.shape[1] == len(
        Config.CONTINUOUS_FEATURE_NAMES
    ), "Incorrect continuous feature dimension."
    assert targets.shape == (Config.BATCH_SIZE, 1), "Incorrect target shape."

    print("Datasets and DataLoaders verified.")

    # --------------------------------------------------------------------------
    # 4. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    continuous_dim = len(Config.CONTINUOUS_FEATURE_NAMES)
    model = HybridTransformerFunnel(
        vocab_sizes=vocab_sizes, continuous_dim=continuous_dim
    )

    # Verification: Forward pass on CPU before moving to device
    with torch.no_grad():
        dummy_logits = model(cat_seq.cpu(), cont_vec.cpu())

    assert dummy_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {dummy_logits.shape}"

    print("Model initialized and architecture verified.")

    # --------------------------------------------------------------------------
    # 5. Training Loop (Engine)
    # --------------------------------------------------------------------------
    print("\n[Step 4] Training Model...")

    engine = Engine(model)

    # Run training
    # This will train for Config.EPOCHS (2) and save the best model
    engine.fit(train_loader, val_loader)

    # Verify model file exists
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    # Get test IDs from the dataframe
    test_ids = test_df[Config.ID_COL].values

    # Generate submission
    engine.generate_submission(test_loader, test_ids)

    # Verify submission file
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission head:\n{submission_df.head()}")

    assert submission_df.shape == (
        len(test_df),
        2,
    ), f"Submission shape mismatch. Expected ({len(test_df)}, 2), got {submission_df.shape}"
    assert list(submission_df.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch."
    assert (
        not submission_df[Config.TARGET_COL].isnull().any()
    ), "Submission contains null predictions."

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
