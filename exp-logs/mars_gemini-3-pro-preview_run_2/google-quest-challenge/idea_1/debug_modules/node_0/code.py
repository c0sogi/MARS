import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders
from library.network import DualAveNet
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. Setup Environment
    warnings.filterwarnings("ignore")
    set_seed(42)
    print("Starting implementation demonstration...")

    # 2. Configure Hyperparameters for Speed
    # We modify the Config class attributes directly to run a fast demo
    print("Adjusting configuration for rapid execution...")
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.VOCAB_SIZE = 500  # Small vocab for speed
    Config.MAX_LEN = 32  # Short sequences
    Config.EMBED_DIM = 16  # Small embedding
    Config.HIDDEN_DIM = 16  # Small hidden layer
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directories exist
    Config.setup()

    # 3. Demonstrate Data Loading
    print("\n--- Step 1: Data Loading & Processing ---")
    # Use a tiny debug sample size and force reprocessing (load_cached_data=False)
    # to demonstrate the pipeline without processing the full dataset.
    debug_size = 12
    train_loader, val_loader, test_loader, vocab, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,
        debug_sample_size=debug_size,
    )

    print(f"Vocabulary Size: {vocab.vocab_size}")
    print(f"Number of Training Batches: {len(train_loader)}")

    # Fetch one batch to verify structure
    q_batch, a_batch, y_batch = next(iter(train_loader))
    print(
        f"Sample Batch Shapes -> Q: {q_batch.shape}, A: {a_batch.shape}, Y: {y_batch.shape}"
    )

    # Assertions to verify data integrity
    assert vocab.vocab_size <= Config.VOCAB_SIZE + 2, "Vocabulary size exceeded limit."
    assert q_batch.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Question tensor shape mismatch."
    assert a_batch.shape == (
        Config.BATCH_SIZE,
        Config.MAX_LEN,
    ), "Answer tensor shape mismatch."
    assert y_batch.shape == (Config.BATCH_SIZE, 30), "Target tensor shape mismatch."
    assert len(test_ids) == debug_size, "Test ID count mismatch."

    # 4. Demonstrate Model Usage
    print("\n--- Step 2: Network Initialization & Forward Pass ---")
    model = DualAveNet()

    # Ensure model is on the correct device (CPU for this quick check)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Perform a forward pass with the sample batch
    with torch.no_grad():
        preds = model(q_batch.to(device), a_batch.to(device))

    print(f"Predictions Shape: {preds.shape}")
    print(
        f"Predictions Range: Min={preds.min().item():.4f}, Max={preds.max().item():.4f}"
    )

    # Assertions to verify model output
    assert preds.shape == (Config.BATCH_SIZE, 30), "Model output shape mismatch."
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Model predictions out of [0,1] range."

    # 5. Demonstrate Trainer (Full Pipeline)
    print("\n--- Step 3: Training & Submission Generation ---")
    # Initialize Trainer with the debug settings
    # This encapsulates the loop: Train -> Validate -> Early Stop -> Predict
    trainer = Trainer(load_cached_data=False, debug_sample_size=debug_size)

    # Run the training loop
    trainer.run(epochs=Config.EPOCHS)

    # 6. Verify Submission File
    print("\n--- Step 4: Submission Verification ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded: {submission_df.shape}")

    # Verify dimensions
    # Rows should equal the debug_sample_size (since test set is also sliced)
    expected_rows = debug_size
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(submission_df)}"

    # Verify columns (qa_id + 30 targets)
    expected_cols = ["qa_id"] + Config.TARGET_COLS
    assert (
        list(submission_df.columns) == expected_cols
    ), "Submission columns do not match requirements."

    # Verify content validity
    pred_values = submission_df[Config.TARGET_COLS].values
    assert (pred_values >= 0).all() and (
        pred_values <= 1
    ).all(), "Submission contains invalid probability values."

    print("\nDemonstration completed successfully. All checks passed.")


if __name__ == "__main__":
    main()
