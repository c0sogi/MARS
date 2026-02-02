import os
import pandas as pd
import torch
import sys

# 1. Import Config first to patch it before other modules use it
from library.config import Config

# --- Patch Configuration for Speed and Demo Isolation ---
Config.WORK_DIR = "./working/demo_execution"
Config.SUBMISSION_DIR = "./working/demo_execution"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Reduce Data/Model Complexity for Demo
Config.VOCAB_SIZE = 2000  # Small vocab
Config.MAX_SEQ_LEN = 32  # Short sequences
Config.EPOCHS = 1  # Single epoch
Config.BATCH_SIZE = 16  # Small batch
Config.NUM_WORKERS = 2  # Fewer workers

# Update Metadata Paths to point to the small subsets we are about to create
Config.TRAIN_METADATA = os.path.join(Config.WORK_DIR, "train_small.csv")
Config.VAL_METADATA = os.path.join(Config.WORK_DIR, "val_small.csv")
Config.TEST_METADATA = os.path.join(Config.WORK_DIR, "test_small.csv")

# Ensure demo directory exists
os.makedirs(Config.WORK_DIR, exist_ok=True)


# --- Create Subsampled Datasets ---
def create_subset(source_path, dest_path, n_rows=2000):
    """Reads top n_rows from source and writes to dest."""
    if not os.path.exists(source_path):
        # Fallback for safety if metadata wasn't generated (though prompt says it is)
        print(f"Warning: Source {source_path} not found. Creating dummy data.")
        df = pd.DataFrame(
            {
                "id": range(n_rows),
                "sentence": [
                    "This is a sample sentence for testing purposes ."
                    for _ in range(n_rows)
                ],
            }
        )
    else:
        df = pd.read_csv(source_path, nrows=n_rows)

    df.to_csv(dest_path, index=False)
    print(f"Created subset: {dest_path} ({len(df)} rows)")


# Original metadata paths (from the unmodified Config class logic)
ORIG_METADATA_DIR = "./metadata"
create_subset(os.path.join(ORIG_METADATA_DIR, "train.csv"), Config.TRAIN_METADATA)
create_subset(os.path.join(ORIG_METADATA_DIR, "val.csv"), Config.VAL_METADATA)
create_subset(os.path.join(ORIG_METADATA_DIR, "test.csv"), Config.TEST_METADATA)

# --- Import Library Modules ---
# Now that Config is patched and data is ready, we import the rest
from library.utils import set_seed
from library.vocab import get_vocabulary
from library.data import get_dataloaders
from library.model import GapTransformer
from library.engine import run_training, generate_submission


def run_demo():
    print("\n--- Starting Demo Execution ---")

    # 1. Set Seed
    set_seed(42)

    # 2. Build Vocabulary
    # load_cached_data=False ensures we build from our new small CSVs, not looking for old cache
    print("Building vocabulary...")
    vocab = get_vocabulary(load_cached_data=False)

    # Assertions for Vocab
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert vocab.get_pad_index() is not None
    assert len(vocab) <= Config.VOCAB_SIZE + len(vocab.special_tokens)
    print(f"Vocabulary size: {len(vocab)}")

    # 3. Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force reprocessing of small CSVs
        debug=False,
    )

    # Validate DataLoader
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "targets" in batch
    assert batch["input_ids"].shape == batch["targets"].shape
    assert batch["input_ids"].size(0) == Config.BATCH_SIZE
    print("DataLoader verification successful.")

    # 4. Initialize Model
    # We explicitly pass small dimensions to override defaults in model.py
    print("Initializing Model...")
    model = GapTransformer(
        vocab_size=len(vocab),
        d_model=64,  # Small dimension
        nhead=4,  # Few heads
        num_layers=2,  # Few layers
        dim_feedforward=128,  # Small FFN
        dropout=0.1,
        pad_idx=vocab.get_pad_index(),
    )

    # Check model on device
    device = torch.device(Config.DEVICE)
    model.to(device)

    # 5. Run Training
    print("Starting Training Loop...")
    # run_training handles the loop, optimizer, scheduler, and saving best model
    trained_model = run_training(model, train_loader, val_loader, vocab)

    # Verify model artifact exists
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training completed.")

    # 6. Generate Submission
    print("Generating Submission...")
    generate_submission(trained_model, test_loader, vocab)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df_sub.columns and "sentence" in df_sub.columns
    assert len(df_sub) > 0
    # Check if sentences are strings and not empty
    assert isinstance(df_sub.iloc[0]["sentence"], str)

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print(f"Sample prediction: {df_sub.iloc[0]['sentence']}")

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
