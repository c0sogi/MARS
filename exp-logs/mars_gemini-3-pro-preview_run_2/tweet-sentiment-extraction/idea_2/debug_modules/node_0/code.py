import os
import sys
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import TweetDataset
from library.model import TweetModel
from library.engine import train_fn, eval_fn, generate_submission


def create_small_dataset(src_path, dest_path, n=50):
    """
    Creates a small subset of the dataset for rapid demonstration purposes.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file {src_path} not found.")

    df = pd.read_csv(src_path)
    # Take top n rows
    df_small = df.head(n).copy()
    df_small.to_csv(dest_path, index=False)
    print(f"Created small dataset at {dest_path} with {len(df_small)} samples.")


def run_demo():
    # 1. Setup Configuration and Seeds
    print("Initializing Configuration...")
    config = Config(
        epochs=1,
        train_batch_size=8,
        valid_batch_size=8,
        max_len=64,  # Reduced max_len for speed
        model_name="roberta-base",
        debug=True,
    )
    seed_everything(config.SEED)

    # 2. Prepare Small Data for Speed
    # We override the config paths to point to temporary small files in ./working
    print("\nPreparing small datasets for demonstration...")
    small_train_path = os.path.join(config.WORKING_DIR, "demo_train.csv")
    small_val_path = os.path.join(config.WORKING_DIR, "demo_val.csv")
    small_test_path = os.path.join(config.WORKING_DIR, "demo_test.csv")

    create_small_dataset(config.TRAIN_PATH, small_train_path, n=32)
    create_small_dataset(config.VAL_PATH, small_val_path, n=16)
    create_small_dataset(config.TEST_PATH, small_test_path, n=16)

    # Override config paths
    config.TRAIN_PATH = small_train_path
    config.VAL_PATH = small_val_path
    config.TEST_PATH = small_test_path

    # Disable caching for the demo to ensure we process the new small files
    # We do this by ensuring the cache filename logic in dataset.py won't pick up old large files
    # or we simply rely on the fact that the cache key includes 'mode' and 'max_len'.
    # Since we changed max_len to 64, it should re-process.

    # 3. Dataset Instantiation and Verification
    print("\n--- Testing TweetDataset ---")
    train_dataset = TweetDataset(mode="train", config=config, load_cached_data=False)

    print(f"Dataset length: {len(train_dataset)}")
    assert len(train_dataset) == 32, "Dataset length mismatch."

    # Fetch one sample
    sample = train_dataset[0]
    required_keys = [
        "ids",
        "mask",
        "targets_start",
        "targets_end",
        "offsets",
        "textID",
        "text",
        "sentiment",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample."

    # Verify shapes
    # ids and mask should be tensors of shape (max_len,) (since it's a single sample, not batched yet)
    assert sample["ids"].shape == (
        config.MAX_LEN,
    ), f"Expected ids shape ({config.MAX_LEN},), got {sample['ids'].shape}"
    assert sample["mask"].shape == (
        config.MAX_LEN,
    ), f"Expected mask shape ({config.MAX_LEN},), got {sample['mask'].shape}"

    print("Dataset verification passed.")

    # 4. Model Instantiation and Forward Pass
    print("\n--- Testing TweetModel ---")
    device = config.DEVICE
    model = TweetModel(config)
    model.to(device)

    # Create a dummy batch
    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False
    )
    batch = next(iter(train_loader))

    input_ids = batch["ids"].to(device)
    attention_mask = batch["mask"].to(device)

    print(f"Input batch shape: {input_ids.shape}")

    # Forward pass
    start_logits, end_logits = model(input_ids, attention_mask)

    print(f"Output logits shape: Start={start_logits.shape}, End={end_logits.shape}")

    # Verify output shapes: [batch_size, max_len]
    expected_shape = (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    assert (
        start_logits.shape == expected_shape
    ), f"Start logits shape mismatch. Expected {expected_shape}, got {start_logits.shape}"
    assert (
        end_logits.shape == expected_shape
    ), f"End logits shape mismatch. Expected {expected_shape}, got {end_logits.shape}"

    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n--- Testing Training Function ---")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Run one epoch of training
    avg_loss = train_fn(train_loader, model, optimizer, device)
    print(f"Training finished. Average Loss: {avg_loss:.4f}")

    assert avg_loss > 0, "Training loss should be positive."
    assert isinstance(avg_loss, float), "Training loss should be a float."

    # 6. Evaluation Loop Demonstration
    print("\n--- Testing Evaluation Function ---")
    val_dataset = TweetDataset(mode="val", config=config, load_cached_data=False)
    val_loader = DataLoader(
        val_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False
    )

    # Load the validation dataframe for ground truth comparison (required by eval_fn)
    df_val = pd.read_csv(config.VAL_PATH)

    val_loss, val_jaccard = eval_fn(val_loader, model, device, df_val)
    print(f"Validation finished. Loss: {val_loss:.4f}, Jaccard: {val_jaccard:.4f}")

    assert val_loss > 0, "Validation loss should be positive."
    assert 0 <= val_jaccard <= 1, "Jaccard score must be between 0 and 1."

    # 7. Submission Generation Demonstration
    print("\n--- Testing Submission Generation ---")
    test_dataset = TweetDataset(mode="test", config=config, load_cached_data=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False
    )

    output_csv = os.path.join(config.OUTPUT_DIR, "demo_submission.csv")
    generate_submission(test_loader, model, device, output_csv)

    # Verify output file
    assert os.path.exists(output_csv), "Submission file was not created."
    df_sub = pd.read_csv(output_csv)
    print(f"Submission file created with {len(df_sub)} rows.")

    assert len(df_sub) == len(test_dataset), "Submission row count mismatch."
    assert (
        "textID" in df_sub.columns and "selected_text" in df_sub.columns
    ), "Submission columns mismatch."

    # Check if selected_text contains strings (and handle NaNs if any, though logic prevents them usually)
    assert df_sub["selected_text"].dropna().shape[0] == len(
        df_sub
    ), "Submission contains null values."

    print("\nAll demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    import warnings

    warnings.filterwarnings("ignore")

    run_demo()
