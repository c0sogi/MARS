import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, load_dataset, load_embeddings
from library.data_loader import TextTokenizer, get_dataloaders, ToxicityDataset
from library.model import BiGRU_Pool_Net
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small sample for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.CACHE_DIR = "./working/demo_cache"

    # Clean up any previous demo cache
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("    Config updated: Debug=True, Epochs=1, BatchSize=16")

    # ------------------------------------------------------------------------
    # 2. Testing Utilities (Data Loading)
    # ------------------------------------------------------------------------
    print("\n[2] Testing library.utils.load_dataset...")

    # Load a small chunk of training data to verify merge logic
    # We use the metadata file provided in the environment
    try:
        df_train = load_dataset(Config.TRAIN_METADATA, Config.RAW_TRAIN_PATH)
        print(f"    Successfully loaded dataset. Shape: {df_train.shape}")

        # Validation
        assert "id" in df_train.columns
        assert "comment_text" in df_train.columns
        assert "toxic" in df_train.columns
        # Check that text is not empty (sampling a few)
        sample_text = df_train.iloc[0]["comment_text"]
        assert isinstance(sample_text, str), "Comment text should be string"
        print("    Data integrity check passed.")
    except Exception as e:
        print(f"    Failed to load dataset: {e}")
        raise

    # ------------------------------------------------------------------------
    # 3. Testing Tokenizer
    # ------------------------------------------------------------------------
    print("\n[3] Testing library.data_loader.TextTokenizer...")

    dummy_texts = [
        "This is a test comment.",
        "Another toxic comment example!",
        "Wikipedia is great.",
    ]

    tokenizer = TextTokenizer(max_features=50, max_len=10)
    tokenizer.fit(dummy_texts)
    sequences = tokenizer.transform(dummy_texts)

    print(f"    Vocabulary size: {len(tokenizer.word_index)}")
    print(f"    Transformed shape: {sequences.shape}")

    # Validation
    assert sequences.shape == (3, 10), "Sequence shape mismatch"
    assert sequences.dtype == np.int32, "Sequence dtype mismatch"
    print("    Tokenizer logic verified.")

    # ------------------------------------------------------------------------
    # 4. Testing DataLoaders (Integration)
    # ------------------------------------------------------------------------
    print("\n[4] Testing library.data_loader.get_dataloaders...")

    # This will trigger process_data, caching, and loader creation
    train_loader, val_loader, test_loader, word_index = get_dataloaders(
        debug=Config.DEBUG,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing from scratch for demo
    )

    # Verify Train Loader
    batch_x, batch_y = next(iter(train_loader))
    print(f"    Train Batch X shape: {batch_x.shape}")
    print(f"    Train Batch Y shape: {batch_y.shape}")

    assert (
        batch_x.shape[0] == Config.BATCH_SIZE
        or batch_x.shape[0] <= Config.DEBUG_SAMPLE_SIZE
    )
    assert batch_x.shape[1] == Config.MAX_LEN
    assert batch_y.shape[1] == Config.NUM_CLASSES
    print("    DataLoader shapes verified.")

    # ------------------------------------------------------------------------
    # 5. Testing Model Architecture
    # ------------------------------------------------------------------------
    print("\n[5] Testing library.model.BiGRU_Pool_Net...")

    vocab_size = len(word_index) + 1  # +1 for 0-index padding if not in word_index
    # Note: TextTokenizer handles PAD/UNK within max_features, but safe to use len(word_index)+2 or similar.
    # The utils.load_embeddings uses len(word_index)+1.

    model = BiGRU_Pool_Net(
        vocab_size=Config.MAX_FEATURES + 2,  # Safety margin for demo
        embed_dim=Config.EMBED_DIM,
        hidden_dim=32,  # Smaller for demo
        output_dim=Config.NUM_CLASSES,
        dropout=0.1,
    )

    # Move to CPU for simple shape check
    model.to("cpu")
    model.eval()

    # Create dummy input based on loader shape
    dummy_input = torch.randint(0, 100, (4, Config.MAX_LEN), dtype=torch.long)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output shape: {output.shape}")

    # Validation
    assert output.shape == (4, Config.NUM_CLASSES), "Model output shape mismatch"
    print("    Model forward pass verified.")

    # ------------------------------------------------------------------------
    # 6. Testing Trainer (Full Pipeline)
    # ------------------------------------------------------------------------
    print("\n[6] Testing library.trainer.Trainer (Full Pipeline)...")

    # Initialize Trainer
    # We rely on the cache created in step 4 to save time, or reload.
    # Since we set load_cached_data=False in step 4, files exist in demo_cache.
    # We set load_cached_data=True here to test cache loading.
    trainer = Trainer(debug=Config.DEBUG, load_cached_data=True)

    # 1. Fit
    print("    Running Trainer.fit()...")
    trainer.fit()

    # Check if model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print("    Training complete. Model saved.")

    # 2. Predict
    print("    Running Trainer.predict()...")
    preds = trainer.predict()

    # Validation
    assert preds is not None
    # In debug mode, test set size is DEBUG_SAMPLE_SIZE
    # However, if the test file is smaller than sample size, it takes full length.
    # Here we expect DEBUG_SAMPLE_SIZE rows.
    print(f"    Prediction shape: {preds.shape}")
    assert preds.shape[1] == Config.NUM_CLASSES

    # 3. Save Submission
    print("    Running Trainer.save_submission()...")

    # In debug mode, the predictions (preds) are smaller than the full sample submission.
    # We must align the sample submission file to match the prediction length
    # BEFORE calling save_submission to avoid a ValueError.
    if Config.DEBUG:
        print(
            "    [Action] Creating a temporary debug sample_submission to match prediction length..."
        )
        # Load original sample submission
        orig_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        # Slice it to match preds length
        debug_sample_sub = orig_sub.head(len(preds)).copy()

        # Save to writable location
        writable_sub_path = os.path.join(Config.WORKING_DIR, "sample_submission.csv")
        debug_sample_sub.to_csv(writable_sub_path, index=False)

        # Override Config path so Trainer uses the sliced file
        Config.SAMPLE_SUBMISSION_PATH = writable_sub_path

    trainer.save_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["id"] + Config.LABEL_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"
    assert len(sub_df) == len(preds), "Submission length mismatch"
    print("    Submission logic verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
