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
    trainer.save_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["id"] + Config.LABEL_COLS
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # In debug mode, the submission file is generated based on the test_loader.
    # The test_loader in debug mode has DEBUG_SAMPLE_SIZE items.
    # The sample_submission.csv has 153164 items.
    # The Trainer.save_submission logic reads sample_submission and assigns preds.
    # CAUTION: In a real debug run, if preds length doesn't match sample_submission length,
    # pandas assignment might fail or require index alignment.
    # Let's check how Trainer handles it.
    # Trainer code: submission[Config.LABEL_COLS] = test_preds
    # If len(test_preds) != len(submission), this raises ValueError.
    # However, in the provided Trainer code, it does exactly that.
    # Thus, for the code to work without error in Debug mode, the Trainer logic assumes
    # we are predicting for the whole set OR the user handles the mismatch.
    #
    # Wait, looking at `get_dataloaders` in `library/data_loader.py`:
    # If debug=True, X_test is sliced: X_test = X_test[: Config.DEBUG_SAMPLE_SIZE]
    # So `test_preds` will have length `DEBUG_SAMPLE_SIZE`.
    # `sample_submission` has full length.
    # `submission[Config.LABEL_COLS] = test_preds` will FAIL if lengths differ.
    #
    # To make this demo pass successfully without modifying library code,
    # we must acknowledge that `Trainer.save_submission` might fail in DEBUG mode
    # if the library doesn't handle slicing the sample_submission dataframe.
    # Looking at `library/trainer.py`:
    # `submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)`
    # `submission[Config.LABEL_COLS] = test_preds`
    #
    # Since I cannot modify `library/trainer.py`, this step will fail in a strict debug run
    # unless I manually handle the prediction saving or mock the sample_submission path
    # to match the debug size.

    print(
        "    [Info] Standard Trainer.save_submission() might fail in DEBUG mode due to length mismatch."
    )
    print(
        "    [Action] Creating a temporary debug sample_submission to allow success..."
    )

    # Create a dummy sample submission matching debug size
    debug_sample_sub = sub_df.head(len(preds)).copy()
    debug_sub_path = os.path.join(Config.INPUT_DIR, "debug_sample_submission.csv")

    # We cannot write to INPUT_DIR (read-only).
    # We must point Config.SAMPLE_SUBMISSION_PATH to a writable location.
    writable_sub_path = os.path.join(Config.WORKING_DIR, "sample_submission.csv")
    debug_sample_sub.to_csv(writable_sub_path, index=False)

    # Override Config path
    Config.SAMPLE_SUBMISSION_PATH = writable_sub_path

    # Retry save
    trainer.save_submission()
    assert os.path.exists(Config.SUBMISSION_PATH)
    final_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(final_df) == len(preds)
    print("    Submission logic verified with aligned data lengths.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
