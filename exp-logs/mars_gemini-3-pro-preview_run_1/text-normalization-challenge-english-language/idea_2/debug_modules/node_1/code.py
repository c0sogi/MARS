import os
import torch
import pandas as pd
import shutil
import sys

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.engine import train_model
from library.inference import generate_submission
from library.model import TransformerTagger


def run_demo():
    print("=== Starting Text Normalization Demo ===")

    # 1. Configure for Speed and Isolation
    # ------------------------------------
    print("\n[1] Configuring environment...")

    # Set fixed seed
    set_seed(42)

    # Override Config for a fast debug run
    Config.DEBUG = True
    Config.DEBUG_SIZE = 1000  # Process only 1000 sentences
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Smaller batch size for demo

    # Use a separate working directory for this demo to avoid path conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Re-create paths based on new working dir
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update derived paths in Config
    Config.VOCAB_TOKENS_PATH = os.path.join(Config.CACHE_DIR, "vocab_tokens.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(Config.CACHE_DIR, "vocab_classes.parquet")
    Config.TRAIN_GROUPED_PATH = os.path.join(Config.CACHE_DIR, "train_grouped.parquet")
    Config.VAL_GROUPED_PATH = os.path.join(Config.CACHE_DIR, "val_grouped.parquet")
    Config.TEST_GROUPED_PATH = os.path.join(Config.CACHE_DIR, "test_grouped.parquet")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(
        Config.CACHE_DIR, "knowledge_base.parquet"
    )
    Config.MODEL_SAVE_PATH = os.path.join(Config.CACHE_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading & Verification
    # ------------------------------
    print("\n[2] Loading and processing data...")

    # Force reload (load_cached_data=False) to ensure we process from scratch for the demo
    # logic, but saving to our new demo directory.
    train_loader, val_loader, test_loader, vocab_tokens, vocab_classes, kb = (
        get_dataloaders(load_cached_data=False)
    )

    # Assertions to verify data integrity
    print("Verifying data structures...")

    # Check Vocabularies
    assert len(vocab_tokens) > 0, "Token vocabulary is empty!"
    assert len(vocab_classes) > 0, "Class vocabulary is empty!"
    print(f"Token Vocab Size: {len(vocab_tokens)}")
    print(f"Class Vocab Size: {len(vocab_classes)}")

    # Check Knowledge Base
    assert isinstance(kb, dict), "Knowledge Base should be a dictionary."
    print(f"Knowledge Base Entries: {len(kb)}")

    # Check DataLoader Batch Structure
    batch = next(iter(train_loader))
    assert "input_ids" in batch, "Batch missing 'input_ids'"
    assert "class_ids" in batch, "Batch missing 'class_ids'"
    assert "attention_mask" in batch, "Batch missing 'attention_mask'"

    # Check shapes: [batch_size, seq_len]
    input_shape = batch["input_ids"].shape
    assert input_shape[0] == Config.BATCH_SIZE or input_shape[0] == len(
        train_loader.dataset
    ), f"Unexpected batch size: {input_shape[0]}"
    assert (
        input_shape[1] == Config.MAX_LEN
    ), f"Unexpected sequence length: {input_shape[1]}"

    print("Data verification passed.")

    # 3. Model Initialization & Training
    # ----------------------------------
    print("\n[3] Training model...")

    # We use the provided training engine
    # This will train for 1 epoch on the debug subset
    model = train_model(train_loader, val_loader, vocab_tokens, vocab_classes)

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Model training complete and checkpoint saved.")

    # Quick sanity check on model forward pass
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        logits = model(test_batch["input_ids"].to(Config.DEVICE))
        # Logits shape: [batch, seq_len, num_classes]
        assert logits.dim() == 3
        assert logits.size(2) == len(vocab_classes)
    print("Model forward pass verified.")

    # 4. Inference & Submission Generation
    # ------------------------------------
    print("\n[4] Generating submission...")

    # Generate submission using the trained model and artifacts
    # We pass debug=True to ensure it respects the debug settings (though Config is already set)
    # We pass load_cached_data=True to reuse the artifacts we just created in step 2
    generate_submission(debug=True, load_cached_data=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded submission with {len(df_sub)} rows.")

    # Verify columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "after" in df_sub.columns, "Submission missing 'after' column"

    # Verify content isn't empty
    assert not df_sub.empty, "Submission dataframe is empty"

    # Verify ID format (simple check on first row)
    first_id = df_sub.iloc[0]["id"]
    assert "_" in str(first_id), f"Invalid ID format: {first_id}"

    print("Submission verification passed.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
