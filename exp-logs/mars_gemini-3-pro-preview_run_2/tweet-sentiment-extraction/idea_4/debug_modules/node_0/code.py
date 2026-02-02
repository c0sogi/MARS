import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch

# Import from the provided library
# We need to import the module that holds the class to monkey-patch it effectively
import library.config
from library.utils import seed_everything, jaccard, get_selected_text
from library.dataset import get_loaders, get_test_loader
from library.model import TweetModel
from library.train import run_fold
from library.inference import predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Tweet Sentiment Extraction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Monkey-patch the Config class to run in debug/demo mode
    # This ensures we don't consume too much time or resources
    library.config.Config.DEBUG = True
    library.config.Config.DEBUG_SAMPLE_SIZE = 32  # Very small subset for speed
    library.config.Config.EPOCHS = 1
    library.config.Config.N_FOLDS = (
        2  # Only run 2 folds if we were running full training
    )
    library.config.Config.TRAIN_BATCH_SIZE = 4
    library.config.Config.VALID_BATCH_SIZE = 8

    # Redirect outputs to a demo directory in working
    demo_dir = "./working/demo_output"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    library.config.Config.BASE_OUTPUT_DIR = demo_dir
    library.config.Config.MODEL_OUTPUT_DIR = demo_dir
    library.config.Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    os.makedirs(library.config.Config.CACHE_DIR, exist_ok=True)

    # Set seed
    seed_everything(library.config.Config.SEED)
    print("    Configuration patched. Output dir:", demo_dir)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test Jaccard
    str1 = "hello world"
    str2 = "hello"
    score = jaccard(str1, str2)
    # Intersection: {hello}, Union: {hello, world}. Score = 1/2 = 0.5
    assert (
        abs(score - 0.5) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.5, got {score}"
    print("    Jaccard function: OK")

    # Test get_selected_text
    # Text: "hello world", Offsets mapping roughly to [(0,5), (6,11)]
    # Indices: start=0 ("hello"), end=0 ("hello")
    # Note: Tokenizer offsets usually handle spaces differently, but we test the logic of the function here.
    dummy_text = "hello world"
    dummy_offsets = [(0, 5), (6, 11)]
    extracted = get_selected_text(dummy_text, 0, 0, dummy_offsets)
    assert (
        extracted == "hello"
    ), f"get_selected_text failed. Expected 'hello', got '{extracted}'"

    extracted_full = get_selected_text(dummy_text, 0, 1, dummy_offsets)
    assert (
        extracted_full == "hello world"
    ), f"get_selected_text failed. Expected 'hello world', got '{extracted_full}'"
    print("    get_selected_text function: OK")

    # -------------------------------------------------------------------------
    # 3. Verify Data Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline...")

    # Get loaders for Fold 0
    # We set load_cached_data=False to force processing logic execution
    train_loader, val_loader = get_loaders(fold=0, load_cached_data=False, debug=True)

    print(f"    Train Loader Batches: {len(train_loader)}")
    print(f"    Val Loader Batches:   {len(val_loader)}")

    # Fetch one batch to inspect
    batch = next(iter(train_loader))

    # Check keys
    required_keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "offsets",
        "text",
        "sentiment",
        "start_tokens",
        "end_tokens",
        "selected_text",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check shapes
    # input_ids shape: (batch_size, max_len)
    b_size, max_len = batch["input_ids"].shape
    assert b_size == library.config.Config.TRAIN_BATCH_SIZE
    assert max_len == library.config.Config.MAX_LEN
    assert batch["start_tokens"].shape == (b_size,)

    print("    Batch shapes and keys: OK")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = library.config.Config.DEVICE
    model = TweetModel(library.config.Config)
    model.to(device)
    model.eval()

    # Run forward pass with the batch fetched earlier
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    # Check output shapes: (batch_size, max_len)
    assert start_logits.shape == (
        b_size,
        max_len,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        b_size,
        max_len,
    ), f"End logits shape mismatch: {end_logits.shape}"

    print("    Model forward pass: OK")

    # Clean up memory
    del (
        model,
        batch,
        input_ids,
        attention_mask,
        token_type_ids,
        start_logits,
        end_logits,
    )
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop (Single Fold)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Fold 0)...")

    # Run training for fold 0
    # This uses the patched config (1 Epoch, small subset)
    best_jaccard = run_fold(0)

    print(f"    Fold 0 completed. Best Jaccard: {best_jaccard:.4f}")

    # Check if model file was created
    expected_model_path = os.path.join(
        library.config.Config.MODEL_OUTPUT_DIR, "model_fold_0.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), "Model file was not saved after training."
    print("    Model artifact saved: OK")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Pipeline...")

    # The inference script writes to ./submission/submission.csv by default in the library
    # We need to make sure we check that location.
    # Note: library.inference.predict hardcodes "./submission", so we check there.

    try:
        predict(debug=True)

        submission_path = "./submission/submission.csv"
        assert os.path.exists(submission_path), "Submission file not found."

        # Validate submission format
        sub_df = pd.read_csv(submission_path)
        assert "textID" in sub_df.columns
        assert "selected_text" in sub_df.columns
        assert len(sub_df) > 0

        # Check for quoted strings (as per prompt requirement, though pandas handles reading/writing quotes)
        # The prompt says: "Note that the selected text needs to be quoted... The file should contain a header..."
        # Pandas to_csv with default settings usually handles quoting if necessary, but let's check content.
        print(f"    Submission generated with {len(sub_df)} rows.")
        print("    First few rows:")
        print(sub_df.head())
        print("    Inference pipeline: OK")

    except Exception as e:
        print(f"    Inference failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
