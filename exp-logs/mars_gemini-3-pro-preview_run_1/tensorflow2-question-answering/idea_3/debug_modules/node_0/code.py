import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import library modules
import library.utils
import library.data_loader
import library.modeling
import library.trainer
import library.inference

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 1
SUBSET_SIZE = 50  # Number of samples to use for this demo
WORKING_DIR = "./working/demo_run"
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# --- Monkey-Patching for Speed ---
# To ensure the code runs quickly, we intercept the metadata loading function
# to return only a small subset of the data. This prevents processing the
# entire dataset which would take too long for a demo.

original_load_metadata = library.utils.load_metadata


def mock_load_metadata(split):
    """
    Loads the real metadata but returns only the first SUBSET_SIZE rows.
    """
    # Load the actual parquet file
    df = original_load_metadata(split)
    # Return a tiny subset
    print(
        f"[Demo] Loading subset of {split} metadata: {SUBSET_SIZE} rows out of {len(df)}"
    )
    return df.head(SUBSET_SIZE)


# Apply patch to relevant modules
library.data_loader.load_metadata = mock_load_metadata
library.inference.load_metadata = mock_load_metadata


def run_demo():
    print(f"--- Starting Demo on {DEVICE} ---")

    # 1. Set Seed for Reproducibility
    print("\n1. Setting Seeds")
    library.utils.set_seed(42)

    # 2. Build Tokenizer
    print("\n2. Building Tokenizer")
    # We use the mocked metadata loader to get a small train dataframe
    train_meta_subset = mock_load_metadata("train")

    # Force load_cached_data=False to ensure we generate vocab from our subset
    # and don't pick up a potentially large existing cache.
    tokenizer = library.data_loader.build_tokenizer(
        train_meta_subset, sample_size=SUBSET_SIZE, load_cached_data=False
    )
    vocab_size = len(tokenizer)
    print(f"Tokenizer built. Vocab size: {vocab_size}")

    # Assertion to ensure tokenizer works
    assert vocab_size >= 2, "Vocabulary should at least contain PAD and UNK"
    encoded = tokenizer.encode("hello world")
    assert isinstance(encoded, list), "Tokenizer encode should return a list"

    # 3. Prepare DataLoaders
    print("\n3. Preparing DataLoaders")
    # We use load_cached_data=False to force reprocessing the flattened data
    # based on our small metadata subset.
    train_loader = library.data_loader.get_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=128,
        neg_ratio=0.5,  # Higher ratio for demo to ensure we have negatives
        load_cached_data=False,
    )

    val_loader = library.data_loader.get_dataloader(
        split="val",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=128,
        load_cached_data=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    required_keys = ["q_input_ids", "c_input_ids", "label_long", "sa_labels"]
    for key in required_keys:
        assert key in sample_batch, f"Batch missing key: {key}"
    print("Batch structure verified.")

    # 4. Initialize Model
    print("\n4. Initializing Model")
    model = library.modeling.DanTqpModel(
        vocab_size=vocab_size, embedding_dim=32, hidden_dim=32  # Small dim for speed
    )
    model.to(DEVICE)

    # Verify forward pass
    with torch.no_grad():
        q_dummy = torch.randint(0, vocab_size, (2, 10)).to(DEVICE)
        c_dummy = torch.randint(0, vocab_size, (2, 20)).to(DEVICE)
        r_logits, e_logits = model(q_dummy, c_dummy)
        assert r_logits.shape == (
            2,
            1,
        ), f"Ranker logits shape mismatch: {r_logits.shape}"
        assert e_logits.shape == (
            2,
            20,
            3,
        ), f"Extractor logits shape mismatch: {e_logits.shape}"
    print("Model forward pass verified.")

    # 5. Training Loop
    print("\n5. Running Training Loop")
    trainer = library.trainer.ModelTrainer(model, DEVICE, learning_rate=0.01)

    # Train for 1 epoch
    trainer.train(
        train_loader, val_loader, epochs=EPOCHS, patience=1, save_path=CHECKPOINT_PATH
    )

    assert os.path.exists(CHECKPOINT_PATH), "Checkpoint file was not created."
    print("Training completed and checkpoint verified.")

    # 6. Inference Pipeline
    print("\n6. Running Inference")

    # Load test data (subset)
    test_loader = library.data_loader.get_dataloader(
        split="test",
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=128,
        load_cached_data=False,
    )

    # Initialize Generator
    generator = library.inference.SubmissionGenerator(
        model,
        DEVICE,
        tokenizer,
        long_threshold=0.4,  # Low threshold to ensure some positives in demo
        short_threshold=0.05,
    )

    # Predict
    predictions = generator.predict(test_loader)
    print(f"Generated predictions for {len(predictions)} examples.")

    # Generate Submission File
    # We use the raw test file path as required by the function
    test_file_raw = os.path.join(
        library.inference.INPUT_DIR, "simplified-nq-test.jsonl"
    )

    generator.generate_submission_file(predictions, test_file_raw, SUBMISSION_PATH)

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file shape: {sub_df.shape}")
    assert "example_id" in sub_df.columns
    assert "PredictionString" in sub_df.columns

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        # Print full traceback for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)
