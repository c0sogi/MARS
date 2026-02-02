import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processor import get_tokenizer, get_dataloader
from library.symbolic_agent import NgramMemory
from library.neural_agent import NeuralTrainer
from library.pipeline import HybridPredictor


def create_demo_metadata(source_dir: str, target_dir: str, n_rows: int = 5000):
    """
    Creates a small subset of the metadata for demonstration purposes.
    Ensures sentence integrity is maintained.
    """
    print(f"Creating demo metadata in {target_dir} (sampling ~{n_rows} rows)...")
    os.makedirs(target_dir, exist_ok=True)

    files = ["train.csv", "val.csv", "test.csv"]

    for fname in files:
        source_path = os.path.join(source_dir, fname)
        target_path = os.path.join(target_dir, fname)

        if not os.path.exists(source_path):
            print(f"Warning: Source file {source_path} not found. Skipping.")
            continue

        # Read a chunk larger than n_rows to ensure we can cut cleanly at a sentence boundary
        df_chunk = pd.read_csv(source_path, nrows=n_rows * 2)

        if "sentence_id" in df_chunk.columns:
            # Find the sentence_id at the n_rows mark
            if len(df_chunk) > n_rows:
                cutoff_sentence_id = df_chunk.iloc[n_rows]["sentence_id"]
                # Filter to include only full sentences up to that ID
                df_subset = df_chunk[
                    df_chunk["sentence_id"] <= cutoff_sentence_id
                ].copy()
                # Limit to roughly n_rows if the sentence blocks are huge,
                # but usually just taking up to the boundary is fine.
                # To be safe, let's just take the unique sentences that fit within n_rows
                unique_sents = df_chunk["sentence_id"].unique()
                # Take first 100 sentences to be safe and fast
                target_sents = unique_sents[:100]
                df_subset = df_chunk[df_chunk["sentence_id"].isin(target_sents)].copy()
            else:
                df_subset = df_chunk.copy()
        else:
            df_subset = df_chunk.head(n_rows).copy()

        print(f"  Saving {fname}: {len(df_subset)} rows")
        df_subset.to_csv(target_path, index=False)


def demo_symbolic_agent(config: Config):
    print("\n=== Demo: Symbolic Agent ===")

    # Initialize Memory
    memory = NgramMemory(config)

    # Build stats from the demo training data
    # We force load_cached_data=False to ensure it processes our new demo data
    memory.build_stats(load_cached_data=False)

    # Validation: Check if stats were populated
    print(f"  Trigrams: {len(memory.trigrams)}")
    print(f"  Bigrams: {len(memory.bigrams)}")
    print(f"  Unigrams: {len(memory.unigrams)}")

    assert len(memory.unigrams) > 0, "Symbolic memory failed to build unigram stats."

    # Validation: Test a specific known mapping if possible,
    # or just ensure query returns something valid (None or str)
    # Let's pick a token from the demo train set to verify
    train_df = pd.read_csv(os.path.join(config.metadata_dir, "train.csv"))

    # Find a case where before != after (normalization happened)
    # If none found in sample, we skip specific assertion
    changes = train_df[train_df["before"] != train_df["after"]]
    if not changes.empty:
        sample = changes.iloc[0]
        # Test Unigram lookup
        res = memory.query_unigram(str(sample["before"]))
        # Note: It might return None if the count didn't beat the identity,
        # or if context matters, but we check type.
        print(
            f"  Query '{sample['before']}' -> '{res}' (Expected: '{sample['after']}')"
        )

    print("Symbolic Agent verification passed.")


def demo_neural_agent(config: Config):
    print("\n=== Demo: Neural Agent ===")

    # 1. Data Loaders
    print("  Getting DataLoaders...")
    # Using load_cached=False to force processing of the demo metadata
    train_loader = get_dataloader(
        config, split="train", load_cached=False, shuffle=True
    )
    val_loader = get_dataloader(config, split="val", load_cached=False, shuffle=False)

    assert len(train_loader) > 0, "Train loader is empty."

    # 2. Tokenizer
    print("  Initializing Tokenizer...")
    tokenizer = get_tokenizer(config)  # This will fit on the processed demo data
    print(f"  Vocab size: {len(tokenizer)}")
    assert len(tokenizer) > 5, "Tokenizer vocab is suspiciously small."

    # 3. Trainer Initialization
    print("  Initializing NeuralTrainer...")
    trainer = NeuralTrainer(config, tokenizer)

    # 4. Training Loop (1 Epoch)
    print("  Running 1 Epoch of training...")
    train_loss = trainer.train_epoch(train_loader)
    print(f"  Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN."

    # 5. Validation
    val_loss = trainer.validate(val_loader)
    print(f"  Val Loss: {val_loss:.4f}")

    # 6. Save/Load Checkpoint
    print("  Testing Checkpoint Save/Load...")
    trainer.save_model(config.model_checkpoint_path)
    assert os.path.exists(config.model_checkpoint_path), "Checkpoint file not created."

    trainer.load_model(config.model_checkpoint_path)
    print("Neural Agent verification passed.")


def demo_pipeline(config: Config):
    print("\n=== Demo: Hybrid Pipeline Inference ===")

    # Initialize Predictor
    predictor = HybridPredictor(config)

    # Verify model loaded status
    # Since we just trained and saved in demo_neural_agent, it should be ready
    if predictor.model_ready:
        print("  Neural model loaded successfully in pipeline.")
    else:
        print("  Warning: Neural model not loaded in pipeline.")

    # Run full submission generation
    print("  Generating submission...")
    predictor.generate_submission()

    # Verify Output
    sub_path = config.submission_path
    assert os.path.exists(sub_path), "Submission file not found."

    df_sub = pd.read_csv(sub_path)
    print(f"  Submission shape: {df_sub.shape}")
    print("  First 5 rows:")
    print(df_sub.head())

    # Check columns
    expected_cols = ["id", "after"]
    assert list(df_sub.columns) == expected_cols, f"Invalid columns: {df_sub.columns}"

    # Check for empty predictions
    assert not df_sub["after"].isnull().any(), "Submission contains null values."

    print("Pipeline verification passed.")


def main():
    # 1. Setup Configuration for Demo
    # We use a separate working directory to avoid messing up real artifacts
    demo_working_dir = "./working/demo_run_output"
    demo_metadata_dir = "./working/demo_metadata"

    # Clean up previous demo runs if they exist
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_metadata_dir):
        shutil.rmtree(demo_metadata_dir)

    config = Config(
        working_dir=demo_working_dir,
        metadata_dir=demo_metadata_dir,
        # Reduce compute requirements for demo
        epochs=1,
        batch_size=16,
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        max_seq_len=64,
        # Use a small subset ratio for plain text to speed up processing
        plain_subset_ratio=0.1,
    )

    seed_everything(config.seed)

    # 2. Prepare Data
    # We sample from the original ./metadata folder provided in the environment
    original_metadata_dir = "./metadata"
    create_demo_metadata(original_metadata_dir, demo_metadata_dir, n_rows=2000)

    # 3. Run Component Demos
    try:
        demo_symbolic_agent(config)
        demo_neural_agent(config)
        demo_pipeline(config)

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nDemo Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
