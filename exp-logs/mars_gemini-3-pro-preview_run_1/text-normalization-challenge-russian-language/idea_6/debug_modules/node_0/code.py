import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, ensure_dir, safe_load_dataframe
from library.data_manager import DataManager
from library.tokenizers import HybridTokenizer
from library.symbolic_model import HierarchicalNgram
from library.neural_arch import DualGranularityTransformer
from library.neural_dataset import NormalizationDataset, NormalizationCollator
from library.neural_trainer import ModelTrainer
from library.inference_engine import HybridRouter


def create_mini_datasets(input_train_path, input_test_path, output_dir):
    """
    Creates small subsets of the data for demonstration purposes.
    """
    print(f"Creating mini datasets in {output_dir}...")
    ensure_dir(os.path.join(output_dir, "placeholder"))

    # Load a small chunk of training data
    # We take enough rows to ensure we have some sentence boundaries
    df_train_full = pd.read_csv(input_train_path, nrows=5000)

    # Split into train/val for the demo (simple split by index for speed)
    split_idx = int(len(df_train_full) * 0.8)
    df_train = df_train_full.iloc[:split_idx].copy()
    df_val = df_train_full.iloc[split_idx:].copy()

    # Load a small chunk of test data
    df_test = pd.read_csv(input_test_path, nrows=500)

    # Define paths
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # Save
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def main():
    # 1. Setup
    seed_everything(42)

    # Define directories
    # We use a specific subdirectory for this demo run
    demo_working_dir = "./working/demo_run_output"
    demo_meta_dir = "./working/demo_metadata_output"

    # Clean up previous runs if they exist
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_meta_dir):
        shutil.rmtree(demo_meta_dir)

    # 2. Create Mini Datasets
    # We read from the provided metadata location
    orig_train_path = "./metadata/train.csv"
    orig_test_path = "./metadata/test.csv"

    mini_train_path, mini_val_path, mini_test_path = create_mini_datasets(
        orig_train_path, orig_test_path, demo_meta_dir
    )

    # 3. Initialize Configuration
    # We override paths to point to our mini datasets
    # We also reduce model size and training duration for speed
    config = Config(
        train_data_path=mini_train_path,
        val_data_path=mini_val_path,
        test_data_path=mini_test_path,
        working_dir=demo_working_dir,
        # Reduced Hyperparameters
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        batch_size=16,
        epochs=1,  # Just one epoch to prove it runs
        bpe_vocab_size=1000,  # Small vocab for small data
        max_char_vocab_size=100,
        debug=True,
    )

    print(f"Run Hash: {config.get_run_hash()}")

    # 4. Data Management & Preprocessing
    dm = DataManager(config)

    # 4a. Reconstruct Sentences (Demonstration)
    df_sent = dm.reconstruct_sentences("train", load_cached_data=False)
    print(f"Reconstructed {len(df_sent)} sentences from mini train set.")

    # 4b. Prepare Neural Sequences
    # This creates the context windows (Left/Right) for the model
    print("Preparing neural sequences...")
    df_train_seq = dm.prepare_neural_sequences("train", load_cached_data=False)
    df_val_seq = dm.prepare_neural_sequences("val", load_cached_data=False)

    # 5. Tokenizers
    print("Training tokenizers...")
    tokenizer = HybridTokenizer(config)
    # We pass the raw dataframe (loaded via helper) to train
    raw_train_df = pd.read_csv(mini_train_path, dtype=object)
    # Handle NaNs as in utils
    raw_train_df["before"] = raw_train_df["before"].fillna("")
    raw_train_df["after"] = raw_train_df["after"].fillna("")

    tokenizer.train_tokenizers(raw_train_df, load_cached_data=False)

    # Validation: Check if tokenizer works
    test_enc = tokenizer.encode(["hello"], "123", ["world"])
    assert "context_left_ids" in test_enc
    assert "target_char_ids" in test_enc
    print("Tokenizer validation passed.")

    # 6. Symbolic Model (N-gram Stats)
    print("Building Symbolic Model stats...")
    symbolic_model = HierarchicalNgram(config)
    symbolic_model.build_stats(train_df=raw_train_df, load_cached_data=False)

    # Validation: Check if stats file exists
    assert os.path.exists(
        config.ngram_stats_path.replace(".npy", "")
        + "_trigram_stats_"
        + config.get_run_hash()
        + ".parquet"
    )
    print("Symbolic model stats built.")

    # 7. Neural Model Training
    print("Initializing Neural Model...")

    # Create Datasets and Loaders
    train_dataset = NormalizationDataset(df_train_seq, tokenizer, config)
    val_dataset = NormalizationDataset(df_val_seq, tokenizer, config)

    collator = NormalizationCollator(
        bpe_pad_id=tokenizer.pad_token_id,
        char_pad_id=tokenizer.char_tokenizer.pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator
    )

    # Initialize Architecture
    model = DualGranularityTransformer(
        config,
        bpe_pad_id=tokenizer.pad_token_id,
        char_pad_id=tokenizer.char_tokenizer.pad_token_id,
    )

    # Initialize Trainer
    trainer = ModelTrainer(config, model, tokenizer)

    # Run Training
    print("Starting Training Loop...")
    trainer.train(train_loader, val_loader)

    # Validation: Check if model checkpoint exists
    assert os.path.exists(config.model_checkpoint_path)
    print("Model training complete and checkpoint saved.")

    # 8. Inference Engine (Hybrid Router)
    print("Initializing Inference Engine...")

    # The router loads the saved artifacts (tokenizers, stats, model) based on config
    router = HybridRouter(config)

    # Generate Submission
    # This will process the mini test set created earlier
    router.generate_submission()

    # 9. Final Validation
    submission_path = config.submission_path
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission generated with {len(df_sub)} rows.")
        print("Head of submission:")
        print(df_sub.head())

        # Verify format
        assert "id" in df_sub.columns
        assert "after" in df_sub.columns
        # Verify we have predictions for the test set size
        # Note: The test set might have fewer rows if some tokens were filtered out or handled differently,
        # but in this pipeline, we output for every row in input test csv.
        # df_test in create_mini_datasets had 500 rows.
        assert len(df_sub) == 500
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
