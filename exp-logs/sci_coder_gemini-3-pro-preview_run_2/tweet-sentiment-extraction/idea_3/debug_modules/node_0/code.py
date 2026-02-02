import os
import torch
import pandas as pd
import numpy as np
import shutil
import warnings
import transformers

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import process_data, get_fold_dls
from library.model import TweetModel
from library.engine import train_fold
from library.inference import predict_test

# Suppress warnings and verbose logs for cleaner output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Sentiment Extraction Pipeline Demo ===\n")

    # 1. Configuration
    # Initialize config in debug mode for speed (samples=100, epochs=2)
    config = Config(debug=True)

    # Override output directory for this demo to keep it isolated
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)
    config.output_dir = demo_dir

    print(f"Configuration:")
    print(f"  Debug Mode: {config.debug}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  Device: {config.device}")

    # Set seeds for reproducibility
    seed_everything(config.seed)
    print("\n[Step 1] Environment configured and seeded.")

    # 2. Data Processing
    print("\n[Step 2] Processing Data...")
    # process_data handles loading metadata, tokenization, and caching
    (train_data, train_targets, train_meta), (test_data, test_meta) = process_data(
        config, load_cached_data=False
    )

    # Validation: Check shapes
    # In debug mode, sample size is 100
    expected_size = config.debug_sample_size
    seq_len = config.max_len

    print(f"  Validating processed data shapes...")
    assert train_data["input_ids"].shape == (
        expected_size,
        seq_len,
    ), f"Train input_ids shape mismatch. Expected ({expected_size}, {seq_len}), got {train_data['input_ids'].shape}"
    assert train_targets["start_idx"].shape == (
        expected_size,
    ), f"Train targets shape mismatch. Expected ({expected_size},), got {train_targets['start_idx'].shape}"

    print("  Data processing and validation successful.")

    # 3. Model Initialization & Forward Pass Check
    print("\n[Step 3] Initializing Model and Verifying Architecture...")
    model = TweetModel(config.model_name, config.dropout)
    model.to(config.device)
    model.eval()

    # Create a dummy batch to verify forward pass
    dummy_batch_size = 4
    dummy_input_ids = torch.randint(0, 1000, (dummy_batch_size, seq_len)).to(
        config.device
    )
    dummy_mask = torch.ones((dummy_batch_size, seq_len)).to(config.device)
    dummy_token_type = torch.zeros((dummy_batch_size, seq_len)).to(config.device)

    with torch.no_grad():
        start_logits, end_logits = model(dummy_input_ids, dummy_mask, dummy_token_type)

    # Validation: Output shapes
    assert start_logits.shape == (
        dummy_batch_size,
        seq_len,
    ), f"Start logits shape mismatch. Expected ({dummy_batch_size}, {seq_len}), got {start_logits.shape}"
    assert end_logits.shape == (
        dummy_batch_size,
        seq_len,
    ), f"End logits shape mismatch. Expected ({dummy_batch_size}, {seq_len}), got {end_logits.shape}"

    print("  Model architecture verification successful.")
    del model, dummy_input_ids, dummy_mask, dummy_token_type
    torch.cuda.empty_cache()

    # 4. Training Loop (Fold 0)
    print("\n[Step 4] Running Training Loop (Fold 0)...")
    # train_fold runs the training for specific epochs and saves the best model
    best_jaccard = train_fold(fold=0, config=config)

    print(f"  Training completed. Best Validation Jaccard: {best_jaccard:.4f}")

    # Validation: Check if model file exists
    expected_model_path = config.get_model_path(0)
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {expected_model_path}")
    print(f"  Model checkpoint verified at: {expected_model_path}")

    # 5. Inference
    print("\n[Step 5] Running Inference on Test Set...")
    # predict_test loads the saved models and generates submission.csv
    # Note: It iterates over config.n_folds. In debug mode n_folds=2.
    # We only trained fold 0, so fold 1 will be skipped with a warning (handled in library code).
    submission_df = predict_test(config)

    # Validation: Check submission file
    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not generated.")

    # Reload to verify content
    df_sub = pd.read_csv(submission_path)

    # Check columns
    required_cols = {"textID", "selected_text"}
    assert required_cols.issubset(
        df_sub.columns
    ), f"Submission missing columns. Found: {df_sub.columns}"

    # Check length (should match debug sample size for test set)
    assert (
        len(df_sub) == expected_size
    ), f"Submission length mismatch. Expected {expected_size}, got {len(df_sub)}"

    # Check for empty predictions (basic sanity check)
    empty_preds = df_sub["selected_text"].isna().sum()
    assert empty_preds == 0, f"Found {empty_preds} NaN values in prediction."

    print(f"  Submission file verified. Shape: {df_sub.shape}")
    print("  First 3 predictions:")
    print(df_sub.head(3).to_string(index=False))

    # Cleanup
    print("\n[Step 6] Cleaning up...")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    print("  Temporary files removed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
