import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
from transformers import AutoTokenizer, logging as transformers_logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.dataset import load_data, EssayDataset
from library.modeling import DebertaRegressor
from library.lexical import train_lexical_fold
from library.engine import train_fold, predict
from library.optimization import optimize_thresholds, apply_thresholds


def run_demonstration():
    # --- 1. Setup & Configuration ---
    print("--- 1. Setup & Configuration ---")
    seed_everything(42)

    # Override Config for a fast demonstration
    # We use the class directly as it is used globally in the library modules
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for speed
    Config.EPOCHS = 1  # Single epoch to verify loop
    Config.N_FOLDS = 1  # Only run one fold
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1

    # Create necessary directories
    Config.create_dirs()

    # Suppress verbose transformers logs
    transformers_logging.set_verbosity_error()

    print("Configuration updated: DEBUG=True, EPOCHS=1, Sample Size=50")

    # --- 2. Data Loading & Dataset ---
    print("\n--- 2. Data Loading & Dataset ---")
    # Load data (handles caching and debug slicing internally)
    train_df = load_data("train", load_cached_data=False)
    val_df = load_data("val", load_cached_data=False)
    test_df = load_data("test", load_cached_data=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Verify shapes match debug size
    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} training samples"
    assert (
        len(val_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} validation samples"

    # Initialize Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = EssayDataset(train_df, tokenizer)

    # Fetch one sample to verify structure
    sample = dataset[0]
    print("Dataset sample keys:", sample.keys())

    # Verify Dataset output
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "score" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert sample["input_ids"].ndim == 1
    # Check score is a float tensor (for regression)
    assert sample["score"].dtype == torch.float32

    # --- 3. Model Instantiation & Architecture Check ---
    print("\n--- 3. Model Instantiation & Architecture Check ---")
    device = Config.DEVICE
    # Instantiate model (pretrained=True to match engine behavior)
    model = DebertaRegressor(Config.MODEL_NAME, pretrained=True)
    model.to(device)
    model.eval()

    # Prepare batch (add batch dimension)
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    print(f"Model output shape: {output.shape}")

    # Verify output shape [Batch_Size, Num_Labels] -> [1, 1]
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"

    # Cleanup to save memory
    del model, input_ids, attention_mask
    torch.cuda.empty_cache()

    # --- 4. Lexical Branch (TF-IDF + Ridge) ---
    print("\n--- 4. Lexical Branch Execution ---")
    # Train the lexical model on the debug data
    lex_model, lex_val_preds, lex_test_preds = train_lexical_fold(
        train_df, val_df, test_df, fold_idx=0, load_cached_data=False
    )

    print(f"Lexical Val Preds Shape: {lex_val_preds.shape}")

    # Verify predictions
    assert len(lex_val_preds) == len(val_df)
    assert len(lex_test_preds) == len(test_df)
    assert isinstance(lex_val_preds, np.ndarray)

    # --- 5. Semantic Branch Training (Engine) ---
    print("\n--- 5. Semantic Branch Training (DeBERTa) ---")
    # Train for 1 epoch on 50 samples
    # This function saves the model to Config.OUTPUT_DIR/deberta_fold_0.bin
    best_qwk = train_fold(0, train_df, val_df)

    print(f"Training completed. Best Validation QWK: {best_qwk:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "deberta_fold_0.bin")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    # --- 6. Inference ---
    print("\n--- 6. Inference ---")
    # Predict using the trained semantic model
    sem_test_preds = predict(test_df, checkpoint_path)

    print(f"Semantic Test Preds Shape: {sem_test_preds.shape}")

    # Verify inference output
    assert len(sem_test_preds) == len(test_df)
    assert not np.isnan(sem_test_preds).any(), "Predictions contain NaNs"

    # --- 7. Threshold Optimization ---
    print("\n--- 7. Threshold Optimization ---")
    # Use lexical validation predictions for demonstration
    y_true = val_df["score"].values
    y_pred_continuous = lex_val_preds

    # Optimize thresholds
    opt_thresholds = optimize_thresholds(y_true, y_pred_continuous)
    print(f"Optimized Thresholds: {opt_thresholds}")

    # Apply thresholds to get integer scores
    final_preds = apply_thresholds(y_pred_continuous, opt_thresholds)
    print(f"Sample Integer Predictions: {final_preds[:10]}")

    # Verify constraints
    assert len(opt_thresholds) == 5, "Should return 5 thresholds for 6 classes"
    assert np.all(np.diff(opt_thresholds) >= 0), "Thresholds must be sorted"
    assert np.all(
        (final_preds >= 1) & (final_preds <= 6)
    ), "Predictions must be in range [1, 6]"

    # Calculate final QWK
    final_score = compute_qwk(y_true, final_preds)
    print(f"Final QWK Score: {final_score:.4f}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
