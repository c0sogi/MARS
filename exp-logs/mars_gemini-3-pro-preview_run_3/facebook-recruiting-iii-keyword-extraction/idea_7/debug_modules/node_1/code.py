import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
# Note: We import Config first to patch it before other modules use it extensively
from library.config import Config
from library.utils import set_seed, calculate_f1_samples, optimize_f1_threshold
from library.data_processing import process_data, TokenizerHandler, TargetEncoder
from library.model import DilatedWideAndDeep
from library.train_eval import run_pipeline


def setup_demo_config():
    """
    Overrides Config parameters to run a fast, small-scale demo.
    """
    print(">>> Setting up Demo Configuration...")

    # 1. Enable Debug Mode and reduce data size
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500  # Small subset for speed

    # 2. Reduce Training intensity
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.PATIENCE = 1

    # 3. Reduce Model/Data complexity for speed
    Config.VOCAB_SIZE = 1000
    Config.TOP_K_TAGS = 20  # Only predict top 20 tags
    Config.EMBED_DIM = 32
    Config.NUM_FILTERS = 16
    Config.MAX_LEN = 64  # Shorter sequence length

    # 4. Redirect paths to a demo working directory
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # We must manually update dependent paths because they were initialized
    # when the module was first imported.
    Config.TOKENIZER_PATH = os.path.join(Config.WORKING_DIR, "tokenizer.json")
    Config.MLB_PATH = os.path.join(Config.WORKING_DIR, "mlb.joblib")

    Config.TRAIN_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "train_tokens.npy")
    Config.TRAIN_LABELS_PATH = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "val_tokens.npy")
    Config.VAL_LABELS_PATH = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "test_tokens.npy")
    Config.TEST_IDS_PATH = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")


def test_utils():
    """
    Validates utility functions.
    """
    print("\n>>> Testing Utility Functions...")

    # Test F1 Calculation
    # Case: Perfect match
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 1], [0, 1, 0]])
    f1 = calculate_f1_samples(y_true, y_pred)
    assert np.isclose(f1, 1.0), f"Expected F1=1.0, got {f1}"

    # Case: No match
    y_pred_wrong = np.array([[0, 1, 0], [1, 0, 1]])
    f1_wrong = calculate_f1_samples(y_true, y_pred_wrong)
    assert np.isclose(f1_wrong, 0.0), f"Expected F1=0.0, got {f1_wrong}"

    print("calculate_f1_samples: Passed")

    # Test Threshold Optimization
    # Create probabilities where 0.6 is a clear cutoff
    y_probs = np.array([[0.8, 0.2], [0.3, 0.9]])
    y_target = np.array([[1, 0], [0, 1]])

    best_thr, best_score = optimize_f1_threshold(y_target, y_probs, step=0.05)

    # Threshold should be somewhere between 0.3 and 0.8.
    # Usually around 0.5 for this clear separation.
    assert 0.3 < best_thr < 0.8, f"Optimal threshold {best_thr} out of expected range"
    assert np.isclose(best_score, 1.0), f"Expected perfect score, got {best_score}"

    print("optimize_f1_threshold: Passed")


def test_data_processing():
    """
    Validates data processing pipeline components.
    """
    print("\n>>> Testing Data Processing...")

    # 1. Run process_data (force reload to test logic)
    # This will generate files in the demo working dir
    (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        test_ids,
        tokenizer,
        encoder,
    ) = process_data(load_cached_data=False)

    # 2. Validate Shapes
    print(f"Train Tokens Shape: {train_tokens.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    assert len(train_tokens) == len(train_labels), "Train tokens and labels mismatch"
    assert train_tokens.shape[1] == Config.MAX_LEN, f"Expected seq len {Config.MAX_LEN}"
    assert (
        train_labels.shape[1] == Config.TOP_K_TAGS
    ), f"Expected {Config.TOP_K_TAGS} classes"

    # 3. Validate Tokenizer
    test_text = ["python pandas"]
    encoded = tokenizer.encode(test_text)
    assert encoded.shape == (1, Config.MAX_LEN), "Tokenizer encoding shape error"
    assert np.any(encoded > 0), "Tokenizer produced all zeros (padding)"

    # 4. Validate Target Encoder
    # Create dummy tags that we know exist in the top K (assuming 'java' or 'c#' are common)
    # However, since we reduced TOP_K_TAGS to 20, we should check what classes exist
    print(f"Encoder Classes (First 5): {encoder.classes_[:5]}")

    # Test transform/inverse transform
    dummy_tags = [encoder.classes_[:2]]  # Take first two valid tags
    binary = encoder.transform(dummy_tags)
    assert binary.shape == (1, Config.TOP_K_TAGS)
    assert binary.sum() == 2, "Expected 2 active tags"

    inversed = encoder.inverse_transform(binary)
    # inversed is a list of tuples
    assert set(inversed[0]) == set(dummy_tags[0]), "Inverse transform mismatch"

    print("Data Processing: Passed")
    return train_tokens, train_labels


def test_model(train_tokens, train_labels):
    """
    Validates Model instantiation and forward pass.
    """
    print("\n>>> Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check

    num_classes = train_labels.shape[1]
    vocab_size = Config.VOCAB_SIZE

    model = DilatedWideAndDeep(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=Config.EMBED_DIM,
        num_filters=Config.NUM_FILTERS,
        kernel_size=Config.KERNEL_SIZE,
        dilation_rates=Config.DILATION_RATES,
        dropout=Config.DROPOUT,
    ).to(device)

    # Create dummy batch
    batch_size = 4
    dummy_input = torch.from_numpy(train_tokens[:batch_size]).long().to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (
        batch_size,
        num_classes,
    ), f"Expected output shape {(batch_size, num_classes)}, got {logits.shape}"

    assert not torch.isnan(logits).any(), "Model produced NaNs"

    print("Model Architecture: Passed")


def test_full_pipeline():
    """
    Runs the full training and inference pipeline using the library function.
    """
    print("\n>>> Testing Full Pipeline (Train/Eval/Predict)...")

    # run_pipeline handles loading data, training, and generating submission
    # We use load_cached_data=True because we generated data in test_data_processing
    run_pipeline(load_cached_data=True)

    # Validate Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Rows: {len(df_sub)}")
    print(df_sub.head(2))

    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if IDs match the test ids stored in cache
    test_ids = np.load(Config.TEST_IDS_PATH)
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count {len(df_sub)} != Test IDs count {len(test_ids)}"

    print("Full Pipeline: Passed")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Setup
    setup_demo_config()

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data Processing
    # We capture return values to use in model test
    tokens, labels = test_data_processing()

    # 4. Verify Model
    test_model(tokens, labels)

    # 5. Verify Full Pipeline
    test_full_pipeline()

    print("\n>>> All Demonstrations Completed Successfully.")
