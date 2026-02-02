import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data as data
import library.model as model
import library.trainer as trainer
import library.inference as inference


def setup_demo_config():
    """
    Overrides default configuration to run a fast, lightweight demo.
    """
    print("--- Setting up Demo Configuration ---")

    # Define demo working directory
    demo_dir = "./working/demo_pipeline"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update directory paths in config
    config.WORKING_DIR = demo_dir
    config.SUBMISSION_DIR = demo_dir
    config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # Update artifact paths (these are originally defined at module level using the old WORKING_DIR)
    config.TOKENIZER_PATH = os.path.join(demo_dir, "tokenizer.json")
    config.LABEL_ENCODER_PATH = os.path.join(demo_dir, "mlb.joblib")
    config.TRAIN_FEATURES_PATH = os.path.join(demo_dir, "train_data.parquet")
    config.VAL_FEATURES_PATH = os.path.join(demo_dir, "val_data.parquet")
    config.TEST_FEATURES_PATH = os.path.join(demo_dir, "test_data.parquet")
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "model_demo.pth")

    # Update Data Hyperparameters for Speed
    config.DEBUG_SAMPLE_SIZE = 500  # Use only 500 samples
    config.VOCAB_SIZE = 1000  # Smaller vocab
    config.MAX_LEN = 64  # Shorter sequences
    config.TOP_K_TAGS = 50  # Fewer classes

    # Update Model Hyperparameters for Speed/Memory
    config.EMBED_DIM = 32
    config.CNN_FILTERS = 16
    config.TRANSFORMER_LAYERS = 1
    config.NUM_HEADS = 2
    config.TRANSFORMER_FF_DIM = 32
    config.DROPOUT = 0.0

    # Update Training Hyperparameters
    config.BATCH_SIZE = 16
    config.NUM_EPOCHS = 2
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    config.DEVICE = (
        "cpu"  # Force CPU for simple demo stability, or use cuda if preferred
    )
    if torch.cuda.is_available():
        config.DEVICE = "cuda"

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")
    print(f"Debug Sample Size: {config.DEBUG_SAMPLE_SIZE}")


def test_utils():
    """Verifies utility functions."""
    print("\n--- Testing Utilities ---")

    # Test F1 Score Calculation
    # Case: Perfect match
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred_logits = np.array(
        [[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]]
    )  # High probs for true classes

    # calculate_f1_score expects probabilities or logits?
    # Looking at utils.py: it takes y_pred and applies threshold directly.
    # So we pass probabilities.
    score = utils.calculate_f1_score(y_pred_logits, y_true, threshold=0.5)

    if score != 1.0:
        raise AssertionError(f"Expected F1 score 1.0, got {score}")

    print("Utils verification passed.")


def test_data_components():
    """Verifies Tokenizer and Encoder logic independently."""
    print("\n--- Testing Data Components ---")

    # 1. Test TextTokenizer
    texts = ["hello world", "hello python code"]
    tokenizer = data.TextTokenizer(vocab_size=10, max_len=5)
    tokenizer.fit(texts)

    # Check vocab
    assert "hello" in tokenizer.vocab
    assert tokenizer.vocab["<PAD>"] == 0

    # Check transform
    tokens = tokenizer.transform(["hello world"])
    assert tokens.shape == (1, 5)
    assert tokens[0][0] == tokenizer.vocab["hello"]
    assert tokens[0][2] == 0  # Padding

    # 2. Test TagEncoder
    tags_list = ["python java", "python c++"]
    encoder = data.TagEncoder(top_k=5)
    encoder.fit(tags_list)

    assert "python" in encoder.tag_to_idx

    # Check transform
    y = encoder.transform(["python"])
    idx = encoder.tag_to_idx["python"]
    assert y[0, idx] == 1.0
    assert y.shape == (1, len(encoder.classes_))

    print("Data components verification passed.")


def test_model_architecture():
    """Verifies Model Forward Pass."""
    print("\n--- Testing Model Architecture ---")

    batch_size = 4
    seq_len = config.MAX_LEN
    vocab_size = config.VOCAB_SIZE
    num_classes = config.TOP_K_TAGS

    # Instantiate Model
    net = model.HybridCNNTransformer(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        cnn_filters=config.CNN_FILTERS,
        cnn_kernel_size=config.CNN_KERNEL_SIZE,
        transformer_layers=config.TRANSFORMER_LAYERS,
        num_heads=config.NUM_HEADS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.DROPOUT,
        num_classes=num_classes,
        max_len=seq_len,
    )
    net.to(config.DEVICE)
    net.eval()

    # Create Dummy Input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        output = net(dummy_input)

    # Assertions
    expected_shape = (batch_size, num_classes)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    print("Model architecture verification passed.")


def run_pipeline_demo():
    """Runs the full training and inference pipeline using the library."""
    print("\n--- Running Full Pipeline Demo ---")

    # 1. Run Training
    # This will use the DEBUG_SAMPLE_SIZE set in setup_demo_config
    # load_cached_data=False forces re-processing for this demo run
    print("Step 1: Training...")
    trained_model, tokenizer, encoder = trainer.run_training(load_cached_data=False)

    # Verify Model Artifacts
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved.")

    # Manually save artifacts in demo mode because library/data.py skips saving when DEBUG_SAMPLE_SIZE is set
    tokenizer.save(config.TOKENIZER_PATH)
    encoder.save(config.LABEL_ENCODER_PATH)

    if not os.path.exists(config.TOKENIZER_PATH):
        raise FileNotFoundError("Tokenizer file was not saved.")

    print("Training completed and artifacts saved.")

    # 2. Run Inference
    # This loads the model we just trained and generates submission.csv
    print("Step 2: Inference...")
    inference.generate_submission(load_cached_data=True)

    # Verify Submission
    if not os.path.exists(config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(config.SUBMISSION_FILE)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Basic check on submission content
    if "Id" not in df_sub.columns or "Tags" not in df_sub.columns:
        raise AssertionError("Submission file missing required columns.")

    # Check if we have predictions for the test set (debug size)
    # Note: prepare_data applies debug size to test set as well
    if len(df_sub) != config.DEBUG_SAMPLE_SIZE:
        # Note: If test.csv has fewer rows than DEBUG_SAMPLE_SIZE, it will be len(test.csv).
        # But here we know test.csv is large.
        raise AssertionError(
            f"Expected {config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"
        )

    print("Pipeline demo completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    utils.set_seed(config.SEED)

    # 1. Setup Environment
    setup_demo_config()

    # 2. Unit Tests
    test_utils()
    test_data_components()
    test_model_architecture()

    # 3. Integration Test (Full Pipeline)
    run_pipeline_demo()
