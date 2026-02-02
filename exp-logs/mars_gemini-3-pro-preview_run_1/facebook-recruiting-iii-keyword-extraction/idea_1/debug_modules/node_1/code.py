import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_f1
from library.data_loader import TextTokenizer, TagEncoder, get_dataloaders
from library.model import BiGRUClassifier
from library.train import run_training
from library.predict import run_prediction

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Stack Exchange Tag Prediction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demonstration
    # -------------------------------------------------------------------------
    print("1. Configuring environment for rapid demonstration...")

    # Create a specific directory for this demo to isolate artifacts
    DEMO_DIR = os.path.join(Config.WORKING_DIR, "demo_run")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config class attributes directly
    Config.IDEA_DIR = DEMO_DIR
    Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.npy")
    Config.TAG_MAP_PATH = os.path.join(DEMO_DIR, "tag_map.npy")
    Config.TOKENIZER_PATH = os.path.join(DEMO_DIR, "tokenizer.json")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Small subset of data
    Config.BATCH_SIZE = 32
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.VOCAB_SIZE = 500  # Small vocabulary
    Config.NUM_TAGS = 20  # Predict only top 20 tags
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 1  # Single layer GRU
    Config.PREDICTION_THRESHOLD = 0.3

    print(f"   Working Directory: {Config.IDEA_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print("\n")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("2. Verifying Utility Functions...")

    # Test Seed
    seed_everything(42)
    rand1 = np.random.rand(5)
    seed_everything(42)
    rand2 = np.random.rand(5)
    assert np.allclose(
        rand1, rand2
    ), "Seed setting did not produce reproducible results."
    print("   [Pass] seed_everything reproducibility check.")

    # Test F1 Calculation
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    y_pred = np.array([[0, 1, 1], [1, 0, 0]])
    score = calculate_f1(y_true, y_pred, average="samples")
    assert score == 1.0, f"F1 calculation failed. Expected 1.0, got {score}"

    y_pred_bad = np.array([[1, 0, 0], [0, 1, 1]])
    score_bad = calculate_f1(y_true, y_pred_bad, average="samples")
    assert score_bad < 1.0, "F1 calculation failed for mismatched data."
    print("   [Pass] calculate_f1 logic check.")
    print("\n")

    # -------------------------------------------------------------------------
    # 3. Verify Data Processing Components (Unit Tests)
    # -------------------------------------------------------------------------
    print("3. Verifying Data Processing Components...")

    # Test TextTokenizer
    dummy_texts = ["python java code", "c++ code error", "java python"]
    tokenizer = TextTokenizer(vocab_size=10, min_freq=1)
    # Manually fit to avoid file caching logic for this unit test
    tokenizer.vocab = tokenizer._compute_vocab(dummy_texts)

    encoded = tokenizer.transform(dummy_texts)
    assert encoded.shape == (
        3,
        Config.MAX_LEN,
    ), f"Tokenizer output shape mismatch. Got {encoded.shape}"
    assert isinstance(encoded, np.ndarray), "Tokenizer output is not a numpy array."
    print("   [Pass] TextTokenizer fit and transform.")

    # Test TagEncoder
    dummy_tags = pd.Series(["python java", "c++", "java python"])
    encoder = TagEncoder(num_tags=3)
    # Manually fit
    encoder.tag_list = encoder._compute_tag_map(dummy_tags)
    encoder.tag_to_idx = {tag: i for i, tag in enumerate(encoder.tag_list)}

    tag_matrix = encoder.transform(dummy_tags)
    assert tag_matrix.shape == (
        3,
        3,
    ), f"TagEncoder output shape mismatch. Got {tag_matrix.shape}"

    # Test Inverse Transform
    # Create a probability matrix that perfectly matches the first row ("python java")
    # Assuming "python" and "java" are in the top 3.
    probs = np.zeros((1, 3))
    # Set high probability for all tags to see what comes back
    probs[0, :] = 0.9
    predicted_str = encoder.inverse_transform(probs, threshold=0.5)[0]
    # Should contain all tags in the map
    assert (
        len(predicted_str.split()) == 3
    ), "TagEncoder inverse_transform failed to retrieve all tags."
    print("   [Pass] TagEncoder fit, transform, and inverse_transform.")
    print("\n")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("4. Verifying Model Architecture...")

    model = BiGRUClassifier(
        vocab_size=100,
        embed_dim=32,
        hidden_dim=32,
        num_layers=1,
        num_classes=10,
        dropout=0.1,
    )

    # Create dummy input: Batch size 4, Seq len 300
    dummy_input = torch.randint(0, 100, (4, 300))

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        4,
        10,
    ), f"Model output shape mismatch. Expected (4, 10), got {logits.shape}"
    print("   [Pass] BiGRUClassifier forward pass and output shape.")
    print("\n")

    # -------------------------------------------------------------------------
    # 5. Execute Training Pipeline
    # -------------------------------------------------------------------------
    print("5. Executing Training Pipeline (Debug Mode)...")
    print("   Note: This involves reading the dataset and training for 1 epoch.")

    # This calls library.train.run_training which uses the Config we modified
    best_f1 = run_training(debug=True)

    # Validation
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    assert os.path.exists(Config.VOCAB_PATH), "Vocabulary cache was not saved."
    assert os.path.exists(Config.TAG_MAP_PATH), "Tag map cache was not saved."

    print(f"   [Pass] Training completed. Best F1: {best_f1:.4f}")
    print(f"   [Pass] Artifacts saved to {Config.IDEA_DIR}")
    print("\n")

    # -------------------------------------------------------------------------
    # 6. Execute Prediction Pipeline
    # -------------------------------------------------------------------------
    print("6. Executing Prediction Pipeline (Debug Mode)...")

    # This calls library.predict.run_prediction
    run_prediction(debug=True)

    # Validation
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "Id" in df_sub.columns and "Tags" in df_sub.columns
    ), "Submission file missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check format of Tags (should be string, space delimited or empty)
    # In debug mode with random weights, tags might be empty strings if threshold is high,
    # or random strings. Just check type.
    assert pd.api.types.is_string_dtype(df_sub["Tags"]) or pd.api.types.is_object_dtype(
        df_sub["Tags"]
    ), "Tags column is not string/object type."

    print("   [Pass] Prediction completed.")
    print(f"   [Pass] Submission file generated at {Config.SUBMISSION_PATH}")
    print("   Head of submission:")
    print(df_sub.head())
    print("\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
