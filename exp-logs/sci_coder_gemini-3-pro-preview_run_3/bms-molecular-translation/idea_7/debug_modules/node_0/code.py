import os
import sys
import torch
import numpy as np
import random
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_dataloader, ChemicalDataset
from library.model import Seq2Seq
from library.trainer import Trainer
from library.utils import LevenshteinMetric


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_demo_config():
    """Overrides Config for a fast demonstration run."""
    print("--- Setting up Demo Configuration ---")

    # Use a specific directory for this demo to avoid overwriting existing work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"

    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update artifact paths since they were defined at import time
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "checkpoint.pth")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 32  # Very small dataset for quick verification
    Config.PRINT_FREQ = 1  # Print frequently to show progress

    # Device
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")


def demonstrate_tokenizer():
    """Verifies Tokenizer functionality."""
    print("\n--- Demonstrating Tokenizer ---")

    # Force rebuild of vocab for the demo to ensure it works with the current metadata
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)

    tokenizer = Tokenizer(load_cached_data=False)

    # Test string
    test_inchi = "InChI=1S/H2O/h1H2"
    print(f"Original: {test_inchi}")

    # Encode
    encoded = tokenizer.encode(test_inchi)
    print(f"Encoded Tensor Shape: {encoded.shape}")
    print(f"Encoded Tensor (first 10): {encoded[:10]}")

    # Assertions
    assert encoded.dim() == 1, "Encoded output should be 1D tensor"
    assert (
        len(encoded) == Config.MAX_LENGTH
    ), f"Encoded length should be {Config.MAX_LENGTH}"
    assert encoded[0] == tokenizer.SOS_IDX, "First token should be SOS"

    # Decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Verify round-trip (ignoring special tokens logic inside decode)
    assert (
        decoded == test_inchi
    ), f"Decoded string '{decoded}' does not match original '{test_inchi}'"
    print("Tokenizer verification passed.")
    return tokenizer


def demonstrate_dataset_and_loader(tokenizer):
    """Verifies Dataset and DataLoader."""
    print("\n--- Demonstrating Dataset & DataLoader ---")

    # Create dataloader in debug mode
    loader = get_dataloader(
        Config.TRAIN_METADATA,
        tokenizer,
        batch_size=Config.BATCH_SIZE,
        mode="train",
        debug=True,
    )

    print(f"Number of batches: {len(loader)}")

    # Fetch one batch
    images, labels, label_lengths = next(iter(loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape[0] == Config.BATCH_SIZE
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == Config.IMAGE_HEIGHT
    assert images.shape[3] == Config.IMAGE_WIDTH
    assert labels.shape[1] == Config.MAX_LENGTH

    print("Dataset and DataLoader verification passed.")
    return loader


def demonstrate_model(tokenizer, loader):
    """Verifies Model architecture and forward pass."""
    print("\n--- Demonstrating Seq2Seq Model ---")

    vocab_size = len(tokenizer)
    model = Seq2Seq(vocab_size).to(Config.DEVICE)

    # Get a batch
    images, labels, _ = next(iter(loader))
    images = images.to(Config.DEVICE)
    labels = labels.to(Config.DEVICE)

    # Forward pass (Training mode with teacher forcing)
    model.train()
    outputs = model(images, labels, teacher_forcing_ratio=1.0)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, Config.MAX_LENGTH, vocab_size)

    # Inference pass (Predict mode)
    model.eval()
    predictions = model.predict(images)

    print(f"Inference Prediction Shape: {predictions.shape}")
    assert predictions.shape == (Config.BATCH_SIZE, Config.MAX_LENGTH)

    print("Model verification passed.")


def demonstrate_metric():
    """Verifies Levenshtein Metric calculation."""
    print("\n--- Demonstrating Metric ---")

    metric = LevenshteinMetric()
    preds = ["InChI=1S/H2O/h1H2", "InChI=1S/CH4"]
    targets = ["InChI=1S/H2O/h1H2", "InChI=1S/CH4/h1H4"]  # 2nd one differs

    metric.update(preds, targets)
    score = metric.get_avg_score()

    print(f"Metric Score: {score}")
    # Distance for 1st pair is 0.
    # Distance for 2nd pair ("InChI=1S/CH4", "InChI=1S/CH4/h1H4") is len("/h1H4") = 5.
    # Avg = 2.5
    assert abs(score - 2.5) < 1e-5, f"Expected score 2.5, got {score}"
    print("Metric verification passed.")


def demonstrate_training_pipeline():
    """Runs the full training loop using the Trainer class."""
    print("\n--- Demonstrating Full Training Pipeline ---")

    # Initialize Trainer
    # This handles Tokenizer, Model, Optimizer, and Checkpoint loading internally
    trainer = Trainer(load_cached_data=True)

    # Run fit (Training + Validation)
    # Using the debug configuration set in setup_demo_config
    trainer.fit(epochs=Config.EPOCHS, debug=Config.DEBUG)

    # Run prediction on test set
    trainer.predict_test(debug=Config.DEBUG)

    # Verify artifacts
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file not found"
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    assert len(df_sub) > 0, "Submission file is empty"

    print("Training pipeline verification passed.")


if __name__ == "__main__":
    try:
        set_seed(42)
        setup_demo_config()

        # 1. Tokenizer
        tokenizer = demonstrate_tokenizer()

        # 2. Data
        loader = demonstrate_dataset_and_loader(tokenizer)

        # 3. Model
        demonstrate_model(tokenizer, loader)

        # 4. Metric
        demonstrate_metric()

        # 5. Full Trainer
        demonstrate_training_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Raise to ensure the process exits with error code if something fails
        raise e
