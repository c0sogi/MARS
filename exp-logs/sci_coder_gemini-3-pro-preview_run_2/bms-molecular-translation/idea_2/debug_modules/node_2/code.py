import os
import sys
import torch
import pandas as pd
import numpy as np
import random
import cv2

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

import importlib
import library.trainer

importlib.reload(library.trainer)

from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset
from library.model import Image2Seq
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set to {seed}")


def demo_tokenizer(config):
    print("\n=== Demo: Tokenizer ===")
    # Load a small subset of training data for vocabulary building
    df_train = pd.read_csv(config.train_metadata_path).head(100)

    tokenizer = Tokenizer(config)
    # Force re-computation to ensure logic works without relying on existing cache
    if os.path.exists(config.tokenizer_cache_path):
        os.remove(config.tokenizer_cache_path)

    tokenizer.fit_on_texts(df_train["InChI"].values, load_cached_data=False)

    # Test string
    sample_inchi = df_train["InChI"].iloc[0]
    print(f"Original: {sample_inchi}")

    # Encode
    seq = tokenizer.text_to_sequence(sample_inchi)
    print(f"Encoded Sequence Shape: {seq.shape}")
    print(f"Encoded Sequence (first 10): {seq[:10]}")

    # Decode
    decoded = tokenizer.sequence_to_text(seq)
    print(f"Decoded: {decoded}")

    # Validation
    assert isinstance(seq, torch.Tensor), "Tokenizer output should be a tensor"
    assert (
        len(seq) == config.max_length
    ), f"Sequence length mismatch. Expected {config.max_length}, got {len(seq)}"
    # Note: Decoded string might stop at EOS, so exact length match isn't expected,
    # but the content should match the original up to the first special token logic.
    assert decoded == sample_inchi, "Decoded text does not match original input"
    print("Tokenizer verification passed.")
    return tokenizer


def demo_dataset(config, tokenizer):
    print("\n=== Demo: Dataset ===")
    df_train = pd.read_csv(config.train_metadata_path).head(50)

    # Initialize Dataset
    dataset = InChiDataset(df_train, tokenizer, config, mode="train")

    # Fetch one sample
    image, label = dataset[0]

    print(f"Image Shape: {image.shape}")
    print(f"Label Shape: {label.shape}")

    # Validation
    assert image.shape == (
        3,
        config.image_size,
        config.image_size,
    ), f"Image shape mismatch. Expected (3, {config.image_size}, {config.image_size}), got {image.shape}"
    assert label.shape == (
        config.max_length,
    ), f"Label shape mismatch. Expected ({config.max_length},), got {label.shape}"
    assert isinstance(image, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    print("Dataset verification passed.")


def demo_model(config, tokenizer):
    print("\n=== Demo: Model ===")
    vocab_size = tokenizer.vocab_size
    model = Image2Seq(config, vocab_size)

    # Move to device
    device = config.device
    model.to(device)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, config.image_size, config.image_size).to(
        device
    )
    dummy_captions = torch.randint(0, vocab_size, (batch_size, config.max_length)).to(
        device
    )

    # Forward pass (Training mode signature)
    print("Running forward pass...")
    outputs = model(dummy_images, dummy_captions)
    print(f"Output Logits Shape: {outputs.shape}")

    # Validation: Output should be (Batch, Max_Len - 1, Vocab_Size)
    expected_shape = (batch_size, config.max_length - 1, vocab_size)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"

    # Prediction pass (Inference mode signature)
    print("Running inference prediction...")
    predictions = model.predict(dummy_images, tokenizer)
    print(f"Predictions type: {type(predictions)}")
    print(f"Number of predictions: {len(predictions)}")

    assert len(predictions) == batch_size, "Number of predictions matches batch size"
    assert isinstance(predictions[0], str), "Prediction items should be strings"
    print("Model verification passed.")


def demo_trainer(config):
    print("\n=== Demo: Trainer ===")
    # Initialize Trainer
    trainer = Trainer(config)

    # Setup (loads data, builds tokenizer, initializes model)
    trainer.setup()

    # Run training loop
    # Config is in debug mode, so it uses very small subsets and few epochs
    print("Starting training loop (Debug Mode)...")
    trainer.fit()

    # Validate artifacts
    assert os.path.exists(
        config.best_model_path
    ), "Best model checkpoint was not saved."
    print(f"Checkpoint saved at {config.best_model_path}")

    # Run prediction
    print("Starting prediction loop (Debug Mode)...")
    trainer.predict()

    assert os.path.exists(config.submission_path), "Submission file was not created."
    df_sub = pd.read_csv(config.submission_path)
    print(f"Submission file created with {len(df_sub)} rows.")
    print("Trainer verification passed.")


if __name__ == "__main__":
    # 1. Setup Configuration
    # Enable debug mode to limit dataset size (1000 train, 200 val) and epochs (2)
    config = Config(debug=True)

    # Further optimize for speed in this demonstration script
    config.epochs = 1
    config.batch_size = 8  # Small batch size for demonstration
    config.num_workers = 2  # Reduce workers overhead

    set_seed(config.seed)

    print(f"Device: {config.device}")
    print(f"Working Directory: {config.working_dir}")

    # 2. Demonstrate Components
    tokenizer = demo_tokenizer(config)
    demo_dataset(config, tokenizer)
    demo_model(config, tokenizer)

    # 3. Demonstrate Full Training Cycle
    demo_trainer(config)

    print("\nAll demonstrations completed successfully.")
