import os
import torch
import numpy as np
import random
import pandas as pd
import cv2
import warnings

# Import from the provided library
from library.config import Config
from library.tokenizer import Tokenizer
from library.dataset import get_loaders, BmsDataset, get_transforms
from library.model import Seq2Seq
from library.utils import LevenshteinMetric, AverageMeter
from library.engine import fit, inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def demonstrate_components():
    print("\n=== Demonstrating Individual Components ===")

    # 1. Configuration
    print("1. Initializing Configuration...")
    # Initialize config with debug=True to use defaults for debugging
    # We will further override these for an extremely fast demo
    config = Config(debug=True, epochs=1, batch_size=4, image_size=128)
    config.subset_size = 20  # Use only 20 samples for component verification
    config.num_workers = 0  # Main process only for simple demo
    config.print_config()

    # 2. Tokenizer
    print("\n2. Testing Tokenizer...")
    tokenizer = Tokenizer(config)
    # Load or build vocab (relies on metadata existing)
    tokenizer.load_or_build_vocab(load_cached_data=True)

    # Test round-trip conversion
    test_inchi = "InChI=1S/H2O/h1H2"
    sequence = tokenizer.text_to_sequence(test_inchi)
    decoded_text = tokenizer.sequence_to_text(sequence)

    print(f"Original: {test_inchi}")
    print(f"Encoded Sequence Shape: {sequence.shape}")
    print(f"Decoded: {decoded_text}")

    # Verification
    # Note: sequence_to_text stops at EOS. Our simple test string doesn't have EOS manually added
    # but text_to_sequence adds SOS and EOS.
    # The tokenizer logic handles SOS/EOS stripping.
    assert decoded_text == test_inchi, "Tokenizer round-trip failed!"
    print("Tokenizer verification passed.")

    # 3. Dataset and DataLoaders
    print("\n3. Testing Dataset and DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(config, tokenizer)

    # Fetch a batch
    images, labels, lengths = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")  # Should be (B, 3, H, W)
    print(f"Batch Labels Shape: {labels.shape}")  # Should be (B, max_len)

    assert images.shape == (
        config.batch_size,
        3,
        config.image_size[0],
        config.image_size[1],
    ), "Image batch shape mismatch"
    assert labels.shape == (
        config.batch_size,
        config.max_len,
    ), "Label batch shape mismatch"
    print("DataLoader verification passed.")

    # 4. Model
    print("\n4. Testing Model Forward Pass...")
    device = config.device
    vocab_size = len(tokenizer)
    model = Seq2Seq(config, vocab_size=vocab_size).to(device)

    images = images.to(device)
    labels = labels.to(device)

    # Forward pass (Training mode with teacher forcing)
    outputs = model(images, text=labels, teacher_forcing_ratio=0.5)
    print(f"Model Output Shape: {outputs.shape}")  # (B, max_len, vocab_size)

    assert outputs.shape == (
        config.batch_size,
        config.max_len,
        vocab_size,
    ), "Model output shape mismatch"

    # Inference mode (Predict)
    preds = model.predict(images)
    print(f"Prediction Shape: {preds.shape}")  # (B, max_len)
    assert preds.shape == (
        config.batch_size,
        config.max_len,
    ), "Prediction shape mismatch"

    print("Model verification passed.")

    # 5. Metrics
    print("\n5. Testing Metrics...")
    metric = LevenshteinMetric()
    # Dummy predictions and targets
    preds_str = ["InChI=1S/C", "InChI=1S/H"]
    targets_str = ["InChI=1S/C", "InChI=1S/O"]  # Distance 0 and 1

    metric.update(preds_str, targets_str)
    score = metric.compute()
    print(f"Levenshtein Score (Expected 0.5): {score}")
    assert abs(score - 0.5) < 1e-6, "Metric computation incorrect"
    print("Metric verification passed.")


def demonstrate_full_pipeline():
    print("\n=== Demonstrating Full Pipeline (Train + Inference) ===")

    # Configure for a very short run
    config = Config(debug=True)
    config.epochs = 1
    config.batch_size = 8
    config.subset_size = 50  # Train on 50 samples
    config.num_workers = 2
    config.image_size = (128, 128)  # Smaller images for speed
    config.print_config()

    # Run Training
    print("\n--- Starting Training Loop ---")
    best_score = fit(config)
    print(f"Training completed. Best Validation Score: {best_score}")

    # Check if checkpoint exists
    assert os.path.exists(config.checkpoint_path), "Checkpoint file was not created!"

    # Run Inference
    print("\n--- Starting Inference Loop ---")
    inference(config)

    # Check submission file
    if os.path.exists(config.submission_path):
        df_sub = pd.read_csv(config.submission_path)
        print(f"Submission file created at {config.submission_path}")
        print(f"Submission shape: {df_sub.shape}")
        print("First 3 rows:")
        print(df_sub.head(3))

        assert (
            df_sub.shape[0] == config.subset_size
        ), f"Submission rows {df_sub.shape[0]} != subset size {config.subset_size}"
        assert (
            "image_id" in df_sub.columns and "InChI" in df_sub.columns
        ), "Submission columns missing"
    else:
        raise FileNotFoundError("Submission file not found after inference!")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    try:
        # 1. Verify individual components logic
        demonstrate_components()

        # 2. Run the integrated pipeline (Train -> Eval -> Save -> Infer)
        demonstrate_full_pipeline()

        print("\nSUCCESS: All demonstrations and validations completed successfully.")

    except Exception as e:
        print(f"\nFAILURE: An error occurred during execution: {e}")
        raise e
