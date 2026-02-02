import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import pandas as pd

# Ensure the current directory is in the python path to import library modules
sys.path.append(".")

# Import from the provided library files
from library.config import Config
from library.utils import levenshtein_distance
from library.tokenizer import Tokenizer
from library.dataset import InChiDataset, get_transforms
from library.model import ResNetTCN
from library.trainer import Trainer


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def run_demo():
    print("--- Starting Library Demonstration ---")
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # -------------------------------------------------------------------------
    print("[1/7] Configuring environment...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only load 50 samples

    # Reduce training parameters
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Main process only for stability in demo

    # Reduce Sequence Length for faster processing
    Config.MAX_LEN = 30

    # Simplify Model for faster initialization and forward pass
    # Reduce TCN depth and disable pretrained weights to avoid downloads
    Config.TCN_NUM_CHANNELS = [256, 256]
    Config.ENCODER_PRETRAINED = False

    # Set paths to a temporary working directory
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.json")
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "model_best.pth")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    print("Configuration updated for demo execution.")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("[2/7] Verifying Utils...")

    str1, str2 = "kitten", "sitting"
    dist = levenshtein_distance(str1, str2)
    # Distance: k->s (s), i (i), t (t), t (t), e->i (s), n (n), +g (i) = 3 edits
    assert dist == 3, f"Levenshtein distance incorrect. Expected 3, got {dist}"
    print(f"Levenshtein distance between '{str1}' and '{str2}' is {dist} (Correct).")

    # -------------------------------------------------------------------------
    # 3. Verify Tokenizer
    # -------------------------------------------------------------------------
    print("[3/7] Verifying Tokenizer...")

    tokenizer = Tokenizer()
    # Force build from metadata (reads train_metadata.csv)
    # Note: This reads the full CSV but is fast enough (~seconds)
    tokenizer.build_vocab(load_cached_data=False)

    # Test Roundtrip
    test_inchi = "InChI=1S/H2O/h1H2"
    seq = tokenizer.text_to_sequence(test_inchi)

    assert isinstance(seq, torch.Tensor), "Tokenizer output should be a Tensor"
    assert (
        len(seq) == Config.MAX_LEN
    ), f"Sequence length {len(seq)} != Config.MAX_LEN {Config.MAX_LEN}"

    decoded_text = tokenizer.sequence_to_text(seq)
    assert (
        decoded_text == test_inchi
    ), f"Tokenizer roundtrip failed. Got: {decoded_text}"

    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print("Tokenizer logic verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Dataset
    # -------------------------------------------------------------------------
    print("[4/7] Verifying Dataset...")

    train_transform = get_transforms("train")
    # Initialize dataset (will sample 50 items due to DEBUG=True)
    dataset = InChiDataset(Config.TRAIN_METADATA, tokenizer, transform=train_transform)

    assert (
        len(dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset size mismatch: {len(dataset)}"

    # Check item structure
    image, label = dataset[0]

    # Image should be (3, 256, 256) based on Config.IMAGE_SIZE
    assert image.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape incorrect: {image.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a Tensor"
    assert len(label) == Config.MAX_LEN, "Label sequence length incorrect"

    print("Dataset loading and transformation verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("[5/7] Verifying Model...")

    device = Config.DEVICE
    vocab_size = tokenizer.get_vocab_size()
    model = ResNetTCN(vocab_size)
    model = model.to(device)

    # Create dummy inputs
    dummy_images = torch.randn(
        Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)
    dummy_captions = torch.randint(
        0, vocab_size, (Config.BATCH_SIZE, Config.MAX_LEN)
    ).to(device)

    # Forward pass
    outputs = model(dummy_images, dummy_captions)

    # Expected output: (Batch, SeqLen, VocabSize)
    expected_shape = (Config.BATCH_SIZE, Config.MAX_LEN, vocab_size)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape {outputs.shape} != {expected_shape}"

    print("Model architecture and forward pass verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Training Step
    # -------------------------------------------------------------------------
    print("[6/7] Verifying Training Step...")

    # Setup Trainer components
    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, shuffle=False
    )  # Reuse for demo

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_IDX)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, tokenizer
    )

    # Manually run one training step to ensure gradients flow and loss calculates
    model.train()

    # Get one batch
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    # Prepare inputs/targets (autoregressive)
    inputs = labels[:, :-1]
    targets = labels[:, 1:]

    # Forward
    outputs = model(images, inputs)

    # Loss calculation
    vocab_size = outputs.size(2)
    loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "Loss is NaN"
    print(f"Single training step completed. Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 7. Verify Inference / Decoding
    # -------------------------------------------------------------------------
    print("[7/7] Verifying Inference Decoding...")

    # Use the trainer's batched_predict method which implements greedy decoding
    model.eval()
    with torch.no_grad():
        predictions = trainer.batched_predict(images)

    assert (
        len(predictions) == Config.BATCH_SIZE
    ), "Number of predictions matches batch size"
    assert isinstance(predictions[0], str), "Prediction is not a string"

    print(f"Sample Prediction: {predictions[0]}")
    print("Inference logic verified.")

    print("\n--- All Checks Passed Successfully ---")


if __name__ == "__main__":
    run_demo()
