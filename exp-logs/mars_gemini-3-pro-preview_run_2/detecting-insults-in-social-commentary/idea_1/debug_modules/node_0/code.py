import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.utils import set_seed, clean_text
from library.vocabulary import Vocabulary, build_vocabulary
from library.dataset import InsultDataset, collate_fn, get_dataloaders
from library.model import NeuralBoE, train_model
from library.trainer import run_training_pipeline
from library.config import CACHE_DIR, SUBMISSION_PATH


def test_vocabulary_logic():
    print("\n=== Testing Vocabulary Logic ===")

    # 1. Test Cleaning
    raw_text = '"Hello World! \\u00e9"'  # Unicode encoded e-acute
    cleaned = clean_text(raw_text)
    print(f"Original: {raw_text}")
    print(f"Cleaned:  {cleaned}")
    # Basic assertion: quotes removed, lowercase
    assert "hello" in cleaned.lower(), "Cleaning failed to preserve content"

    # 2. Test Fitting
    texts = ["hello world", "hello python", "world of code"]
    vocab = Vocabulary()
    vocab.fit(texts, min_freq=1)

    print(f"Vocab size: {len(vocab.stoi)}")
    assert "hello" in vocab.stoi
    assert "world" in vocab.stoi
    assert "python" in vocab.stoi

    # 3. Test Transform
    indices = vocab.transform("hello python unknown")
    print(f"Indices for 'hello python unknown': {indices}")

    # Check that 'unknown' gets mapped to UNK token (index 1)
    unk_idx = vocab.stoi["<UNK>"]
    assert indices[-1] == unk_idx, "Unknown token not mapped to UNK index"

    # 4. Test Save/Load
    test_cache = os.path.join(CACHE_DIR, "test_vocab")
    vocab.save(test_cache)

    vocab_loaded = Vocabulary()
    vocab_loaded.load(test_cache)
    assert vocab_loaded.stoi == vocab.stoi, "Loaded vocabulary does not match saved one"
    print("Vocabulary save/load verified.")


def test_dataset_and_model_logic():
    print("\n=== Testing Dataset and Model Logic ===")

    # Mock Data
    # Indices: [2, 3] and [4]
    data = pd.DataFrame({"indices": [[2, 3, 4], [5, 6]], "Insult": [0, 1]})

    # 1. Dataset
    dataset = InsultDataset(data)
    assert len(dataset) == 2
    item0_indices, item0_label = dataset[0]
    assert torch.equal(item0_indices, torch.tensor([2, 3, 4], dtype=torch.long))
    assert item0_label == 0.0

    # 2. DataLoader & Collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    text, offsets, labels = next(iter(loader))

    print(f"Batch Text Shape: {text.shape}")  # Should be (5,) -> 3 + 2 tokens
    print(f"Batch Offsets: {offsets}")  # Should be [0, 3]
    print(f"Batch Labels: {labels}")

    assert text.shape[0] == 5
    assert offsets.tolist() == [0, 3]
    assert torch.equal(labels, torch.tensor([0.0, 1.0]))

    # 3. Model Forward Pass
    vocab_size = 10
    embed_dim = 8
    hidden_dim = 4

    model = NeuralBoE(vocab_size, embed_dim, hidden_dim)
    output = model(text, offsets)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    assert (
        0.0 <= output.min() and output.max() <= 1.0
    ), "Output probabilities out of range [0, 1]"
    print("Dataset and Model logic verified.")


def test_full_pipeline():
    print("\n=== Testing Full Training Pipeline ===")

    # We will run the pipeline with minimal epochs to ensure speed.
    # We force `load_cached_data=False` to ensure the pipeline builds from scratch
    # using the provided metadata files.

    # Clean up previous submission if exists
    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)

    try:
        run_training_pipeline(
            load_cached_data=False,
            epochs=1,  # Run only 1 epoch for speed
            batch_size=32,
            learning_rate=0.01,
            patience=1,
            embed_dim=32,
            hidden_dim=32,
            device="cpu",  # Use CPU for this quick test to avoid overhead/setup issues
        )
    except Exception as e:
        raise AssertionError(f"Pipeline execution failed: {e}")

    # Verify submission file creation
    if not os.path.exists(SUBMISSION_PATH):
        raise AssertionError(f"Submission file was not created at {SUBMISSION_PATH}")

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file created with {len(df_sub)} rows.")

    # Check values
    assert "Insult" in df_sub.columns
    assert df_sub["Insult"].min() >= 0
    assert df_sub["Insult"].max() <= 1
    print("Pipeline execution verified.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Run demonstrations
    test_vocabulary_logic()
    test_dataset_and_model_logic()
    test_full_pipeline()

    print("\nAll tests passed successfully.")
