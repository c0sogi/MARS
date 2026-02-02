import os
import torch
import numpy as np
import random
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.text_utils import TextUtils
from library.data_factory import DataFactory, get_dataloaders
from library.neural_ranker import SiameseGatedConvRanker
from library.training_engine import TrainingEngine
from library.answer_extractor import SlidingWindowExtractor, detect_yes_no
from library.inference_pipeline import predict_and_format


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_text_utils():
    print("\n--- 1. Demonstrating TextUtils ---")

    # Test Tokenization
    text = "Hello, world! This is a test."
    tokens = TextUtils.tokenize(text)
    print(f"Original: '{text}'")
    print(f"Tokenized: {tokens}")
    assert tokens == ["hello", ",", "world", "!", "this", "is", "a", "test"]

    # Test Vocab Building (Mock data)
    texts = ["hello world", "hello python", "world of code"]
    vocab = TextUtils.build_vocab(texts, min_freq=1, load_cached_data=False)
    print(f"Vocab size: {len(vocab)}")
    assert "hello" in vocab
    assert TextUtils.PAD_TOKEN in vocab

    # Test Text to Indices
    indices = TextUtils.text_to_indices("hello code", vocab, max_len=5)
    print(f"Indices for 'hello code': {indices}")
    assert len(indices) == 5
    assert indices[0] == vocab["hello"]
    # 'code' appears once in texts, min_freq=1, so it should be in vocab
    assert indices[1] == vocab["code"]
    assert indices[2] == TextUtils.PAD_INDEX


def demo_data_loading_and_model(device):
    print("\n--- 2. Demonstrating Data Loading & Neural Model ---")

    # Override Config for Speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Small subset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # 1. Build Vocab from actual training data (subset)
    # We read the raw text to build vocab quickly for the demo
    print("Building vocabulary from training data subset...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH).head(Config.DEBUG_SIZE)

    # We need to fetch the actual text to build a meaningful vocab
    # Using DataFactory.build_file_index to get offsets
    file_index = DataFactory.build_file_index(
        Config.TRAIN_DATA_PATH, load_cached_data=True
    )

    sample_texts = []
    with open(Config.TRAIN_DATA_PATH, "rb") as f:
        for ex_id in train_df["example_id"].astype(str):
            if ex_id in file_index:
                f.seek(file_index[ex_id])
                import json

                entry = json.loads(f.readline())
                sample_texts.append(entry["question_text"])
                sample_texts.append(entry["document_text"])

    vocab = TextUtils.build_vocab(sample_texts, min_freq=2, load_cached_data=False)
    print(f"Built vocab size: {len(vocab)}")

    # 2. Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(vocab, debug=True)

    # 3. Initialize Model
    # We skip loading GloVe to save time/bandwidth, using random init
    embeddings = TextUtils.load_glove_embeddings(
        vocab, glove_path=None, load_cached_data=False
    )
    model = SiameseGatedConvRanker(len(vocab), embeddings).to(device)

    # 4. Test Forward Pass with a Batch
    print("Testing forward pass...")
    batch = next(iter(train_loader))
    q_idx = batch["q_indices"].to(device)
    c_idx = batch["c_indices"].to(device)
    labels = batch["labels"].to(device)

    # Handle flattening logic normally done in training loop
    if c_idx.dim() == 3:
        b, n, l = c_idx.shape
        q_idx_flat = q_idx.unsqueeze(1).expand(-1, n, -1).reshape(-1, q_idx.size(1))
        c_idx_flat = c_idx.reshape(-1, l)

        logits = model(q_idx_flat, c_idx_flat)
        print(f"Logits shape: {logits.shape}")
        assert logits.shape[0] == b * n

    return model, train_loader, val_loader, test_loader, vocab


def demo_training(model, train_loader, val_loader, device):
    print("\n--- 3. Demonstrating Training Engine ---")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    engine = TrainingEngine(model, device, optimizer)

    # Train one epoch
    print("Training for one epoch...")
    loss = engine.train_one_epoch(train_loader)
    print(f"Epoch Loss: {loss:.4f}")
    assert not np.isnan(loss)

    # Evaluate
    print("Evaluating...")
    val_loss, val_acc = engine.evaluate(val_loader)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    assert not np.isnan(val_loss)


def demo_answer_extractor():
    print("\n--- 4. Demonstrating Answer Extractor ---")

    extractor = SlidingWindowExtractor(window_size=5, threshold=1)

    question = "who is the president"
    long_answer = "The current president is John Doe of the United States."

    # Expected: "president is John Doe" or similar depending on tokenization
    start, end, text = extractor.extract(question, long_answer)

    print(f"Question: {question}")
    print(f"Long Answer: {long_answer}")
    print(f"Extracted: '{text}' (indices: {start}-{end})")

    assert start != -1
    assert "president" in text or "John" in text

    # Test Yes/No detection
    print(f"Detect Yes/No ('Yes, he is'): {detect_yes_no('Yes, he is')}")
    print(f"Detect Yes/No ('No'): {detect_yes_no('No')}")
    print(f"Detect Yes/No ('John Doe'): {detect_yes_no('John Doe')}")

    assert detect_yes_no("yes") == "YES"
    assert detect_yes_no("no, never") == "NO"
    assert detect_yes_no("maybe") == "NONE"


def demo_inference(model, test_loader, vocab, device):
    print("\n--- 5. Demonstrating Inference Pipeline ---")

    output_csv = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Run prediction
    predict_and_format(
        model=model,
        test_loader=test_loader,
        vocab=vocab,
        device=device,
        jsonl_path=Config.TEST_DATA_PATH,
        output_path=output_csv,
    )

    # Verify output
    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    print(f"Submission shape: {df.shape}")
    print(df.head())
    assert "example_id" in df.columns
    assert "PredictionString" in df.columns


if __name__ == "__main__":
    # Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Text Utils
    demo_text_utils()

    # 2. Data & Model
    model, train_loader, val_loader, test_loader, vocab = demo_data_loading_and_model(
        device
    )

    # 3. Training
    demo_training(model, train_loader, val_loader, device)

    # 4. Answer Extraction
    demo_answer_extractor()

    # 5. Inference
    demo_inference(model, test_loader, vocab, device)

    print("\nAll demonstrations completed successfully.")
