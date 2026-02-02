import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import (
    load_jsonl,
    format_prediction_string,
    load_ground_truth_data,
    compute_f1_score,
)
from library.preprocessing import Tokenizer, build_embedding_matrix
from library.dataset import NQDataset
from library.model import InteractionGridCNN
from library.engine import Trainer, set_seed


def main():
    print("--- Starting Demonstration Script ---")

    # 1. Override Configuration for Speed
    # We modify the Config class attributes directly to create a "micro" environment
    print("\n[1] Configuring environment for fast execution...")
    Config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples for vocab/dataset
    Config.VOCAB_SIZE = 1000  # Smaller vocab size for speed
    Config.NUM_EPOCHS = 1  # Just 1 epoch for demonstration
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EMBED_DIM = 32  # Smaller embeddings

    # Use a separate cache directory for this demo to avoid conflicts or long loading times
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Update derived cache paths in Config
    Config.VOCAB_CACHE_FILE = os.path.join(Config.CACHE_DIR, "vocab.npy")
    Config.EMBED_MATRIX_CACHE_FILE = os.path.join(
        Config.CACHE_DIR, "embedding_matrix.npy"
    )
    Config.TRAIN_FEATURES_CACHE = os.path.join(
        Config.CACHE_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_CACHE = os.path.join(Config.CACHE_DIR, "val_features.parquet")

    # 2. Preprocessing: Tokenizer and Embeddings
    print("\n[2] Testing Preprocessing (Tokenizer & Embeddings)...")

    # Initialize Tokenizer
    tokenizer = Tokenizer()
    # Force recompute (load_cached_data=False) to use our small sample size
    tokenizer.fit(
        Config.TRAIN_FILE, sample_size=Config.DEBUG_SAMPLE_SIZE, load_cached_data=False
    )

    print(f"Tokenizer fitted. Vocab size: {tokenizer.vocab_size}")
    assert (
        tokenizer.vocab_size > 2
    ), "Vocabulary should contain more than just special tokens."
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1

    # Test Encoding
    sample_text = "what is the capital of france"
    encoded = tokenizer.encode(sample_text, max_len=10)
    assert len(encoded) == 10, "Encoded sequence length mismatch."
    print(f"Encoded '{sample_text}': {encoded}")

    # Build Embedding Matrix
    embedding_matrix = build_embedding_matrix(tokenizer, load_cached_data=False)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    assert embedding_matrix.shape == (
        tokenizer.vocab_size,
        Config.EMBED_DIM,
    ), "Embedding matrix shape mismatch."

    # 3. Dataset and DataLoader
    print("\n[3] Testing Dataset and DataLoader...")

    # Create Train Dataset
    # We use load_cached_data=False to ensure we generate the small dataset now
    train_dataset = NQDataset(
        mode="train",
        tokenizer=tokenizer,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
        load_cached_data=False,
        expand_candidates=False,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print(
            "Warning: Train dataset is empty. Check if metadata matches input file IDs."
        )
    else:
        # Check one item
        item = train_dataset[0]
        print("Sample item keys:", item.keys())
        assert "q_ids" in item
        assert "c_ids" in item
        assert "rank_label" in item
        assert item["q_ids"].shape == (
            Config.MAX_Q_LEN,
        ), f"Question shape mismatch: {item['q_ids'].shape}"
        assert item["c_ids"].shape == (
            Config.MAX_C_LEN,
        ), f"Context shape mismatch: {item['c_ids'].shape}"

    # Create Validation Dataset
    val_dataset = NQDataset(
        mode="val",
        tokenizer=tokenizer,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
        load_cached_data=False,
        expand_candidates=False,
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoaders
    # Note: drop_last=False to ensure we get data even if len < batch_size
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False
    )

    # 4. Model Initialization and Forward Pass
    print("\n[4] Testing Model Initialization and Forward Pass...")

    model = InteractionGridCNN(embedding_matrix)
    model.to(Config.DEVICE)

    # Get a batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        q_batch = batch["q_ids"].to(Config.DEVICE)
        c_batch = batch["c_ids"].to(Config.DEVICE)

        # Forward pass
        outputs = model(q_batch, c_batch)

        print("Model output keys:", outputs.keys())
        assert "rank_logits" in outputs
        assert "start_logits" in outputs
        assert "end_logits" in outputs
        assert "yn_logits" in outputs

        # Check shapes
        B = q_batch.size(0)
        assert outputs["rank_logits"].shape == (
            B,
        ), f"Rank logits shape mismatch: {outputs['rank_logits'].shape}"
        assert outputs["start_logits"].shape == (
            B,
            Config.MAX_C_LEN,
        ), "Start logits shape mismatch"
        assert outputs["yn_logits"].shape == (
            B,
            Config.NUM_YN_CLASSES,
        ), "Yes/No logits shape mismatch"
    else:
        print("Skipping forward pass check due to empty dataloader.")

    # 5. Training Loop
    print("\n[5] Testing Training Loop...")

    if len(train_loader) > 0:
        trainer = Trainer(model, train_loader, val_loader)

        # Run training
        # Since we set epochs=1 and dataset is small, this should be fast
        trainer.train(epochs=Config.NUM_EPOCHS)

        # Check if model file was saved (might not be if validation loss doesn't improve, but path should exist)
        print(
            f"Training loop finished. Model save path configured as: {Config.MODEL_SAVE_PATH}"
        )
    else:
        print("Skipping training loop due to empty dataloader.")

    # 6. Utils and Metrics
    print("\n[6] Testing Utility Functions and Metrics...")

    # Test format_prediction_string
    pred_str = format_prediction_string(10, 20, "NONE")
    assert pred_str == "10:20"
    pred_str_yn = format_prediction_string(-1, -1, "YES")
    assert pred_str_yn == "YES"
    pred_str_invalid = format_prediction_string(20, 10, "NONE")  # start > end
    assert pred_str_invalid == ""
    print("Prediction string formatting verified.")

    # Test compute_f1_score with synthetic data
    # Create synthetic predictions
    preds_data = {
        "example_id": ["1_long", "1_short", "2_long", "2_short", "3_long", "3_short"],
        "PredictionString": ["10:20", "YES", "5:15", "", "", "NO"],
    }
    preds_df = pd.DataFrame(preds_data)

    # Create synthetic Ground Truth
    gt_data = {
        "example_id": ["1", "2", "3"],
        "valid_long": [["10:20"], ["5:15"], []],
        "valid_short": [["YES"], ["1:2"], ["NO"]],
    }
    gt_df = pd.DataFrame(gt_data)

    precision, recall, f1 = compute_f1_score(preds_df, gt_df)
    print(
        f"Computed Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}"
    )

    # Manual verification logic:
    # Long:
    # 1: Pred 10:20, GT 10:20 -> TP
    # 2: Pred 5:15, GT 5:15 -> TP
    # 3: Pred "", GT [] -> TN (ignored)
    # Short:
    # 1: Pred YES, GT YES -> TP
    # 2: Pred "", GT 1:2 -> FN
    # 3: Pred NO, GT NO -> TP
    # Totals: TP=4, FP=0, FN=1
    # Precision = 4/4 = 1.0
    # Recall = 4/5 = 0.8

    assert (
        abs(precision - 1.0) < 1e-6
    ), f"Precision mismatch: expected 1.0, got {precision}"
    assert abs(recall - 0.8) < 1e-6, f"Recall mismatch: expected 0.8, got {recall}"

    print("Metric calculation verified.")

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
