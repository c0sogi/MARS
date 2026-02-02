import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library import utils
from library import vocab
from library import data
from library import model
from library import engine


def run_demonstration():
    print("=== Starting Demonstration of Library Modules ===")

    # --- 1. Configuration for Speed ---
    print("\n[1] Configuring parameters for rapid demonstration...")
    # Modify Config class attributes directly to limit resource usage
    Config.DEBUG_SAMPLE_SIZE = 100  # Limit to 100 samples for vocab/dataset
    Config.VOCAB_SIZE = 500  # Small vocab
    Config.EMBEDDING_DIM = 32  # Small embedding dim
    Config.HIDDEN_SIZE = 32  # Small hidden size
    Config.BATCH_SIZE = 10  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.MAX_Q_LEN = 10  # Short sequences
    Config.MAX_C_LEN = 50  # Short sequences

    # Ensure reproducibility
    utils.set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 2. Vocabulary & Embeddings ---
    print("\n[2] Building Vocabulary and Embedding Matrix...")

    # Force rebuild to ensure it uses our small debug sample size
    # We pass load_cached_data=False to ignore any existing large vocab files
    vocabulary = vocab.build_vocab(Config.TRAIN_DATA_PATH, load_cached_data=False)

    # Verify Vocabulary
    assert len(vocabulary) > 2, "Vocabulary should contain at least PAD and UNK tokens"
    assert vocabulary.lookup_token(Config.PAD_TOKEN) == 0
    assert vocabulary.lookup_token(Config.UNK_TOKEN) == 1
    print(f"Vocabulary size: {len(vocabulary)}")

    # Build Embedding Matrix
    embedding_matrix = vocab.build_embedding_matrix(
        vocabulary, glove_path=None, load_cached_data=False  # Use random initialization
    )

    # Verify Embeddings
    expected_shape = (len(vocabulary), Config.EMBEDDING_DIM)
    assert (
        embedding_matrix.shape == expected_shape
    ), f"Embedding shape mismatch. Expected {expected_shape}, got {embedding_matrix.shape}"
    print("Embedding matrix built successfully.")

    # --- 3. Data Loading ---
    print("\n[3] Initializing Dataset and DataLoader...")

    # Check if metadata exists (it should based on the prompt)
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_META_PATH}")

    # Initialize Dataset
    train_dataset = data.NQDataset(
        metadata_path=Config.TRAIN_META_PATH,
        vocab=vocabulary,
        mode="train",
        debug_limit=Config.DEBUG_SAMPLE_SIZE,
    )

    val_dataset = data.NQDataset(
        metadata_path=Config.VAL_META_PATH,
        vocab=vocabulary,
        mode="val",
        debug_limit=Config.DEBUG_SAMPLE_SIZE,
    )

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=data.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data.collate_fn,
    )

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    if not sample_batch:
        print(
            "Warning: Empty batch received (possibly due to aggressive filtering in dataset)."
        )
    else:
        print("Sample batch keys:", sample_batch.keys())
        assert "q_input_ids" in sample_batch
        assert "c_input_ids" in sample_batch
        assert sample_batch["q_input_ids"].shape[1] == Config.MAX_Q_LEN
        assert sample_batch["c_input_ids"].shape[1] == Config.MAX_C_LEN
        print("Data loading verified.")

    # --- 4. Model Initialization ---
    print("\n[4] Initializing SiameseBiLSTM Model...")
    model_instance = model.SiameseBiLSTM(
        embedding_matrix=embedding_matrix,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Verify Forward Pass
    if sample_batch:
        with torch.no_grad():
            outputs = model_instance(
                sample_batch["q_input_ids"].to(device),
                sample_batch["c_input_ids"].to(device),
            )

        # Check output shapes
        batch_dim = sample_batch["q_input_ids"].size(0)
        assert outputs["rank_score"].shape == (batch_dim,)
        assert outputs["span_start_logits"].shape == (batch_dim, Config.MAX_C_LEN)
        assert outputs["yn_logits"].shape == (batch_dim, 3)
        print("Model forward pass verified.")

    # --- 5. Training Loop ---
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = optim.Adam(model_instance.parameters(), lr=Config.LEARNING_RATE)

    train_loss = engine.train_one_epoch(
        model=model_instance,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=1,
    )
    assert isinstance(train_loss, float)
    print("Training epoch completed.")

    # --- 6. Evaluation ---
    print("\n[6] Running Evaluation...")
    val_loss, metrics = engine.evaluate(
        model=model_instance, dataloader=val_loader, device=device
    )

    assert isinstance(val_loss, float)
    assert "rank_acc" in metrics
    assert "span_em" in metrics
    assert "yn_acc" in metrics
    print("Evaluation completed.")

    # --- 7. Checkpointing ---
    print("\n[7] Testing Checkpoint Save/Load...")
    ckpt_path = os.path.join(Config.WORKING_DIR, "demo_checkpoint.pth")

    utils.save_checkpoint(
        model=model_instance,
        optimizer=optimizer,
        epoch=1,
        loss=val_loss,
        path=ckpt_path,
    )

    assert os.path.exists(ckpt_path), "Checkpoint file was not created"

    loaded_ckpt = utils.load_checkpoint(ckpt_path, model_instance, optimizer)
    assert loaded_ckpt is not None
    assert loaded_ckpt["epoch"] == 1
    print("Checkpoint mechanism verified.")

    # --- 8. Utility Verification ---
    print("\n[8] Verifying Metric Utilities...")
    # Exact Match
    em_score = utils.compute_exact_match("hello world", "hello world")
    assert em_score == 1.0
    em_score_fail = utils.compute_exact_match("hello", "world")
    assert em_score_fail == 0.0

    # F1 Score (Span overlap)
    # Overlap: {2, 3}
    f1_score = utils.compute_f1((1, 4), (2, 5))
    # Pred: [1, 2, 3], Gold: [2, 3, 4] -> Common: [2, 3] (2 items)
    # Precision: 2/3, Recall: 2/3 -> F1: 2/3 approx 0.666
    assert 0.6 < f1_score < 0.7
    print("Metric utilities verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
