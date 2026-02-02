import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import json

# Import from the provided library
from library.config import Config, set_seed
from library.utils import (
    is_semiotic,
    save_json,
    load_json,
    save_checkpoint,
    load_checkpoint,
)
from library.data_factory import (
    CharTokenizer,
    TransformerDataset,
    _add_context_and_filter,
)
from library.hfbb_model import HFBBEngine
from library.transformer_model import Seq2SeqTransformer, create_mask
from library.hybrid_system import HybridPredictor


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup Environment and Overrides
    # We create a separate working directory for the demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.HFBB_CACHE_DIR = os.path.join(DEMO_DIR, "hfbb_cache")
    Config.TRANSFORMER_CACHE_DIR = os.path.join(DEMO_DIR, "transformer_cache")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")
    Config.MODEL_CHECKPOINT = os.path.join(DEMO_DIR, "seq2seq_demo.pth")
    Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.json")

    # Override Model Hyperparameters for rapid CPU execution
    Config.EMBED_DIM = 16
    Config.HIDDEN_DIM = 32
    Config.NUM_LAYERS = 1
    Config.NUM_HEADS = 2  # Must divide EMBED_DIM
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEVICE = "cpu"  # Force CPU for demo stability/simplicity

    # Set seed
    set_seed(42)

    # ==========================================
    # 2. Demonstrate Utils
    # ==========================================
    print("\n--- Testing Utils ---")

    # Test is_semiotic
    assert is_semiotic("123") is True, "is_semiotic failed for digits"
    assert is_semiotic("abc") is False, "is_semiotic failed for letters"
    assert is_semiotic("10kg") is True, "is_semiotic failed for mixed"

    # Test JSON I/O
    test_data = {"key": "value", "list": [1, 2, 3]}
    json_path = os.path.join(DEMO_DIR, "test.json")
    save_json(test_data, json_path)
    loaded_data = load_json(json_path)
    assert loaded_data == test_data, "JSON save/load mismatch"
    print("Utils verified.")

    # ==========================================
    # 3. Demonstrate Data Factory
    # ==========================================
    print("\n--- Testing Data Factory ---")

    # A. Tokenizer
    texts = ["hello", "world", "123"]
    tokenizer = CharTokenizer()
    tokenizer.fit_on_texts(texts)

    encoded = tokenizer.encode("hello", add_special_tokens=True)
    decoded = tokenizer.decode(encoded, remove_special_tokens=True)

    assert decoded == "hello", f"Tokenizer roundtrip failed: {decoded} != hello"
    assert len(encoded) == len("hello") + 2, "Tokenizer did not add SOS/EOS"

    # Save/Load Tokenizer
    tokenizer.save(Config.VOCAB_PATH)
    tokenizer_loaded = CharTokenizer()
    tokenizer_loaded.load(Config.VOCAB_PATH)
    assert len(tokenizer.char2idx) == len(
        tokenizer_loaded.char2idx
    ), "Tokenizer load size mismatch"
    print("Tokenizer verified.")

    # B. Context Generation & Filtering
    # Create dummy dataframe
    data = {
        "sentence_id": [0, 0, 0, 1, 1],
        "token_id": [0, 1, 2, 0, 1],
        "before": ["The", "123", "end", "Next", "sent"],
        "after": ["the", "one two three", "end", "next", "sent"],
        "class": ["PLAIN", "CARDINAL", "PLAIN", "PLAIN", "PLAIN"],
    }
    df_raw = pd.DataFrame(data)

    # Process
    df_proc = _add_context_and_filter(df_raw, is_train=True, load_cached_data=False)

    # Verification
    # Row 1 (token_id 1 in sent 0) is "123". It is semiotic.
    # Prev should be "The", Next should be "end".
    row_semiotic = df_proc[df_proc["before"] == "123"].iloc[0]
    assert row_semiotic["prev"] == "The", "Context 'prev' incorrect"
    assert row_semiotic["next"] == "end", "Context 'next' incorrect"

    # Row 0 ("The") is not semiotic, should be filtered out because is_train=True
    assert "The" not in df_proc["before"].values, "Filtering non-semiotic failed"

    print("Data processing verified.")

    # C. Dataset
    dataset = TransformerDataset(df_proc, tokenizer)
    sample = dataset[0]

    assert "input_ids" in sample
    assert "labels" in sample
    assert sample["original_text"] == "123"
    assert sample["input_ids"].dtype == torch.long
    print("Dataset verified.")

    # ==========================================
    # 4. Demonstrate HFBB Model (Tier 1)
    # ==========================================
    print("\n--- Testing HFBB Engine ---")

    # Create synthetic training data for HFBB
    # Pattern: "prev_A" "curr_A" "next_A" -> "norm_A"
    hfbb_data = {
        "sentence_id": [0, 0, 0],
        "token_id": [0, 1, 2],
        "before": ["prev_A", "curr_A", "next_A"],
        "after": ["prev_A", "norm_A", "next_A"],
        "class": ["PLAIN", "TEST", "PLAIN"],
    }
    df_hfbb = pd.DataFrame(hfbb_data)

    # Save to a temporary CSV and update Config to point to it
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    df_hfbb.to_csv(mini_train_path, index=False)
    df_hfbb.to_csv(mini_val_path, index=False)  # Reuse for val

    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path

    engine = HFBBEngine()
    engine.fit(load_cached_data=False)

    # Query Trigram
    res, level = engine.query("curr_A", "prev_A", "next_A")
    assert res == "norm_A", f"HFBB Trigram lookup failed. Got {res}"
    assert level == "trigram", f"HFBB Level incorrect. Got {level}"

    # Query Unigram (should fallback if context doesn't match but token does)
    # Note: Our dummy data has only one occurrence.
    # If we query with unknown context, it should hit unigram if unigram map was built.
    res_uni, level_uni = engine.query("curr_A", "unknown_prev", "unknown_next")
    assert res_uni == "norm_A", "HFBB Unigram fallback failed"
    assert level_uni == "unigram", f"HFBB Level incorrect. Got {level_uni}"

    print("HFBB Engine verified.")

    # ==========================================
    # 5. Demonstrate Transformer Model (Tier 2)
    # ==========================================
    print("\n--- Testing Transformer Model ---")

    vocab_size = len(tokenizer)
    model = Seq2SeqTransformer(
        num_encoder_layers=Config.NUM_LAYERS,
        num_decoder_layers=Config.NUM_LAYERS,
        emb_size=Config.EMBED_DIM,
        nhead=Config.NUM_HEADS,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dim_feedforward=Config.HIDDEN_DIM,
        dropout=0.0,
    )

    # Dummy Input: Batch=2, Seq=10
    src = torch.randint(0, vocab_size, (2, 10))
    tgt = torch.randint(0, vocab_size, (2, 8))

    # Create masks
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
        src, tgt, Config.PAD_IDX, "cpu"
    )

    # Forward pass
    logits = model(
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        src_padding_mask,
    )

    assert logits.shape == (
        2,
        8,
        vocab_size,
    ), f"Transformer output shape mismatch: {logits.shape}"
    print("Transformer forward pass verified.")

    # ==========================================
    # 6. Demonstrate Hybrid System (Integration)
    # ==========================================
    print("\n--- Testing Hybrid System ---")

    predictor = HybridPredictor(device="cpu")

    # A. Train Systems (Mocking the loop)
    # We use the mini_train.csv created earlier.
    # It contains "curr_A" (token_id 1). "curr_A" is not semiotic (no digits).
    # To test Transformer training, we need semiotic tokens in training data.

    train_data_semiotic = {
        "sentence_id": [0, 0, 0],
        "token_id": [0, 1, 2],
        "before": ["start", "999", "end"],
        "after": ["start", "nine nine nine", "end"],
        "class": ["PLAIN", "CARDINAL", "PLAIN"],
    }
    df_train_sem = pd.DataFrame(train_data_semiotic)
    df_train_sem.to_csv(mini_train_path, index=False)
    df_train_sem.to_csv(mini_val_path, index=False)

    # Force retrain to trigger the Trainer loop
    predictor.train_systems(load_cached_data=False, epochs=1, force_retrain=True)

    assert (
        predictor.transformer is not None
    ), "Transformer model not initialized after training"
    assert predictor.hfbb.is_fitted is True, "HFBB not fitted"

    # B. Generate Submission
    # Create a test file
    # Row 0: "curr_A" -> Should be handled by HFBB (if we kept the old HFBB map, but we refitted on new data)
    # Actually, we refitted HFBB on `df_train_sem` ("999").
    # So "999" should be handled by HFBB Trigram.
    # Let's add a new semiotic token "888" that is NOT in training.
    # HFBB will fail (return None). It is semiotic, so it goes to Transformer.

    test_data = {
        "sentence_id": [0, 0, 0],
        "token_id": [0, 1, 2],
        "before": ["start", "888", "end"],
    }
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")
    pd.DataFrame(test_data).to_csv(mini_test_path, index=False)
    Config.TEST_CSV = mini_test_path

    # Run inference
    predictor.generate_submission(load_cached_data=False)

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # We expect 3 rows (0_0, 0_1, 0_2)
    assert (
        len(df_sub) == 3
    ), f"Submission length mismatch. Expected 3, got {len(df_sub)}"

    # Check IDs
    expected_ids = ["0_0", "0_1", "0_2"]
    assert df_sub["id"].tolist() == expected_ids, "Submission IDs mismatch"

    # Check content logic
    # 0_0: "start" -> HFBB Trigram match (from training data) -> "start"
    # 0_1: "888" -> HFBB miss -> Semiotic -> Transformer -> Prediction (likely garbage due to 1 epoch/tiny data, but string)
    # 0_2: "end" -> HFBB Trigram match -> "end"

    pred_0 = df_sub[df_sub["id"] == "0_0"]["after"].iloc[0]
    assert pred_0 == "start", f"Expected identity/HFBB match for 'start', got {pred_0}"

    pred_1 = df_sub[df_sub["id"] == "0_1"]["after"].iloc[0]
    assert isinstance(pred_1, str), "Prediction for 888 is not a string"
    assert len(pred_1) > 0, "Prediction for 888 is empty"

    print("Hybrid System verified.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
